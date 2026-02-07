import pickle
import os
import dataBase.Logger as Logger
import numpy as np
import subprocess
import math
import torch
from torch_geometric.data import Data
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import shutil
import multiprocessing as mp
import uuid



def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def get_hyperedges(cond_val):
    # exclude macros from hyperedges
    hyp_e = {}
    _, E = cond_val.edge_index.shape
    insts = cond_val.edge_index.T[:E//2, :]
    # inst_pins = cond_val.edge_pin_id[:E//2, :] if "edge_pin_id" in cond_val else cond_val.edge_attr[:E//2, :] 

    # # We add 1 to all ids, such that ids are 1 -> num(nodes) inclusive
    # # exclude macros and ports from hyperedges
    # for i, p in zip(insts, inst_pins):
    #     u, v = i.tolist()
    #     u_id = tuple(p[:len(p)//2].tolist())
    #     if (u, u_id) not in hyp_e:
    #         hyp_e[(u, u_id)] = []
    #     if not (cond_val.is_macros[u] or cond_val.is_ports[u]):
    #         hyp_e[(u, u_id)] += [u+1]
    #     if not (cond_val.is_macros[v] or cond_val.is_ports[v]):
    #         hyp_e[(u, u_id)] += [v+1]

    for eid, (u, v) in enumerate(insts.tolist()):
        hyp_e[eid] = []

        if not (cond_val.is_macros[u] or cond_val.is_ports[u]):
            hyp_e[eid].append(u + 1)

        if not (cond_val.is_macros[v] or cond_val.is_ports[v]):
            hyp_e[eid].append(v + 1)
    
    # remove hyperedges with <= 1 nodes
    output_hyperedges = {}
    for k, v in hyp_e.items():
        if len(v) > 1:
            output_hyperedges[k] = v
    return output_hyperedges

def run_hmetis(j, input_cond, num_clusters, algorithm = "hmetis", ubfactor=5, temp_dir = "logs/temp", verbose = False):
    """
    algorithm is one of "hmetis", "shmetis", "khmetis"
    """
    hyp_e = get_hyperedges(input_cond)
    try:
        os.makedirs(temp_dir)
    except FileExistsError:
        pass
    
    salt = np.random.randint(1e12) # so we can run operations in parallel
    hmetis_input_file = os.path.join(temp_dir, f'edges-{salt}.txt')
    # input file
    # first line is number of hyperedges and number of vertices
    # i th line (excluding comment lines) contains the vertices that are included in the (i−1)th hyperedge
    with open(hmetis_input_file, 'w') as fp:
        fp.write(f'{len(hyp_e)} {input_cond.x.shape[0]}\n')
        for k in hyp_e:
            fp.write(' '.join(map(str, hyp_e[k])) + '\n')

    hmetis_output_file = f'{hmetis_input_file}.part.{num_clusters}'  # 'logs/temp/edges-286505397008.txt.part.512'

    if algorithm == "hmetis":
        # hmetis HGraphFile Nparts UBfactor Nruns CType Rtype Vcycle Reconst debuglevel
        nruns = 25 # 10 is shmetis default
        ctype = 1 # shmetis default
        rtype = 1 # shemtis default
        vcycle = 3 # 1 is shmetis default; consider using 3 for slower but better
        reconst = 1 # 0 is shmetis default; 1 is reconstruct partial hyperedges
        dbglvl = 0 # 0 is no debugging; 31 is for everything
        ret = subprocess.run([
            './hmetis', 
            hmetis_input_file,  # 'logs/temp/edges-286505397008.txt'
            str(num_clusters),  # 512
            str(int(ubfactor)), # 5
            str(nruns),  # 25
            str(ctype), 
            str(rtype),
            str(vcycle),
            str(reconst),
            str(dbglvl),
            ], capture_output = not verbose)
        if ret.returncode != 0:
            print("hmetis failed:")
            print(ret.stdout)
            print(ret.stderr)
            raise RuntimeError("hmetis failed")

        if not os.path.exists(hmetis_output_file):
            print(f"job:{j} Output not generated: {hmetis_output_file}")
            print("input file exists:", os.path.exists(hmetis_input_file))
            raise FileNotFoundError(hmetis_output_file)
    else:
        raise NotImplementedError

    # hmetis_output_file = f'{hmetis_input_file}.part.{num_clusters}'  # 'logs/temp/edges-286505397008.txt.part.512'
    with open(hmetis_output_file, 'r') as fp:
        assigned_parts = list(map(int, fp.readlines()))
    assert len(assigned_parts) == input_cond.x.shape[0], f"error parsing shmetis output. expected lines {input_cond.x.shape[0]} but got {len(assigned_parts)}"
    
    # clean up hmetis i/o files
    os.remove(hmetis_input_file)
    os.remove(hmetis_output_file)
    
    return assigned_parts

def cluster(j, input_cond, num_clusters, ubfactor=5, temp_dir = "logs/temp", verbose = False, placements = None, algorithm = "hmetis"):
    """
    placements should be (B, V, 2) tensor
    Note: outputs will be on same device as inputs
    """

    # move to cpu for clustering
    if placements is None:
        placements = torch.zeros_like(input_cond.x).unsqueeze(dim=0) # (B, V, 2)
    placements_device = placements.device
    cond_device = input_cond.x.device
    placements.to(device = "cpu")
    input_cond.to(device = "cpu")
    _, E = input_cond.edge_index.shape
    assert E % 2 == 0, "cond edge index assumed to contain forward and reverse edges"

    # perform clustering
    if algorithm in ("shmetis", "hmetis", "khmetis"):
        assigned_parts = run_hmetis(j, input_cond, num_clusters, algorithm = algorithm, ubfactor=ubfactor, temp_dir = temp_dir, verbose = verbose)
    else:
        raise NotImplementedError

    # create new cond_val
    B, _, _ = placements.shape
    components = list(zip(input_cond.x, assigned_parts, input_cond.is_ports, input_cond.is_macros, range(len(input_cond.is_ports))))
    data_x = []
    data_ports = [torch.tensor(False) for _ in range(num_clusters)]
    data_macros = [torch.tensor(False) for _ in range(num_clusters)]
    cluster_area = [torch.tensor(0, dtype = input_cond.x.dtype) for _ in range(num_clusters)]
    cluster_centroid = torch.zeros((B, num_clusters, 2), dtype=placements.dtype, device=placements.device)
    non_clustered_ids = [] # (V_unclustered) list containing IDs in output (clustered) cond
    # so non_clustered_id[instance_idx] = cluster_idx
    for i, (size, part, port, macro, _) in enumerate(components):
        if port or macro:
            non_clustered_ids.append(len(data_macros))
            data_x.append(size)
            data_ports.append(port)
            data_macros.append(macro)
        else:
            non_clustered_ids.append(part)
            obj_area = size[0]*size[1]
            cluster_area[part] += obj_area
            cluster_centroid[:, part, :] += obj_area * placements[:, i, :]
    macro_port_placements = placements[:, torch.logical_or(input_cond.is_macros, input_cond.is_ports), :]
    cluster_placements = cluster_centroid / torch.stack(cluster_area, dim=0).view((1, num_clusters, 1)).to(device=cluster_centroid.device) # 计算质心
    output_placements = torch.cat((cluster_placements, macro_port_placements), dim = 1)

    # ensuring a 1:1 aspect ratio  # 把每个 cluster 当成一个正方形 macro，用 √area 作为它的宽和高
    data_x_macro = [torch.tensor([math.sqrt(cluster_area[part]), math.sqrt(cluster_area[part])], dtype=input_cond.x.dtype) for part in range(num_clusters)]
    data_x = data_x_macro + data_x

    # new edges should specify the index of the cluster
    new_id_raw = lambda u : components[u][1] if not(components[u][2] or components[u][3]) else non_clustered_ids[u]
    new_id = lambda x : new_id_raw(x.item())

    # generate new pin ids
    # if "edge_pin_id" in input_cond:
    #     edge_index_unique = input_cond.edge_index[:, :E//2].T # (E, 2)
    #     edge_pin_id_unique = input_cond.edge_pin_id[:E//2, :] # (E, 2)
    #     sources = torch.cat((
    #         edge_index_unique[:,0:1].double(), 
    #         edge_pin_id_unique[:,0:1].double(),
    #         ), dim=1)
    #     dests = torch.cat((
    #         edge_index_unique[:,1:2].double(),
    #         edge_pin_id_unique[:,1:2].double(),
    #         ), dim=1)
    #     edge_endpoints = torch.cat((sources, dests), dim=0) # (2E, 2)
    #     _, global_pin_ids = torch.unique(edge_endpoints, return_inverse=True, dim=0) # (E_u, 3), (2E)
    #     global_pin_ids = global_pin_ids.view(2, E//2)
    #     reverse_pin_ids = torch.cat((global_pin_ids[1:2, :], global_pin_ids[0:1, :]), dim=0)
    #     global_pin_ids = torch.cat((global_pin_ids, reverse_pin_ids), dim=1) # (2, 2E)
    #     output_edge_pin_id = torch.stack([
    #         torch.tensor([u_id, v_id], dtype=input_cond.edge_pin_id.dtype)
    #         for (u, v), (u_id, v_id) in zip(input_cond.edge_index.T, global_pin_ids.T)
    #         if new_id(u) != new_id(v)])

    # ensuring all ports are at the center of the macro
    # port_macro = [data_x_macro[part][0]/2 for part in range(num_clusters)]
    new_port = lambda u, e: 0 if u < num_clusters else e
    try: # DEBUGGING TODO
        output_edge_index = torch.stack([
            torch.tensor([new_id(u), new_id(v)], dtype=input_cond.edge_index.dtype) 
            for (u, v), e in zip(input_cond.edge_index.T, input_cond.edge_attr)
            if new_id(u) != new_id(v)]).movedim(-1, 0)
        output_edge_attr = torch.stack([
            torch.tensor([new_port(new_id(u), e[0]), new_port(new_id(u), e[1]), new_port(new_id(v), e[2]), new_port(new_id(v), e[3])], dtype=input_cond.edge_attr.dtype)
            for (u, v), e in zip(input_cond.edge_index.T, input_cond.edge_attr)
            if new_id(u) != new_id(v)])
        output_cluster_map = torch.tensor(non_clustered_ids, dtype = torch.int)
    except:
        import ipdb; ipdb.set_trace()

    if (output_placements.isinf().any().item() or output_placements.isnan().any().item()):
        empty_clusters = set(assigned_parts) - set(non_clustered_ids)
        zero_area_clusters = set([i for i, area in enumerate(cluster_area) if area.item() <= 1e-12])
        print("WARNING: Empty cluster(s) detected: ", input_cond, empty_clusters, zero_area_clusters)
        # deal with clusters with no standard cells
        for empty_cluster_idx in sorted(empty_clusters, reverse=True):
            # remove entry for empty cluster from masks and placement
            data_x.pop(empty_cluster_idx)
            data_ports.pop(empty_cluster_idx)
            data_macros.pop(empty_cluster_idx)
            output_placements = torch.cat((output_placements[:, :empty_cluster_idx, :], output_placements[:, (empty_cluster_idx+1):, :]), dim=1)
            # remap edge_index and cluster_map
            output_edge_index = output_edge_index - (output_edge_index > empty_cluster_idx).int()
            output_cluster_map = output_cluster_map - (output_cluster_map > empty_cluster_idx).int()
        assert not (output_placements.isinf().any().item() or output_placements.isnan().any().item()), "nans still present in data"
        assert output_edge_index.max() < len(data_x), "edge indices mapped incorrectly while removing empty clusters"
        assert output_cluster_map.max() < len(data_x), "output cluster maps generated incorrectly while removing empty clusters"

    output_x = torch.stack(data_x, dim=0)
    output_ports = torch.stack(data_ports, dim=0)
    output_macros = torch.stack(data_macros, dim=0)

    output_cond = Data(
        x = output_x,
        is_ports = output_ports,
        is_macros = output_macros,
        edge_index = output_edge_index,
        edge_attr = output_edge_attr,
        cluster_map = output_cluster_map,
    )
    # if "edge_pin_id" in input_cond:
    #     output_cond.edge_pin_id = output_edge_pin_id
    # (shallow) copy over other attributes in cond
    for k in input_cond.keys():
        if not k in output_cond:
            output_cond[k] = input_cond[k]

    # return to original device
    placements.to(device = placements_device)
    input_cond.to(device = cond_device)

    return  output_placements.to(device = placements_device), output_cond.to(device = cond_device),

def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def process_one(args):
    j, x, cond = args

    # 修复 mask bug
    N = cond.x.shape[0]
    cond.is_macros = cond.is_macros[:N]
    cond.is_ports  = cond.is_ports[:N]
    num_clusterable = (~(cond.is_macros | cond.is_ports)).sum()
    print(f"[INFO] - job: {j}, num_obj:{N}, num_clusterable:{num_clusterable}")
    # 每个进程一个独立 temp_dir，防止 hmetis 冲突
    temp_dir = f"logs/temp_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    # temp_dir = f"/dev/shm/hmetis_temp_worker_{os.getpid()}"   # /dev/shm/ 是内存盘，I/O 速度极快，对 hmetis 提速非常明显
    # temp_dir = "logs/temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    max_allowed = num_clusterable // 10
    num_clusters = 1 << int(math.log2(max_allowed))
    sample = cluster(
        j,
        cond,
        min(512, num_clusters),
        temp_dir=temp_dir,
        placements=x.unsqueeze(0),
        algorithm="hmetis"
    )

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

    return sample


def main():
    pyGDir = f"data-gen/outputs/v2.61/cluster/"
    os.makedirs(pyGDir, exist_ok=True)

    for i in range(2600, 7501, 100): # range(100, 2501, 100):
        pyGFile = f"data-gen/outputs/v2.61/{i:08d}.pickle"
        Logger.printPlace(f"deal with PyG file {pyGFile} ...")

        data = load_pickle(pyGFile)

        # tasks = [(j, data[j][0], data[j][1]) for j in range(len(data))]

        sample_batch = []

        # 根据你机器CPU核数调整 max_workers
        with mp.Pool(10) as pool:
            # sample_batch = pool.map(    # 
            #     process_one,
            #     [(j, data[j][0], data[j][1]) for j in range(len(data))]
            # )
            sample_batch = list(pool.imap_unordered(
                process_one,
                [(j, data[j][0], data[j][1]) for j in range(len(data))]
            ))
        # with ProcessPoolExecutor(max_workers=8) as executor:
        #     for sample in tqdm(executor.map(process_one, tasks), total=len(tasks)):
        #         sample_batch.append(sample)

        save_pickle(sample_batch, os.path.join(pyGDir, f"{i:08d}.pickle"))


if __name__ == "__main__":
    main()
