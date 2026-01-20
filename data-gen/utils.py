from omegaconf import OmegaConf
from PIL import Image, ImageDraw
import numpy as np
import torch
import pickle

import subprocess
import os

def save_cfg(cfg, path):
    with open(path, "w") as f:
        OmegaConf.save(config=cfg, f=f)

def load_cfg(path):
    with open(path, "r") as f:
        return OmegaConf.load(f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def visualize(x, attr, mask = None):
    """ 
    Visualizes the X with node attributes, returning an numpy image
    x,y are floats normalized to canvas size (from -1 to 1)
    attr are also normalized to canvas size
    x: (V, 2) x,y position
    attr: (V, 2) sizes
    """
    width, height = 128, 128
    background_color = "white"
    image = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(image)

    h_step = 1 / len(x)
    for i, (pos, shape) in enumerate(zip(x, attr)):
        left = pos[0]
        top = pos[1] + shape[1]
        right = pos[0] + shape[0]
        bottom = pos[1]
        inbounds = (left>=-1) and (top<=1) and (right<=1) and (bottom>=-1)

        left = (0.5 + left/2) * width
        right = (0.5 + right/2) * width
        top = (0.5 - top/2) * height
        bottom = (0.5 - bottom/2) * height

        color = hsv_to_rgb(i * h_step, 1 if (mask is None or not mask[i]) else 0.2, 0.9 if inbounds else 0.5)
        draw.rectangle([left, top, right, bottom], fill=color)

    return np.array(image)

def visualize_placement(x, cond, plot_edges = True):
    """ 
    Visualizes the X with node attributes, returning an numpy image
    All coordinates are normalized w.r.t canvas size
    x is (V, 2) tensor with 2D coordinates describing placement of center of instances
    cond is pytorch geometric Data object with the following:
    - x is (V, 2) tensor with sizes of instances
    - edge_index (2, E)
    - edge_attr (E, 4) tensor describing pin locations, measured relative to center of instance
    """
    width, height = 2048, 2048
    background_color = "white"
    image = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(image)

    def canvas_to_pixel_coord(x):
        # x is (B, 2) tensor representing normalized 2D coordinates in canvas space
        output = torch.zeros_like(x)
        output[:,0] = (0.5 + x[:,0]/2) * width
        output[:,1] = (0.5 - x[:,1]/2) * height
        return output

    V,_ = x.shape
    mask = cond.is_ports if "is_ports" in cond else None
    h_step = 1 / V

    left_bottom = x - cond.x/2
    right_top = x + cond.x/2
    inbounds = torch.logical_and(left_bottom >= -1, right_top <= 1)
    inbounds = torch.logical_and(inbounds[:,0], inbounds[:,1])

    left_bottom_px = canvas_to_pixel_coord(left_bottom)
    right_top_px = canvas_to_pixel_coord(right_top)
    
    for i in range(V):
        color = hsv_to_rgb(
            i * h_step, 
            1 if (mask is None or not mask[i]) else 0.2, 
            0.9 if inbounds[i] else 0.5,
        )
        draw.rectangle([left_bottom_px[i,0], right_top_px[i,1], right_top_px[i,0], left_bottom_px[i,1]], fill=color, width=0)

    # get pin positions
    unique_edges = cond.edge_attr.shape[0]//2
    u_pos = cond.edge_attr[:unique_edges,:2] + x[cond.edge_index[0,:unique_edges]]
    v_pos = cond.edge_attr[:unique_edges,2:4] + x[cond.edge_index[1,:unique_edges]]
    u_pos = canvas_to_pixel_coord(u_pos)
    v_pos = canvas_to_pixel_coord(v_pos)

    # plot edges
    if plot_edges:
        for i in range(unique_edges):
            draw.line([tuple(u_pos[i].detach().cpu().numpy()), tuple(v_pos[i].detach().cpu().numpy())], fill="gray")
    # plot pin positions
    draw.point([(row[0], row[1]) for row in u_pos.detach().cpu().numpy()], fill="black")
    draw.point([(row[0], row[1]) for row in v_pos.detach().cpu().numpy()], fill="yellow")
    
    return np.array(image)

def hsv_to_rgb(h, s, v):
    """
    Converts HSV (Hue, Saturation, Value) color space to RGB (Red, Green, Blue).
    h: float [0, 1] - Hue
    s: float [0, 1] - Saturation
    v: float [0, 1] - Value
    Returns: tuple (r, g, b) representing RGB values in the range [0, 255]
    """
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q

    return int(r * 255), int(g * 255), int(b * 255)

def debug_plot_img(x, name = "debug_img", rescale = False, autoscale = False):
    # x is (C, H, W) image, this function plots and saves to file
    # assumes images are [-1, 1]
    import matplotlib.pyplot as plt
    # scaling
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    if len(x.shape)==3 and (x.shape[-1] not in [1, 2, 3]):
        x = np.moveaxis(x,0,-1)
    if rescale:
        x = (x + 1)/2 if not autoscale else (x-x.min())/(x.max()-x.min())
    plt.imshow(x)
    plt.savefig(name, dpi=1000)

def run_hmetis(input_cond, num_clusters, algorithm = "hmetis", ubfactor=5, temp_dir = "logs/temp", verbose = False):
    """
    algorithm is one of "hmetis", "shmetis", "khmetis"
    """
    hyp_e = get_hyperedges_2(input_cond)
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

    # call hmetis and parse output
    # shmetis HGraphFile Nparts UBfactor
    if algorithm == "shmetis":
        subprocess.run([
            './shmetis', 
            hmetis_input_file, 
            str(num_clusters), 
            str(int(ubfactor)),
            ], capture_output = not verbose)
    elif algorithm == "hmetis":
        # hmetis HGraphFile Nparts UBfactor Nruns CType Rtype Vcycle Reconst debuglevel
        nruns = 25 # 10 is shmetis default
        ctype = 1 # shmetis default
        rtype = 1 # shemtis default
        vcycle = 3 # 1 is shmetis default; consider using 3 for slower but better
        reconst = 1 # 0 is shmetis default; 1 is reconstruct partial hyperedges
        dbglvl = 0 # 0 is no debugging; 31 is for everything
        subprocess.run([
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
    elif algorithm == "khmetis":
        # khmetis HGraphFile Nparts UBfactor Nruns CType Otype Vcycle debuglevel
        nruns = 10 # 10
        ctype = 1 # shmetis default
        otype = 2 # 2 is minimize sum of ext. degrees
        vcycle = 3 # 1 is shmetis default; consider using 3 for slower but better
        dbglvl = 0 # 0 is no debugging; 31 is for everything
        subprocess.run([
            './khmetis', 
            hmetis_input_file, 
            str(num_clusters), 
            str(int(ubfactor)), 
            str(nruns), 
            str(ctype), 
            str(otype),
            str(vcycle),
            str(dbglvl),
            ], capture_output = not verbose)
    else:
        raise NotImplementedError

    hmetis_output_file = f'{hmetis_input_file}.part.{num_clusters}'  # 'logs/temp/edges-286505397008.txt.part.512'
    with open(hmetis_output_file, 'r') as fp:
        assigned_parts = list(map(int, fp.readlines()))
    assert len(assigned_parts) == input_cond.x.shape[0], f"error parsing shmetis output. expected lines {input_cond.x.shape[0]} but got {len(assigned_parts)}"
    
    # clean up hmetis i/o files
    # os.remove(hmetis_input_file)
    # os.remove(hmetis_output_file)
    
    return assigned_parts

def get_hyperedges_2(cond_val):
    # TODO test for edge cases like hyperedges with >2 vertices
    # exclude macros from hyperedges
    hyp_e = {}
    _, E = cond_val.edge_index.shape
    insts = cond_val.edge_index.T[:E//2, :]
    inst_pins = cond_val.edge_pin_id[:E//2, :] if "edge_pin_id" in cond_val else cond_val.edge_attr[:E//2, :] 

    # We add 1 to all ids, such that ids are 1 -> num(nodes) inclusive
    # exclude macros and ports from hyperedges
    for i, p in zip(insts, inst_pins):
        u, v = i.tolist()
        u_id = tuple(p[:len(p)//2].tolist())
        if (u, u_id) not in hyp_e:
            hyp_e[(u, u_id)] = []
        if not (cond_val.is_macros[u] or cond_val.is_ports[u]):
            hyp_e[(u, u_id)] += [u+1]
        if not (cond_val.is_macros[v] or cond_val.is_ports[v]):
            hyp_e[(u, u_id)] += [v+1]
    
    # remove hyperedges with <= 1 nodes
    output_hyperedges = {}
    for k, v in hyp_e.items():
        if len(v) > 1:
            output_hyperedges[k] = v
    return output_hyperedges