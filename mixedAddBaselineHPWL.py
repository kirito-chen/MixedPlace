
import pickle
from torch_geometric.data import Data
import os
import torch

def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def main():

    # 1. 读入 dreamplace运行的初始线长
    baseHpwlMap = {}
    # baseHpwlPath = "/home/pc/data/cjq/work/chipdiffusion-mixedPlace/data-gen/outputs/v2.61/cluster/0-2500hpwl.csv"
    baseHpwlPath = "/home/pc/data/cjq/work/chipdiffusion-mixedPlace/data-gen/outputs/v2.61/cluster/2500-7500hpwl.csv"
    with open(baseHpwlPath, "r") as f:
        for line in f:
            k, v = line.strip().split(",")
            baseHpwlMap[int(k)] = float(v)

    fileId = 0
    pyGDir = "data-gen/outputs/v2.61/clusterWithBase"
    for i in range(100, 7501, 100): # range(100, 2501, 100):  # 7501
        pyGFile = f"data-gen/outputs/v2.61/cluster/{i:08d}.pickle"
        # pyGFile = "/home/pc/data/cjq/work/chipdiffusion-main/data-gen/outputs/v2.61/00000100.pickle"
        print(f"deal with PyG file {pyGFile} ...")
        data = load_pickle(pyGFile)
        sample_batch = []
        for j in range(len(data)):
            baselineHPWL = baseHpwlMap[i - 100 + j]
            x = data[j][0][0]  # 去掉一个维度， [1,x,2] -> [x,2]
            cond = data[j][1]
            new_cond = Data(
                x = cond.x,
                is_ports = cond.is_ports,
                is_macros = cond.is_macros,
                edge_index = cond.edge_index,
                edge_attr = cond.edge_attr,
                cluster_map = cond.cluster_map,
                # baselineHPWL = baselineHPWL
                baselineHPWL = torch.tensor([baselineHPWL], dtype=torch.float)
            )
            sample = (x, new_cond)
            sample_batch.append(sample)
            fileId += 1
        save_pickle(sample_batch, os.path.join(pyGDir, f"{i:08d}.pickle"))


if __name__ == "__main__":
    main()