import pickle
import torch
from parse_utils import DefFile
import os

import shutil
import re

def update_def_component_xy(line, new_x, new_y):
    """
    Replace the ( x y ) part in a DEF COMPONENT line.
    """
    return re.sub(
        r"\(\s*-?\d+\s+-?\d+\s*\)",
        f"( {new_x:.4f} {new_y:.4f} )",
        line
    )

def process_one_sample(sample_path, output_dir, sample_idx, condPath, oralDEFPath, 
                       cluster_scale_factor, fileName, oralLEFPath):
    with open(sample_path, "rb") as f:
        clustered_placement = pickle.load(f)

    with open(condPath, "rb") as f:
        clustered_graph = pickle.load(f)
    cluster_map = clustered_graph.cluster_map

    base_def = DefFile(oralDEFPath)

    assert base_def.components[0].startswith("COMPONENTS")
    assert base_def.components[-1].startswith("END COMPONENTS")
    assert len(base_def.components) == len(cluster_map) + 2

    ### 新加的
    os.makedirs(output_dir, exist_ok=True)
    out_dir = os.path.join(output_dir, fileName)
    os.makedirs(out_dir, exist_ok=True)
    out_map_path = os.path.join(out_dir, f"{fileName}.txt")
    with open(out_map_path, "w") as outf:
        for orig_id in range(len(cluster_map)):
            comp_idx = orig_id + 1  # 跳过 "COMPONENTS"
            cluster_id = cluster_map[orig_id].item()
            line = base_def.components[comp_idx]
            parts = line.split()
            if clustered_graph.is_macros[cluster_id]: # clustered_graph.is_ports[cluster_id] or 
                outf.write(f"{orig_id}  {parts[1]}  {cluster_id}  {parts[1]}\n")
            else:
                outf.write(f"{orig_id}  {parts[1]}  {cluster_id}  cluster{cluster_id}\n")
    return 

    for orig_id in range(len(cluster_map)):
        comp_idx = orig_id + 1  # 跳过 "COMPONENTS"

        cluster_id = cluster_map[orig_id].item()
        new_xy = clustered_placement[cluster_id]

        x = new_xy[0].item() * cluster_scale_factor
        y = new_xy[1].item() * cluster_scale_factor

        old_line = base_def.components[comp_idx]
        new_line = update_def_component_xy(old_line, x, y)
        base_def.components[comp_idx] = new_line

    os.makedirs(output_dir, exist_ok=True)
    out_dir = os.path.join(output_dir, fileName)
    os.makedirs(out_dir, exist_ok=True)
    out_def_path = os.path.join(out_dir, f"{fileName}.def")
    out_lef_path = os.path.join(out_dir, f"{fileName}.lef")
    shutil.copy(oralLEFPath, out_lef_path)
    print(f"[INFO] copied LEF {oralLEFPath} → {out_lef_path}")
    base_def.write_output(out_def_path)

    print(f"[OK] save file: {out_def_path}")



def main():
    condPathTemplate = "datasets/graph/ibm.cluster512.v1/graph{}.pickle"
    oralDEFPathTemplate = "benchmarks/ibm/ibm{idx:02d}/ibm{idx:02d}.def"
    oralLEFPathTemplate = "benchmarks/ibm/ibm{idx:02d}/ibm{idx:02d}.lef"
    fileNameTemplate = "ibm{idx:02d}"
    cluster_scale_factor = 100.0  # 固定值

    # 两套实验配置
    exp_configs = [
        # {
        #     "name": "proposed",
        #     "sample_dir": "logs/diffusion_debug/ibm.cluster512.v1.eval_guided.300/again_paper_model/samples",
        #     "output_dir": "benchmarks/ibm/2512201545-best-time1000",
        # },
        # {
        #     "name": "baseline",
        #     "sample_dir": "logs/diffusion_debug/ibm.cluster512.v1.eval_guided.300/again_baseline/samples",
        #     "output_dir": "benchmarks/ibm/baseline-time1000",
        # },
        # 输出映射关系
        {
            "name": "cluster_map",
            "sample_dir": "logs/diffusion_debug/ibm.cluster512.v1.eval_guided.300/again_baseline/samples", 
            "output_dir": "benchmarks/ibm/cluster_map"
        },
    ]

    for cfg in exp_configs:
        print(f"\n=== Processing {cfg['name']} ===")

        for i in range(18):  # 0 ~ 17
            print(i)
            sample_path = os.path.join(cfg["sample_dir"], f"sample{i}.pkl")

            if not os.path.exists(sample_path):
                print(f"[WARN] missing {sample_path}, skip")
                continue

            condPath = condPathTemplate.format(i)
            oralDEFPath = oralDEFPathTemplate.format(idx=i+1)   # i=0 -> ibm01
            oralLEFPath = oralLEFPathTemplate.format(idx=i+1)
            fileName = fileNameTemplate.format(idx=i+1)
            print(f"oralDEFPath:{oralDEFPath}")
            print(f"oralLEFPath:{oralLEFPath}")
            print(f"fileName:{fileName}")


            process_one_sample(
                sample_path=sample_path,
                output_dir=cfg["output_dir"],
                sample_idx=i,
                condPath=condPath,
                oralDEFPath=oralDEFPath,
                cluster_scale_factor=cluster_scale_factor,
                fileName=fileName,
                oralLEFPath=oralLEFPath
            )

if __name__=="__main__":
    main()