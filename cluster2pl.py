import pickle

def read_mapping(file_path):
    mapping = {}

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行

            parts = line.split()
            if len(parts) < 4:
                continue  # 防御式编程，防止格式异常
            
            # 0  p1  51  cluster51
            key = parts[1]      # p1, p2, a22 ...
            value = int(parts[2])    # 51 

            mapping[key] = value

    return mapping

def read_nodes(file_path):
    header_lines = []
    nodes = []

    num_nodes = None
    num_terminals = None

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("NumNodes"):
                num_nodes = int(line.split(":")[1])
            elif line.startswith("NumTerminals"):
                num_terminals = int(line.split(":")[1])
            elif num_nodes is None:
                # NumNodes 之前的行，原样保存
                header_lines.append(line.rstrip("\n"))
            else:
                parts = line.split()
                if len(parts) >= 3:
                    node_id = parts[0]
                    sizex = int(parts[1])
                    sizey = int(parts[2])
                    is_terminal = ("terminal" in parts)

                    nodes.append((node_id, sizex, sizey, is_terminal))

    return header_lines, num_nodes, num_terminals, nodes


def read_result(result_path, cluster_scale_factor):
    
    with open(result_path, "rb") as f:
        clustered_placement = pickle.load(f)
    result_xy_map = {
        idx: (
            max(0, int(row[0] * cluster_scale_factor)),
            max(0, int(row[1] * cluster_scale_factor))
        )
        for idx, row in enumerate(clustered_placement)
    }
    return result_xy_map

def update_pl_with_result(
    input_pl_path,
    output_pl_path,
    mapping,
    result_xy,
):
    data = ""
    with open(input_pl_path, "r") as fin:
        for i, line in enumerate(fin):
            # 前 5 行原样输出
            if i < 5 or line.strip() == "":
                data += line
                continue

            parts = line.split()
            if len(parts) < 5:
                data += line
                continue

            old_id = parts[0]

            # 没有映射的，原样输出
            if old_id not in mapping:
                data += line
                continue

            new_id = mapping[old_id]

            # 映射后的 id 没有结果坐标 → 原样输出
            if new_id not in result_xy:
                data += line
                continue

            # 用 result 中的坐标
            newxy = result_xy[new_id]
            x = newxy[0]
            y = newxy[1]

            orient = parts[4]
            
            fixed = ""
            if len(parts) == 6:
                fixed = parts[5]
            temp = f"{old_id} {x:>10} {y:>10} : {orient:>4} {fixed}\n"
            data += temp
    # print(data)
    # exit()
    with open(output_pl_path, "w") as fout:
        fout.write(data)



def process_one_benchmark(name, time_stamp, sampleName):
    mapping_path = f"/home/pc/data/cjq/work/chipdiffusion-main/benchmarks/ibm/cluster_map/{name}/{name}.txt"
    input_pl_path   = f"/home/pc/data/cjq/work/chipdiffusion-main/benchmarks/ibmBookshelfModifyFixedMacro/{name}/{name}.pl"
    result_path = f"logs/diffusion_debug/ibm.cluster512.v1.eval_guided.300/{time_stamp}/samples/{sampleName}.pkl"
    cluster_scale_factor = 100 
    output_pl_path   = f"benchmarks/ibmBookshelfModifyFixedMacro/{name}/{name}.pl"

    mapping = read_mapping(mapping_path)

    result_xy = read_result(result_path, cluster_scale_factor)

    update_pl_with_result(
        input_pl_path=input_pl_path,
        output_pl_path=output_pl_path,
        mapping=mapping,
        result_xy=result_xy,
    )
    print(f"[{name}] write {output_pl_path}")

def main():
    for i in range(1, 19):
        # i = 12
        name = f"ibm{i:02d}"   # ibm01, ibm02, ..., ibm18
        sampleName = f"sample{i-1}" # sample0, sample1, ..., sample17
        time_stamp = "2603031112" # "2603021129_train_best" # "2601271702"
        print(f"\n=== Processing {name} ===")
        process_one_benchmark(name, time_stamp, sampleName)

        # break

if __name__=="__main__":
    main()