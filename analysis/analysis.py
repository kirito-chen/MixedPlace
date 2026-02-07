import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import dataBase.DataBase as DataBase
import dataBase.Logger as Logger
import time

from collections import defaultdict
import matplotlib.pyplot as plt


def main():
    startTime = time.time()

    # base_dir = "benchmarks/ibmBookshelfModifyFixedMacro"
    # output_file = "analysis/netlist_stats.csv"
    # pic_dir = "analysis/pic"

    base_dir = "data-gen/outputs/v2.61/CircuitGen"
    output_file = "analysis/CircuitGen/netlist_stats.csv"
    pic_dir = "analysis/CircuitGen/pic"

    # 写入表头
    with open(output_file, "w") as f:
        f.write(
            "case,numNodes,numMacros,numPins,numNets,"
            "total_area,macro_area,macro_ratio,"
            "avg_degree,max_degree,top10_ratio,macro_net_ratio,density\n"
        )

    # 循环 ibm01 -> ibm18
    # for i in range(1, 19):
    #     case = f"ibm{i:02d}"
    for i in range(0, 2):
        case= f"CircuitGen{i:04d}"
        auxInputFile = os.path.join(base_dir, case, f"{case}.aux")

        if not os.path.exists(auxInputFile):
            print(f"[Skip] {auxInputFile} not found")
            continue

        print(f"\nProcessing {case} ...")

        dataBase = DataBase.DataBase()
        dataBase.auxFileName = case
        dataBase.readBookshelf(auxInputFile, macroHeight = 17)

        # ========== 统计 ==========
        numMacros = len(dataBase.macros)
        numNodes  = len(dataBase.nodes)
        numPins   = len(dataBase.pins)
        numNets   = len(dataBase.nets)

        macro_area = sum(m.width * m.height for _, m in dataBase.macros.items())
        total_area = sum(n.width * n.height for _, n in dataBase.nodes.items())
        die_area = dataBase.dieWidth * dataBase.dieHeight
        density = total_area / die_area
        macro_ratio = macro_area / total_area if total_area > 0 else 0

        degrees = [len(net.pinIndex) for _, net in dataBase.nets.items()]
        avg_degree = sum(degrees) / len(degrees)
        max_degree = max(degrees)
        degrees_sorted = sorted(degrees, reverse=True)
        top10_ratio = sum(degrees_sorted[:10]) / sum(degrees)

        macro_set = set(dataBase.macros.keys())
        macro_nets = sum(
            any(dataBase.pins[pid].nodeName in macro_set for pid in net.pinIndex)
            for _, net in dataBase.nets.items()
        )
        macro_net_ratio = macro_nets / numNets

        # ========== 打印到终端 ==========
        print(f"numNodes={numNodes}, numMacros={numMacros}, macro_ratio={macro_ratio:.3f}, avg_degree={avg_degree:.2f}")

        # ========== 写入文件 ==========
        with open(output_file, "a") as f:
            f.write(
                f"{case},{numNodes},{numMacros},{numPins},{numNets},"
                f"{total_area:.2f},{macro_area:.2f},{macro_ratio:.6f},"
                f"{avg_degree:.4f},{max_degree},{top10_ratio:.6f},{macro_net_ratio:.6f}, {density:.4f}\n"
            )
        # ========== 统计每个 node 的 degree ==========
        node_degree = defaultdict(int)

        for _, net in dataBase.nets.items():
            for pid in net.pinIndex:
                node = dataBase.pins[pid].nodeName
                node_degree[node] += 1

        # 分开 macro 和 cell
        macro_set = set(dataBase.macros.keys())
        cell_set  = set(dataBase.nodes.keys()) - macro_set

        macro_degrees = [node_degree[n] for n in macro_set if n in node_degree]
        cell_degrees  = [node_degree[n] for n in cell_set if n in node_degree]

        # node 分布
        macro_x, macro_y = build_distribution_from_list(macro_degrees)
        cell_x, cell_y   = build_distribution_from_list(cell_degrees)

        # net 分布
        net_pin_counts = [len(net.pinIndex) for _, net in dataBase.nets.items()]
        net_x, net_y = build_distribution_from_list(net_pin_counts)

        # ========== 保存 CSV ==========
        csv_dir = os.path.join(pic_dir, "csv")
        os.makedirs(csv_dir, exist_ok=True)

        save_distribution_csv(cell_x, cell_y, os.path.join(csv_dir, f"{case}_cell_degree.csv"))
        save_distribution_csv(macro_x, macro_y, os.path.join(csv_dir, f"{case}_macro_degree.csv"))
        save_distribution_csv(net_x, net_y, os.path.join(csv_dir, f"{case}_net_degree.csv"))

        # ========== 画图 ==========
        plot_node(cell_x, cell_y, macro_x, macro_y, case, pic_dir)
        plot_net(net_x, net_y, case, pic_dir)


    print(f"\nAll done. Results saved to {output_file}")
    print(f"Total time: {time.time() - startTime:.2f}s")

def build_distribution(degrees):
    """给定 degree list -> (x,y)"""
    dist = defaultdict(int)
    for d in degrees:
        dist[d] += 1
    xs = sorted(dist.keys())
    ys = [dist[x] for x in xs]
    return xs, ys


def plot_node(cell_x, cell_y, macro_x, macro_y, case, pic_dir):
    plt.figure(figsize=(8, 5))

    plt.plot(cell_x, cell_y, label="Cells", linewidth=2)
    plt.plot(macro_x, macro_y, label="Macros", linewidth=2)

    plt.xlabel("Pin count (degree)")
    plt.ylabel("Number of nodes")
    plt.title(f"{case} node degree distribution")

    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()

    plt.tight_layout()
    os.makedirs(pic_dir, exist_ok=True)
    plt.savefig(os.path.join(pic_dir, f"{case}_node_degree.png"), dpi=300)
    plt.close()

def plot_net(net_x, net_y, case, pic_dir):
    plt.figure(figsize=(8, 5))

    plt.plot(net_x, net_y, linewidth=2)

    plt.xlabel("Pin count per net")
    plt.ylabel("Number of nets")
    plt.title(f"{case} net degree distribution")

    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.tight_layout()
    os.makedirs(pic_dir, exist_ok=True)
    plt.savefig(os.path.join(pic_dir, f"{case}_net_degree.png"), dpi=300)
    plt.close()

def save_distribution_csv(xs, ys, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("x,y\n")
        for x, y in zip(xs, ys):
            f.write(f"{x},{y}\n")

def build_distribution_from_list(values):
    dist = defaultdict(int)
    for v in values:
        dist[v] += 1
    xs = sorted(dist.keys())
    ys = [dist[x] for x in xs]
    return xs, ys



if __name__ == "__main__":
    main()
