import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import dataBase.DataBase as DataBase
import dataBase.Logger as Logger
import time


def main():
    startTime = time.time()

    base_dir = "benchmarks/ibmBookshelfModifyFixedMacro"
    output_file = "analysis/netlist_stats.txt"

    # 写入表头
    with open(output_file, "w") as f:
        f.write(
            "case,numNodes,numMacros,numPins,numNets,"
            "total_area,macro_area,macro_ratio,"
            "avg_degree,max_degree,top10_ratio,macro_net_ratio\n"
        )

    # 循环 ibm01 -> ibm18
    for i in range(1, 19):
        case = f"ibm{i:02d}"
        auxInputFile = os.path.join(base_dir, case, f"{case}.aux")

        if not os.path.exists(auxInputFile):
            print(f"[Skip] {auxInputFile} not found")
            continue

        print(f"\nProcessing {case} ...")

        dataBase = DataBase.DataBase()
        dataBase.auxFileName = case
        dataBase.readBookshelf(auxInputFile)

        # ========== 统计 ==========
        numMacros = len(dataBase.macros)
        numNodes  = len(dataBase.nodes)
        numPins   = len(dataBase.pins)
        numNets   = len(dataBase.nets)

        macro_area = sum(m.width * m.height for _, m in dataBase.macros.items())
        total_area = sum(n.width * n.height for _, n in dataBase.nodes.items())
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
                f"{avg_degree:.4f},{max_degree},{top10_ratio:.6f},{macro_net_ratio:.6f}\n"
            )

    print(f"\nAll done. Results saved to {output_file}")
    print(f"Total time: {time.time() - startTime:.2f}s")



if __name__ == "__main__":
    main()
