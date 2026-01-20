
import pickle

with open("outputs/v1.61/00000100.pickle", "rb") as f:
    sample_batch = pickle.load(f)

print("样本数:", len(sample_batch))
for i, (positions, data) in enumerate(sample_batch):
    print(f"样本 {i}:")
    print("  positions:", type(positions), getattr(positions, "shape", None))
    print("  Data object:", type(data))
    # 查看 Data 的字段和形状
    print("   x:", data.x.shape if hasattr(data, "x") else None)
    print("   edge_index:", data.edge_index.shape if hasattr(data, "edge_index") else None)
    print("   edge_attr:", data.edge_attr.shape if hasattr(data, "edge_attr") else None)
    print("   is_ports:", data.is_ports.shape if hasattr(data, "is_ports") else None)
    # 比如还可以打印前几个值
    print("   positions[:5]:", positions[:5])
    print("   data.x[:5]:", data.x[:5])
    print("   data.edge_attr[:5]:", data.edge_attr[:5])
    break  # 如果样本太多，这里可以只看第一个
