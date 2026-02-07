import torch
import re

class Tester:
    def readAndWritePl(self, pl_path, x, cluster_map):
        pattern = re.compile(r"^\s*(\S+)\s+([\d\.]+)\s+([\d\.]+)\s+:\s+(\S+)(.*)$")

        new_lines = []

        with open(pl_path, "r") as f:
            for line in f:
                line_strip = line.strip()

                if not line_strip or line_strip.startswith("#") or line_strip.startswith("UCLA"):
                    new_lines.append(line)
                    continue

                m = pattern.match(line)
                if not m:
                    new_lines.append(line)
                    continue

                name, old_x, old_y, orient, suffix = m.groups()

                if name.startswith("a") and name[1:].isdigit():
                    originid = int(name[1:])

                    if originid in cluster_map:
                        clusterid = cluster_map[originid].item()
                        new_x, new_y = x[clusterid].tolist()

                        new_line = f"{name}  {new_x:.4f}  {new_y:.4f} : {orient}{suffix}\n"
                        new_lines.append(new_line)
                        continue

                new_lines.append(line)

        with open(pl_path, "w") as f:
            f.writelines(new_lines)


# -------------------------
# 构造测试用例
# -------------------------

# 1. 写一个测试 pl 文件
pl_content = """UCLA pl 1.0
# Created  : test

a0  100.0  200.0 : N
a1  300.0  400.0 : N /FIXED
a2  500.0  600.0 : N
"""

pl_path = "test.pl"
with open(pl_path, "w") as f:
    f.write(pl_content)

print("===== 原始文件 =====")
with open(pl_path) as f:
    print(f.read())


# 2. 构造 cluster_map (originid -> clusterid)
cluster_map = {
    0: torch.tensor(1),
    1: torch.tensor(0),
    2: torch.tensor(2),
}

# 3. 构造 x tensor (clusterid -> 新坐标)
x = torch.tensor([
    [1111.0, 2222.0],  # cluster 0
    [3333.0, 4444.0],  # cluster 1
    [5555.0, 6666.0],  # cluster 2
])

# 4. 调用你的函数
tester = Tester()
tester.readAndWritePl(pl_path, x, cluster_map)

# 5. 查看修改后的文件
print("===== 修改后文件 =====")
with open(pl_path) as f:
    print(f.read())
