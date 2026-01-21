from distributions import get_distribution
import utils
import torch
import shapely
import numpy as np
from torch_geometric.data import Data
import torch_geometric.utils as tgu

from utils import run_hmetis

import random

class V1:
    def __init__(
            self, 
            max_instance, 
            stop_density,
            max_attempts_per_instance,
            aspect_ratio_dist, 
            instance_size_dist, 
            num_terminals_dist,
            edge_dist,
            source_terminal_dist,
        ):
        self.max_instance = max_instance
        self.stop_density = stop_density
        self.aspect_ratio_dist = aspect_ratio_dist
        self.instance_size_dist = instance_size_dist
        self.num_terminals_dist = num_terminals_dist
        self.edge_dist = edge_dist
        self.max_attempts_per_instance = max_attempts_per_instance
        self.source_terminal_dist = source_terminal_dist

    def sample_old(self):
        # Generate instance sizes
        aspect_ratio = get_distribution(**self.aspect_ratio_dist).sample((self.max_instance,))
        long_size = get_distribution(**self.instance_size_dist).sample((self.max_instance,))
        short_size = aspect_ratio * long_size
        long_x = get_distribution("bernoulli", {"probs": 0.5}).sample((self.max_instance,))

        x_sizes = long_x * long_size + (1-long_x) * (short_size)
        y_sizes = (1-long_x) * long_size + (long_x) * (short_size)

        # sort by area, descending order
        areas = x_sizes * y_sizes
        _, indices = torch.sort(areas, descending=True)
        x_sizes = x_sizes[indices]
        y_sizes = y_sizes[indices]

        # place samples individually
        placement = Placement()
        density = 0
        for (x_size, y_size) in zip(x_sizes, y_sizes):
            x_size = float(x_size)
            y_size = float(y_size)
            dist_params = {"low": torch.tensor([-1.0, -1.0]), "high": torch.tensor([1.0-x_size, 1.0-y_size])}
            candidate_dist =  get_distribution("uniform", dist_params)
            
            for attempt_num in range(self.max_attempts_per_instance):
                candidate_pos = candidate_dist.sample()
                # print(instance_idx, attempt_num, candidate_pos)
                if placement.check_legality(candidate_pos[0].item()+x_size/2, candidate_pos[1].item()+y_size/2, x_size, y_size):
                    placement.commit_instance(candidate_pos[0].item()+x_size/2, candidate_pos[1].item()+y_size/2, x_size, y_size)
                    break
            density += (x_size * y_size)/4.0
            if density >= self.stop_density:
                break
        
        positions = placement.get_positions()
        sizes = placement.get_sizes()
        positions = positions - sizes/2 # use bottom left as coordinates
        num_instances = positions.shape[0]

        # sample number of terminals
        num_terminals = get_distribution(**self.num_terminals_dist).sample((num_instances,)).int() # TODO condition dist on area of instance per Rent's
        max_num_terminals = torch.max(num_terminals)
        terminal_offsets = self.get_terminal_offsets(sizes[:,0], sizes[:,1], max_num_terminals, reference="bottom_left")

        # generate edges
        terminal_positions = positions.unsqueeze(dim=1) + terminal_offsets # (V, T, 2)
        terminal_distances = self.get_terminal_distances(terminal_positions)
        edge_exists = get_distribution(**self.edge_dist).sample(terminal_distances) # (V, T, V, T)
        is_source = get_distribution(**self.source_terminal_dist).sample((num_instances, max_num_terminals))

        # delete edges between same instance, among other things
        edge_exists = self.process_edge_matrix(edge_exists, is_source, num_terminals)

        # convert to edge list and generate attributes
        edge_index, edge_attr = self.generate_edge_list(edge_exists, terminal_offsets)
        mask = placement.get_mask()

        data = Data(x=sizes, edge_index=edge_index, edge_attr=edge_attr, is_ports=mask)
        return positions, data

    def sample(self):
        device = "cpu"

        # ==============================
        # 1. configuration
        # ==============================
        num_macros = 100
        num_cells = 100_000
        N = num_macros + num_cells

        avg_fanout = 3
        max_fanout = 20

        # ==============================
        # 2. generate sizes
        # ==============================
        # macros: larger
        macro_area = torch.exp(torch.randn(num_macros) * 0.6 + 0.0)
        macro_aspect = torch.rand(num_macros) * 1.5 + 0.5
        macro_x = torch.sqrt(macro_area * macro_aspect)
        macro_y = macro_area / macro_x

        # cells: smaller
        cell_area = torch.exp(torch.randn(num_cells) * 0.4 - 3.5)
        cell_aspect = torch.rand(num_cells) * 0.5 + 0.75
        cell_x = torch.sqrt(cell_area * cell_aspect)
        cell_y = cell_area / cell_x

        x_sizes = torch.cat([macro_x, cell_x], dim=0)
        y_sizes = torch.cat([macro_y, cell_y], dim=0)

        sizes = torch.stack([x_sizes, y_sizes], dim=1)  # (N,2)

        # ==============================
        # 3. generate positions (no legality)
        # ==============================
        # all instances randomly offset around chip center
        positions = torch.randn(N, 2) * 0.5
        positions = torch.clamp(positions, -1.0, 1.0)

        # ==============================
        # 4. generate fanout
        # ==============================
        fanout = torch.poisson(torch.full((N,), avg_fanout))
        fanout = torch.clamp(fanout, 0, max_fanout).int()

        # ==============================
        # 5. generate edges
        # ==============================
        src_list = []
        dst_list = []

        for i in range(N):
            k = int(fanout[i])
            if k == 0:
                continue
            dst = torch.randint(0, N, (k,))
            src = torch.full((k,), i)
            src_list.append(src)
            dst_list.append(dst)

        src = torch.cat(src_list)
        dst = torch.cat(dst_list)

        edge_index = torch.stack([src, dst], dim=0)  # (2, E)

        # ==============================
        # 6. edge_attr: use dx, dy
        # ==============================
        edge_attr = positions[dst] - positions[src]  # (E, 2)

        # ==============================
        # 7. mask: mark macros as ports (optional)
        # ==============================
        is_ports = torch.zeros(N, dtype=torch.bool)
        is_ports[:num_macros] = True   # treat macros as ports if you want

        # ==============================
        # 8. build PyG data
        # ==============================
        data = Data(
            x=sizes,
            edge_index=edge_index,
            edge_attr=edge_attr,
            is_ports=is_ports
        )

        return positions, data

    def get_terminal_offsets(self, x_sizes, y_sizes, max_num_terminals, reference="center"):
        # TODO check reference points for these
        # NOTE here we assume reference point (for computing offset) is center of instance
        # NOTE outputs use whatever units are being used for sizes
        # x_sizes: (num_instances)
        # y_sizes: (num_instances)
        # max_num_terminals: int
        half_perim = (x_sizes + y_sizes)
        
        terminal_locations = get_distribution("uniform", {"low": 0, "high": half_perim}).sample((max_num_terminals,)) # (max_term, num_instances)
        terminal_flip = get_distribution("bernoulli", {"probs": 0.5}).sample((max_num_terminals, x_sizes.shape[0])) # (max_term, num_instances)
        terminal_flip = (2 * terminal_flip) - 1

        x_sizes = x_sizes.unsqueeze(dim=0)
        y_sizes = y_sizes.unsqueeze(dim=0)
        terminal_offset_x = torch.clamp(terminal_locations, torch.zeros_like(x_sizes), x_sizes) - (x_sizes/2) 
        terminal_offset_y = torch.clamp(terminal_locations-x_sizes, torch.zeros_like(y_sizes), y_sizes) - (y_sizes/2)

        terminal_offset_x = terminal_flip * terminal_offset_x
        terminal_offset_y = terminal_flip * terminal_offset_y

        if reference == "bottom_left":
            terminal_offset_x += x_sizes/2
            terminal_offset_y += y_sizes/2

        terminal_offset = torch.stack((terminal_offset_x, terminal_offset_y), dim=-1).movedim(1, 0) # (num_instances, max_term, 2)
        return terminal_offset

    def get_terminal_distances(self, terminal_positions, norm_order=1):
        # given global terminal positions (V, T, 2)
        # compute and return pairwise L1 distances between terminals
        # optionally, can specify x for Lx distance using norm_order
        V, T, _ = terminal_positions.shape
        t_pos_1 = terminal_positions.view(V, T, 1, 1, 2)
        t_pos_2 = terminal_positions.view(1, 1, V, T, 2)
        delta_pos = t_pos_1 - t_pos_2 # (V, T, V, T, 2)
        distance = torch.norm(delta_pos, p=norm_order, dim=-1) # (V, T, V, T)
        return distance

    def process_edge_matrix(self, edge_exists, is_source, num_terminals):
        # edge_existence tensor (V, T, V, T)
        # is_source int tensor (V, T) (0=sink, 1=source)
        # num terminals int tensor (V)
        # process  as follows:
        # remove all edges ending in a source
        # remove all edges originating from a sink
        # remove all edges between same instance
        # remove all edges starting or ending in nonexistent terminal
        V, T, _, _ = edge_exists.shape
        assert is_source.shape == edge_exists.shape[:2]
        assert num_terminals.shape == (V,)

        # generate terminal filter
        terminal_filter = torch.zeros((V, T))
        for i, num_terminal in enumerate(num_terminals):
            terminal_filter[i, :num_terminal] = 1

        source_filter = (terminal_filter * is_source).view(V, T, 1, 1)
        sink_filter = (terminal_filter * (1-is_source)).view(1, 1, V, T)
        self_edge_filter = (1-torch.eye(V)).view(V, 1, V, 1)
        
        edges = edge_exists * source_filter
        edges = edges * sink_filter
        edges = edges * self_edge_filter
        return edges

    def generate_edge_list(self, edge_exists, terminal_offsets):
        V, T, _, _ = edge_exists.shape
        edges = torch.nonzero(edge_exists) # (E, 4:v,t,v,t) int64
        edge_index_forward = edges[:,(0,2)]
        edge_index_reverse = edges[:,(2,0)]

        edge_attr_source = terminal_offsets[edges[:,0], edges[:,1], :] # (E, 2)
        edge_attr_sink = terminal_offsets[edges[:,2], edges[:,3], :] # (E, 2)
        edge_attr_forward = torch.concat((edge_attr_source, edge_attr_sink), dim=-1)
        edge_attr_reverse = torch.concat((edge_attr_sink, edge_attr_source), dim=-1)
        
        # create undirected edge index and attr
        edge_index = torch.concat((edge_index_forward, edge_index_reverse), dim=0).T # (2, E)
        edge_attr = torch.concat((edge_attr_forward, edge_attr_reverse), dim=0) # (E, 4)
        
        # return copy
        return edge_index.clone(), edge_attr.clone()

class V2:
    def __init__(
            self, 
            max_instance, 
            stop_density_dist,
            max_hard_macros,  # new
            cell_aspect_ratio_dist,
            random_cell,
            max_cells,
            num_clusters,
            max_attempts_per_instance,
            aspect_ratio_dist, 
            instance_size_dist, 
            num_terminals_dist,
            edge_dist,
            source_terminal_dist,
            interior_terminals_dist = None,
            interior_terminals_loc = "uniform",
            zero_edge_attr = False,
            distance_norm_order = 1, # order of norm for measuring distance
        ):
        self.max_instance = max_instance
        self.stop_density_dist = stop_density_dist
        self.aspect_ratio_dist = aspect_ratio_dist
        self.instance_size_dist = instance_size_dist
        self.num_terminals_dist = num_terminals_dist
        self.interior_terminals_dist = interior_terminals_dist
        self.interior_terminals_loc = interior_terminals_loc
        self.edge_dist = edge_dist
        self.max_attempts_per_instance = max_attempts_per_instance
        self.source_terminal_dist = source_terminal_dist
        self.zero_edge_attr = zero_edge_attr
        self.distance_norm_order = distance_norm_order

        # new
        self.max_hard_macros = max_hard_macros
        self.max_cells = max_cells
        self.num_clusters = num_clusters
        self.cell_aspect_ratio_dist = cell_aspect_ratio_dist
        self.random_cell = random_cell

    def sample_old(
            self, 
            size_dist_timer=None, # optional timers
            place_timer=None,
            terminal_timer=None,
            edge_timer=None,
        ):

        size_dist_timer.start() if size_dist_timer else None

        # Generate stop density
        stop_density = get_distribution(**self.stop_density_dist).sample()  # 均匀分布 [0.75, 0.9]

        # Generate instance sizes
        aspect_ratio = get_distribution(**self.aspect_ratio_dist).sample((self.max_instance,)) # 均匀分布 [0.25, 1.0] 采样400个
        long_size = get_distribution(**self.instance_size_dist).sample((self.max_instance,)) # clipped_exp 截断的指数分布  [0.02, 1.0]
        short_size = aspect_ratio * long_size
        long_x = get_distribution("bernoulli", {"probs": 0.5}).sample((self.max_instance,)) # 伯努利分布（二项分布）  Bernoulli(p=0.5)  每次的结果是0或者1

        x_sizes = long_x * long_size + (1-long_x) * (short_size)
        y_sizes = (1-long_x) * long_size + (long_x) * (short_size)

        # sort by area, descending order
        areas = x_sizes * y_sizes
        _, indices = torch.sort(areas, descending=True)   # 降序排列
        x_sizes = x_sizes[indices]
        y_sizes = y_sizes[indices]

        size_dist_timer.stop() if size_dist_timer else None
        place_timer.start() if place_timer else None

        # place samples individually
        placement = Placement()   # 负责检查合法性（不重叠、在芯片范围内等）并记录放置好的模块
        density = 0
        for (x_size, y_size) in zip(x_sizes, y_sizes):
            x_size = float(x_size)
            y_size = float(y_size)
            dist_params = {"low": torch.tensor([(x_size/2)-1.0, (y_size/2)-1.0]), "high": torch.tensor([1.0-(x_size/2), 1.0-(y_size/2)])}
            candidate_dist =  get_distribution("uniform", dist_params)
            
            for attempt_num in range(self.max_attempts_per_instance):
                candidate_pos = candidate_dist.sample()  # 在布局范围内采样， 中心点的位置
                # print(instance_idx, attempt_num, candidate_pos)
                if placement.check_legality(candidate_pos[0].item(), candidate_pos[1].item(), x_size, y_size):
                    placement.commit_instance(candidate_pos[0].item(), candidate_pos[1].item(), x_size, y_size) # 这里都不会是port
                    break

            # ??? 下面这个密度累加如果上面的for循环中找不到合法位置呢，这个密度同样会记录这个模块的面积
            density += (x_size * y_size)/4.0  # 将 [-1,1]*[-1,1] 映射到[0,1]*[0,1]
            if density >= stop_density:
                break
        
        positions = placement.get_positions()
        sizes = placement.get_sizes()
        num_instances = positions.shape[0]

        place_timer.stop() if place_timer else None
        terminal_timer.start() if terminal_timer else None

        # sample number of terminals
        instance_area = sizes[:, 0] * sizes[:, 1]
        num_terminals = get_distribution(**self.num_terminals_dist).sample(instance_area).int() # 面积越大越有概率产生更多的引脚
        # num_terminals = torch.clip(num_terminals, min=1, max=32)
        num_terminals = torch.clip(num_terminals, min=1, max=256) # This is what it should be
        max_num_terminals = torch.max(num_terminals)
        terminal_offsets = self.get_terminal_offsets(sizes[:,0], sizes[:,1], max_num_terminals, reference="center") # 每个模块都产生了 max_num_terminals 的pin

        terminal_timer.stop() if terminal_timer else None
        edge_timer.start() if edge_timer else None

        # generate edges
        terminal_positions = positions.unsqueeze(dim=1) + terminal_offsets # (V, T, 2)  # 转换成pin的实际坐标
        terminal_distances = self.get_terminal_distances(terminal_positions, norm_order=self.distance_norm_order) # (V, T, V, T)
        edge_exists = get_distribution(**self.edge_dist).sample(terminal_distances) # (V, T, V, T)  # 根据距离产生边，约近概率越大
        is_source = get_distribution(**self.source_terminal_dist).sample((num_instances, max_num_terminals))

        # delete edges between same instance, among other things
        edge_exists = self.process_edge_matrix(edge_exists, is_source, num_terminals)  # 删除一些不合理的边（自环，源点结尾，汇点开始）
        self.connect_isolated_instances(edge_exists, terminal_distances) # 现存的边edge_exists  连接孤立点

        # convert to edge list and generate attributes
        edge_index, edge_attr = self.generate_edge_list(edge_exists, terminal_offsets)
        mask = placement.get_mask()
        if self.zero_edge_attr:
            edge_attr = 0 * edge_attr
        
        edge_timer.stop() if edge_timer else None

        data = Data(x=sizes, edge_index=edge_index, edge_attr=edge_attr, is_ports=mask)
        return positions, data
    
    def sample(
        self, 
        size_dist_timer=None, # optional timers
        place_timer=None,
        terminal_timer=None,
        edge_timer=None,
        idx=None
    ):
        # print("==============================")
        # print("DEBUG: Generate a new case ...")
        size_dist_timer.start() if size_dist_timer else None

        # Generate stop density
        stop_density = get_distribution(**self.stop_density_dist).sample()  # 均匀分布 [0.75, 0.9]

        # ==================== 采样行数 ====================
        # 在100-500中采样行数
        num_rows = torch.randint(100, 501, (1,)).item()
        row_height = 2.0 / num_rows  # 行高

        # ==================== 采样macro和cell数量 ====================
        # macro数量在一定范围内随机
        num_macros = torch.randint(150, self.max_hard_macros, (1,)).item()  # 50-150 个macro

        # cell数量在一定范围内随机
        num_cells = torch.randint(80000, self.max_cells, (1,)).item()  # 80000-210000 个cell

        # print(f"DEBUG: Rows: {num_rows}, Row height: {row_height:.4f}")
        # print(f"DEBUG: Macros: {num_macros}, Cells: {num_cells}")

        # ==================== 生成macro模块 ====================
        #################  方式1 可能存在行高小于最小行高row_height的情况
        aspect_ratio = get_distribution(**self.aspect_ratio_dist).sample((num_macros,)) # 均匀分布 [0.25, 1.0] 采样num_macros个
        long_size = get_distribution(**self.instance_size_dist).sample((num_macros,)) # clipped_exp 截断的指数分布  [0.02, 1.0]
        short_size = aspect_ratio * long_size
        long_x = get_distribution("bernoulli", {"probs": 0.5}).sample((num_macros,)) # 伯努利分布（二项分布）

        x_sizes = long_x * long_size + (1-long_x) * (short_size)
        y_sizes = (1-long_x) * long_size + (long_x) * (short_size)

         # 确保y_sizes至少是3倍row_height
        min_y_size = 3 * row_height
        y_sizes_too_small = y_sizes < min_y_size
        
        if y_sizes_too_small.any():
            # 对于y_sizes小于3倍行高的宏块，进行缩放
            # 保持宽高比不变，同时确保y_sizes至少为3倍行高
            scale_factors = min_y_size / y_sizes[y_sizes_too_small]
            y_sizes[y_sizes_too_small] = min_y_size
            x_sizes[y_sizes_too_small] = x_sizes[y_sizes_too_small] * scale_factors

        ################# 方式2 整数倍  用aspect_ratio 范围变成[0.25, 1.75]来代替旋转
        # aspect_ratio = get_distribution(**self.aspect_ratio_dist).sample((num_macros,)) # 均匀分布 [0.25, 1.75]
        # # macro的y_size: row_height的整数倍，范围在3倍到1/2*num_rows倍之间
        # min_multiple = 3
        # max_multiple = num_rows // 2  # 整数除法，确保是整数
        # max_multiple = max(max_multiple, min_multiple)  # 确保max_multiple不小于min_multiple
        
        # # 随机生成倍数
        # multiples = torch.randint(min_multiple, max_multiple + 1, (num_macros,))
        # # 计算y_sizes
        # y_sizes = multiples.float() * row_height
        # # 这意味着x_size会是y_size的0.25到1.75倍
        # x_sizes = y_sizes * aspect_ratio



        # sort by area, descending order
        areas = x_sizes * y_sizes
        _, indices = torch.sort(areas, descending=True)   # 降序排列
        x_sizes = x_sizes[indices]
        y_sizes = y_sizes[indices]

        size_dist_timer.stop() if size_dist_timer else None
        place_timer.start() if place_timer else None

        # 放置macro模块
        placement = Placement()   # 负责检查合法性（不重叠、在芯片范围内等）并记录放置好的模块
        macro_density = 0
        for i, (x_size, y_size) in enumerate(zip(x_sizes, y_sizes)):
            x_size = float(x_size)
            y_size = float(y_size)
            dist_params = {"low": torch.tensor([(x_size/2)-1.0, (y_size/2)-1.0]), 
                        "high": torch.tensor([1.0-(x_size/2), 1.0-(y_size/2)])}
            candidate_dist = get_distribution("uniform", dist_params)
            is_placed = False
            for attempt_num in range(self.max_attempts_per_instance):
                candidate_pos = candidate_dist.sample()  # 在布局范围内采样，中心点的位置
                if placement.check_legality(candidate_pos[0].item(), candidate_pos[1].item(), x_size, y_size):
                    placement.commit_instance(candidate_pos[0].item(), candidate_pos[1].item(), x_size, y_size)
                    is_placed = True
                    break
            if is_placed:
                macro_density += (x_size * y_size)/4.0  # 将 [-1,1]*[-1,1] 映射到[0,1]*[0,1]
                if macro_density >= stop_density - 0.2: # 预留20%放cell
                    break
        
        # 获取已放置的macro数量
        macro_positions = placement.get_positions()
        macro_sizes = placement.get_sizes()
        actual_macros = macro_positions.shape[0]
        # print(f"DEBUG: Placed {actual_macros} macros")
        
        # random_cell = True
        # 生成cell模块 方式1  随机生成
        if self.random_cell:
            # ==================== 生成cell模块 ====================
            # 计算剩余可用的密度
            remaining_density = max(0.0, stop_density - macro_density)
            # print(f"Remaining density for cells: {remaining_density:.4f}")
            if remaining_density <= 0:
                remaining_density = 0.1
            # cell的y_size就是行高
            cell_y_sizes = torch.full((num_cells,), row_height)
            # 使用与macro相同的宽高比分布，但可以限制在较小范围
            cell_aspect_ratio = get_distribution(**self.cell_aspect_ratio_dist).sample((num_cells,))
            cell_x_sizes = cell_y_sizes * cell_aspect_ratio

            # 随机打乱cell顺序，避免总是保留前面的cell
            indices = torch.randperm(num_cells)
            cell_x_sizes = cell_x_sizes[indices]
            cell_y_sizes = cell_y_sizes[indices]

            # 计算每个cell的密度并累加，直到达到剩余密度
            cell_density = 0
            actual_num_cells = 0
            for i in range(num_cells):
                cell_area = cell_x_sizes[i] * cell_y_sizes[i]
                cell_density_increment = cell_area / 4.0
                
                if cell_density + cell_density_increment <= remaining_density:
                    cell_density += cell_density_increment
                    actual_num_cells += 1
                else:
                    # 超过剩余密度，停止添加cell
                    break
            # 只保留前actual_num_cells个cell
            if actual_num_cells > 0:
                cell_x_sizes = cell_x_sizes[:actual_num_cells]
                cell_y_sizes = cell_y_sizes[:actual_num_cells]
                
                # cell位置在芯片中心附近随机偏移
                cell_positions = torch.randn((actual_num_cells, 2)) * 0.3
                cell_positions = torch.clamp(cell_positions, -0.95, 0.95)
                
                cell_sizes = torch.stack([cell_x_sizes, cell_y_sizes], dim=1)
            else:
                cell_positions = torch.zeros((0, 2))
                cell_sizes = torch.zeros((0, 2))

            
            place_timer.stop() if place_timer else None

        else:
            # 生成cell模块 方式2  生成soft macro用于拆分cel
            # ==================== 生成soft macro用于拆分cell ====================
            # 计算剩余可用的密度
            remaining_density = max(0.0, stop_density - macro_density)
            # print(f"Remaining density for cells: {remaining_density:.4f}")
            
            if remaining_density <= 0:
                # print("No remaining density for cells, using random placement")
                # 如果没有剩余密度，使用随机放置
                cell_y_sizes = torch.full((num_cells,), row_height)
                cell_aspect_ratio = get_distribution(**self.cell_aspect_ratio_dist).sample((num_cells,))
                cell_x_sizes = cell_y_sizes * cell_aspect_ratio
                cell_positions = torch.randn((num_cells, 2)) * 0.3
                cell_positions = torch.clamp(cell_positions, -0.95, 0.95)
            else:
                # 生成soft macro，每个soft macro将被拆分为多个cell
                # 估算每个soft macro可以拆分的cell数量
                # soft macro高度：row_height的3-10倍
                # cell高度固定为row_height，宽度为row_height的0.25-1.5倍
                # 平均每个soft macro可拆分出大约 5 * (1.0/0.875) ≈ 5-6个cell
                
                # 需要的soft macro数量估算
                avg_cells_per_soft_macro = 6
                num_soft_macros_needed = max(1, num_cells // avg_cells_per_soft_macro)
                num_soft_macros_needed = min(num_soft_macros_needed, 1000)  # 最多1000个soft macro
                
                # 生成soft macro尺寸
                min_soft_multiple = 3
                max_soft_multiple = 10  # soft macro高度最大为10倍行高
                
                soft_multiples = torch.randint(min_soft_multiple, max_soft_multiple + 1, (num_soft_macros_needed,))
                soft_y_sizes = soft_multiples.float() * row_height
                
                # 使用与macro相同的宽高比分布，但可以限制在较小范围
                soft_aspect_ratio = get_distribution(**self.cell_aspect_ratio_dist).sample((num_soft_macros_needed,))
                soft_x_sizes = soft_y_sizes * soft_aspect_ratio
                
                # 确保soft macro尺寸在合理范围内
                soft_x_sizes = torch.clamp(soft_x_sizes, 0.01, 1.0)
                soft_y_sizes = torch.clamp(soft_y_sizes, 0.01, 1.0)
                
                # 按面积降序排列
                soft_areas = soft_x_sizes * soft_y_sizes
                _, soft_indices = torch.sort(soft_areas, descending=True)
                soft_x_sizes = soft_x_sizes[soft_indices]
                soft_y_sizes = soft_y_sizes[soft_indices]
                
                # 放置soft macro
                soft_macro_density = 0
                soft_macro_positions = []
                soft_macro_sizes = []
                soft_macros_placed = 0
                
                for i, (x_size, y_size) in enumerate(zip(soft_x_sizes, soft_y_sizes)):
                    if soft_macro_density >= remaining_density:
                        break
                        
                    x_size = float(x_size)
                    y_size = float(y_size)
                    dist_params = {"low": torch.tensor([(x_size/2)-1.0, (y_size/2)-1.0]), 
                                "high": torch.tensor([1.0-(x_size/2), 1.0-(y_size/2)])}
                    candidate_dist = get_distribution("uniform", dist_params)
                    is_placed = False
                    
                    for attempt_num in range(self.max_attempts_per_instance):
                        candidate_pos = candidate_dist.sample()
                        if placement.check_legality(candidate_pos[0].item(), candidate_pos[1].item(), x_size, y_size):
                            placement.commit_instance(candidate_pos[0].item(), candidate_pos[1].item(), x_size, y_size)
                            is_placed = True
                            soft_macro_positions.append([candidate_pos[0].item(), candidate_pos[1].item()])
                            soft_macro_sizes.append([x_size, y_size])
                            soft_macros_placed += 1
                            break
                    
                    if is_placed:
                        soft_macro_density += (x_size * y_size)/4.0
                
                # print(f"Placed {soft_macros_placed} soft macros, soft macro density: {soft_macro_density:.4f}")
                
                # 将soft macro拆分为cell
                cell_positions_list = []
                cell_sizes_list = []
                cells_generated = 0
                
                for soft_idx in range(soft_macros_placed):
                    if cells_generated >= num_cells:
                        break
                        
                    # 获取soft macro的位置和尺寸
                    soft_pos = soft_macro_positions[soft_idx]
                    soft_size = soft_macro_sizes[soft_idx]
                    soft_x = soft_size[0]
                    soft_y = soft_size[1]
                    soft_center_x = soft_pos[0]
                    soft_center_y = soft_pos[1]
                    
                    # 计算soft macro的边界
                    left = soft_center_x - soft_x/2
                    right = soft_center_x + soft_x/2
                    bottom = soft_center_y - soft_y/2
                    top = soft_center_y + soft_y/2
                    
                    # 将soft macro按行拆分为cell
                    # 每行高度为row_height
                    num_rows_in_soft = int(soft_y / row_height)
                    if num_rows_in_soft < 1:
                        num_rows_in_soft = 1
                    
                    for row_idx in range(num_rows_in_soft):
                        if cells_generated >= num_cells:
                            break
                            
                        # 计算该行的y坐标范围
                        row_bottom = bottom + row_idx * row_height
                        row_top = row_bottom + row_height
                        
                        # 计算该行的y中心
                        row_center_y = (row_bottom + row_top) / 2
                        
                        # 将该行按cell宽度拆分为多个cell
                        current_x = left
                        while current_x < right and cells_generated < num_cells:
                            # 随机生成cell宽度（row_height的0.25到1.5倍）
                            cell_width = row_height * (torch.rand(1).item() * (1.5 - 0.25) + 0.25)
                            
                            # 如果剩余空间不够放一个cell，调整最后一个cell的宽度
                            if current_x + cell_width > right:
                                cell_width = right - current_x
                            
                            # 确保cell宽度不小于最小宽度
                            min_cell_width = row_height * 0.25
                            if cell_width < min_cell_width and current_x > left:
                                # 如果cell太小，合并到前一个cell（这里简化处理，跳过）
                                break
                            
                            # 计算cell的x中心
                            cell_center_x = current_x + cell_width/2
                            
                            # 记录cell的位置和尺寸
                            cell_positions_list.append([cell_center_x, row_center_y])
                            cell_sizes_list.append([cell_width, row_height])
                            
                            # 更新当前位置和cell计数器
                            current_x += cell_width
                            cells_generated += 1
                
                # 如果生成的cell数量不足，补充一些随机cell
                # if len(cell_positions_list) < num_cells:
                #     additional_cells = num_cells - len(cell_positions_list)
                #     print(f"Adding {additional_cells} additional random cells")
                    
                #     for _ in range(additional_cells):
                #         # 随机生成cell位置（在芯片范围内）
                #         cell_pos = torch.rand(2) * 2 - 1  # [-1, 1]
                #         cell_pos = torch.clamp(cell_pos, -0.95, 0.95)
                        
                #         # 随机生成cell宽度
                #         cell_width = row_height * (torch.rand(1).item() * (1.5 - 0.25) + 0.25)
                        
                #         cell_positions_list.append(cell_pos.tolist())
                #         cell_sizes_list.append([cell_width, row_height])
                
                # 转换为tensor
                cell_positions = torch.tensor(cell_positions_list, dtype=torch.float32)[:num_cells]
                cell_sizes = torch.tensor(cell_sizes_list, dtype=torch.float32)[:num_cells]
    
        
        # ==================== 合并所有模块 ====================
        all_positions = torch.cat([macro_positions, cell_positions], dim=0)
        # all_sizes = torch.cat([macro_sizes, torch.stack([cell_x_sizes, cell_y_sizes], dim=1)], dim=0)
        all_sizes = torch.cat([macro_sizes, cell_sizes], dim=0)
        num_instances = all_positions.shape[0]
        
        
        
        # ==================== 生成terminal（引脚） ====================
        # 简化：所有cell都生成4为标准差的terminal
        terminal_timer.start() if terminal_timer else None

        # 计算所有模块的面积
        instance_area = all_sizes[:, 0] * all_sizes[:, 1]

        # 创建terminal数量数组
        num_terminals_list = []

        # 为macro生成terminal数量（使用原有的面积相关分布）
        if actual_macros > 0:
            # 提取macro的面积
            macro_areas = instance_area[:actual_macros]
            # 使用原有的分布生成macro引脚数
            macro_num_terminals = get_distribution(**self.num_terminals_dist).sample(macro_areas).int()
            # 限制在合理范围内
            macro_num_terminals = torch.clip(macro_num_terminals, min=1, max=100)
            num_terminals_list.extend(macro_num_terminals.tolist())

        # 为cell生成terminal数量（正态分布，标准差为4）
        mean_terminals_cell = 4  # cell平均引脚数
        std_terminals_cell = 2   # cell引脚数标准差

        for i in range(actual_macros, num_instances):
            # cell: 使用正态分布生成引脚数
            # 生成正态分布随机数，然后四舍五入取整
            num_term_float = torch.randn(1) * std_terminals_cell + mean_terminals_cell
            num_term = int(torch.round(num_term_float).item())
            # 确保引脚数在合理范围内（1到20之间）
            num_term = max(1, min(num_term, 20))
            num_terminals_list.append(num_term)
        
        num_terminals = torch.tensor(num_terminals_list, dtype=torch.long)
        max_num_terminals = torch.max(num_terminals)
        
        # print(f"DEBUG: Macro terminals range: using area-based distribution")
        # print(f"DEBUG: Cell terminals: mean={mean_terminals_cell}, std={std_terminals_cell}")
        # print(f"DEBUG: Max terminals: {max_num_terminals.item()}")
        
        # 获取terminal offsets
        terminal_offsets = self.get_terminal_offsets(all_sizes[:,0], all_sizes[:,1], 
                                                    max_num_terminals, reference="center")
        
        terminal_timer.stop() if terminal_timer else None
        edge_timer.start() if edge_timer else None
        
        # ==================== 生成net（超边结构） ====================
        # 按照网表规则生成超边：macro与多个cell相连，形成超边
        
        # 1. 创建有效终端列表
        valid_terminals = []
        for v in range(num_instances):
            for t in range(num_terminals[v]):
                valid_terminals.append((v, t))
        
        num_valid_terminals = len(valid_terminals)
        # print(f"Total valid terminals: {num_valid_terminals}")
        
        # 2. 生成超边
        # 网表规则：每个超边通常由一个macro连接到多个cell
        # 少数超边连接多个macro或多个cell
        
        edges = []  # 存储普通边
        edge_pin_ids = []  # 存储每条边所属的超边ID
        hyperedge_id = 0
        
        # 首先，为每个macro生成主要超边（连接到多个cell）
        # 每个macro至少有一个超边
        for macro_idx in range(actual_macros):
            if num_terminals[macro_idx] == 0:
                continue
                
            # 确定这个macro的超边数量（1-3个）
            num_hyperedges_for_macro = torch.randint(1, 4, (1,)).item()
            
            for h in range(num_hyperedges_for_macro):
                # 确定超边大小（连接到这个macro的cell数量）
                # 大多数超边连接2-5个cell，少数连接更多
                rand_val = torch.rand(1).item()
                if rand_val < 0.8:
                    num_cells_in_hyperedge = torch.randint(2, 6, (1,)).item()  # 2-5个cell
                elif rand_val < 0.95:
                    num_cells_in_hyperedge = torch.randint(5, 15, (1,)).item()  # 5-14个cell
                else:
                    num_cells_in_hyperedge = torch.randint(15, 30, (1,)).item()  # 15-29个cell
                
                # 确保不超过可用的cell数量
                available_cells = num_instances - actual_macros
                if available_cells < num_cells_in_hyperedge:
                    num_cells_in_hyperedge = max(1, available_cells)
                
                # 随机选择cell
                cell_indices = torch.randint(actual_macros, num_instances, (num_cells_in_hyperedge,))
                cell_indices = torch.unique(cell_indices)  # 去重
                
                # 为macro选择一个引脚
                if num_terminals[macro_idx] > 0:
                    macro_pin = torch.randint(0, num_terminals[macro_idx], (1,)).item()
                else:
                    continue
                
                # 为每个选中的cell生成连接到macro的边
                for cell_idx in cell_indices:
                    if num_terminals[cell_idx] > 0:
                        # 为cell选择一个引脚
                        cell_pin = torch.randint(0, num_terminals[cell_idx], (1,)).item()
                        
                        # 添加边：macro -> cell
                        edges.append((macro_idx, macro_pin, cell_idx, cell_pin))
                        edge_pin_ids.append(hyperedge_id)
                        
                        # 也可以添加反向边（如果需要无向图）
                        # edges.append((cell_idx, cell_pin, macro_idx, macro_pin))
                        # edge_pin_ids.append(hyperedge_id)
                
                hyperedge_id += 1
        
        # 其次，生成cell之间的超边（cell-to-cell超边）
        # 这些超边通常较小，大部分只连接2个cell
        num_cell_hyperedges = torch.randint(num_cells // 10, num_cells // 5, (1,)).item()
        
        for h in range(num_cell_hyperedges):
            # 确定超边大小（连接多少cell）
            rand_val = torch.rand(1).item()
            if rand_val < 0.9:
                hyperedge_size = torch.randint(1, 3, (1,)).item()  # 90%的cell超边只连接1-2个cellpyt
            elif rand_val < 0.98:
                hyperedge_size = torch.randint(3, 6, (1,)).item()  # 8%连接3-5个cell
            else:
                hyperedge_size = torch.randint(5, 10, (1,)).item()  # 2%连接5-9个cell
            
            # 随机选择cell
            cell_indices = torch.randint(actual_macros, num_instances, (hyperedge_size,))
            cell_indices = torch.unique(cell_indices)
            
            # 如果只有1个或0个有效cell，跳过
            if len(cell_indices) < 2:
                continue
            
            # 为超边生成连接（生成树或完全连接）
            # 对于小超边（≤3），生成完全连接；对于大超边，生成树形连接
            if len(cell_indices) <= 3:
                # 完全连接
                for i in range(len(cell_indices)):
                    for j in range(i+1, len(cell_indices)):
                        cell1 = cell_indices[i].item()
                        cell2 = cell_indices[j].item()
                        
                        if num_terminals[cell1] > 0 and num_terminals[cell2] > 0:
                            pin1 = torch.randint(0, num_terminals[cell1], (1,)).item()
                            pin2 = torch.randint(0, num_terminals[cell2], (1,)).item()
                            
                            edges.append((cell1, pin1, cell2, pin2))
                            edge_pin_ids.append(hyperedge_id)
            else:
                # 树形连接：随机生成一个生成树
                shuffled_cells = cell_indices[torch.randperm(len(cell_indices))]
                connected_cells = [shuffled_cells[0].item()]
                
                for i in range(1, len(shuffled_cells)):
                    new_cell = shuffled_cells[i].item()
                    # 随机选择一个已连接的cell
                    connect_cell = random.choice(connected_cells)
                    
                    if num_terminals[new_cell] > 0 and num_terminals[connect_cell] > 0:
                        pin1 = torch.randint(0, num_terminals[new_cell], (1,)).item()
                        pin2 = torch.randint(0, num_terminals[connect_cell], (1,)).item()
                        
                        edges.append((connect_cell, pin2, new_cell, pin1))
                        edge_pin_ids.append(hyperedge_id)
                        
                    connected_cells.append(new_cell)
            
            hyperedge_id += 1
        
        # 最后，生成macro-to-macro超边（较少）
        if actual_macros >= 2:
            num_macro_hyperedges = torch.randint(actual_macros // 5, actual_macros // 2, (1,)).item()
            
            for h in range(num_macro_hyperedges):
                # macro超边通常较小，连接2-3个macro
                hyperedge_size = torch.randint(2, min(4, actual_macros+1), (1,)).item()
                
                # 随机选择macro
                macro_indices = torch.randint(0, actual_macros, (hyperedge_size,))
                macro_indices = torch.unique(macro_indices)
                
                if len(macro_indices) < 2:
                    continue
                
                # 生成完全连接（macro超边通常较小）
                for i in range(len(macro_indices)):
                    for j in range(i+1, len(macro_indices)):
                        macro1 = macro_indices[i].item()
                        macro2 = macro_indices[j].item()
                        
                        if num_terminals[macro1] > 0 and num_terminals[macro2] > 0:
                            pin1 = torch.randint(0, num_terminals[macro1], (1,)).item()
                            pin2 = torch.randint(0, num_terminals[macro2], (1,)).item()
                            
                            edges.append((macro1, pin1, macro2, pin2))
                            edge_pin_ids.append(hyperedge_id)
                
                hyperedge_id += 1
        
        # print(f"Generated {len(edges)} ordinary edges from {hyperedge_id} hyperedges")
        
        # 3. 确保边数达到目标
        min_edges = int(num_instances * 1.5)
        max_edges = int(num_instances * 1.8)
        target_edges = torch.randint(min_edges, max_edges + 1, (1,)).item()
        
        current_edges = len(edges)
        if current_edges < target_edges:
            # 添加额外的随机边以达到目标
            additional_edges_needed = target_edges - current_edges
            # print(f"Adding {additional_edges_needed} additional random edges")
            
            # 将终端分为macro终端和cell终端
            macro_terminals = []
            cell_terminals = []
            
            for idx, (v, t) in enumerate(valid_terminals):
                if v < actual_macros:
                    macro_terminals.append(idx)
                else:
                    cell_terminals.append(idx)
            
            num_macro_terminals = len(macro_terminals)
            num_cell_terminals = len(cell_terminals)
            
            # 设置选择权重
            src_macro_weight = 0.75  # 75%的概率选择macro作为源
            
            for _ in range(additional_edges_needed):
                # 随机选择源终端，倾向于选择macro
                rand_val = torch.rand(1).item()
                
                if rand_val < src_macro_weight and macro_terminals:
                    src_idx = random.choice(macro_terminals)
                elif cell_terminals:
                    src_idx = random.choice(cell_terminals)
                else:
                    src_idx = random.choice(macro_terminals) if macro_terminals else 0
                
                src_v, src_t = valid_terminals[src_idx]
                
                # 随机选择目标终端
                target_idx = torch.randint(0, num_valid_terminals, (1,)).item()
                target_v, target_t = valid_terminals[target_idx]
                
                # 避免自环
                if src_v != target_v:
                    edges.append((src_v, src_t, target_v, target_t))
                    edge_pin_ids.append(hyperedge_id)
                    hyperedge_id += 1
        
        # 4. 转换为edge_exists矩阵（稀疏表示）
        if edges:
            edges_tensor = torch.tensor(edges, dtype=torch.long)  # (E, 4)
            edge_pin_ids_tensor = torch.tensor(edge_pin_ids, dtype=torch.long)  # (E,)
        else:
            edges_tensor = torch.zeros((0, 4), dtype=torch.long)
            edge_pin_ids_tensor = torch.zeros((0,), dtype=torch.long)
        
        # 5. 生成edge_index和edge_attr
        edge_index_forward = edges_tensor[:, [0, 2]].t()  # (2, E)
        edge_index_reverse = edges_tensor[:, [2, 0]].t()  # (2, E)
        edge_index = torch.cat([edge_index_forward, edge_index_reverse], dim=1)
        
        # 生成edge_attr
        edge_attr_source = terminal_offsets[edges_tensor[:, 0], edges_tensor[:, 1]]  # (E, 2)
        edge_attr_sink = terminal_offsets[edges_tensor[:, 2], edges_tensor[:, 3]]    # (E, 2)
        edge_attr_forward = torch.cat([edge_attr_source, edge_attr_sink], dim=1)  # (E, 4)
        edge_attr_reverse = torch.cat([edge_attr_sink, edge_attr_source], dim=1)  # (E, 4)
        edge_attr = torch.cat([edge_attr_forward, edge_attr_reverse], dim=0)
        
        # 创建对应的edge_pin_id（双向边属于同一个超边）
        edge_pin_ids_forward = edge_pin_ids_tensor
        edge_pin_ids_reverse = edge_pin_ids_tensor
        edge_pin_ids = torch.cat([edge_pin_ids_forward, edge_pin_ids_reverse], dim=0)
        
        # 统计超边信息
        if edge_pin_ids_tensor.numel() > 0:
            unique_hyperedge_ids = torch.unique(edge_pin_ids_tensor)
            hyperedge_size_stats = {}
            for hid in unique_hyperedge_ids:
                count = (edge_pin_ids_tensor == hid).sum().item()
                hyperedge_size_stats[count] = hyperedge_size_stats.get(count, 0) + 1
            
            # print(f"Hyperedge size distribution: {sorted(hyperedge_size_stats.items())}")

        if False:
            # ==================== 聚类cell ====================
            print("DEBUG: Starting cell clustering...")
            try:
                # 准备数据用于聚类
                # 只对cell进行聚类
                cell_start_idx = actual_macros
                cell_end_idx = actual_macros + num_cells
                
                # 获取cell的位置和尺寸
                cell_positions = all_positions[cell_start_idx:cell_end_idx]
                cell_sizes = all_sizes[cell_start_idx:cell_end_idx]
                cell_sizes = all_sizes
                
                # # 创建cell的Data对象，只包含cell之间的边
                cell_edge_mask = (edge_index[0] >= cell_start_idx) & (edge_index[0] < cell_end_idx) & \
                                (edge_index[1] >= cell_start_idx) & (edge_index[1] < cell_end_idx)
                cell_edge_index = edge_index[:, cell_edge_mask]
                
                # # 重新索引cell的边，使其从0开始
                cell_edge_index = cell_edge_index - cell_start_idx
                
                # 创建cell的Data对象
                # macro_mask = torch.ones(actual_macros, dtype=torch.bool)
                # cell_mask = torch.zeros(num_cells, dtype=torch.bool)
                # mask = torch.cat([macro_mask, cell_mask], dim=0)
                # is_macros = torch.cat([torch.ones(actual_macros, dtype=torch.bool), torch.zeros(num_cells, dtype=torch.bool)], dim=0)
                # cell_data = Data(x=all_sizes, edge_index=edge_index, edge_attr=edge_attr, is_macros = is_macros, is_ports=mask)
                
                # 设置聚类参数
                num_clusters = self.num_clusters
                ubfactor = 5
                
                print(f"Clustering {num_cells} cells into {num_clusters} clusters...")
                
                try:
                    assigned_parts = run_hmetis(
                        input_cond=cell_data,
                        num_clusters=num_clusters,
                        algorithm="hmetis",
                        ubfactor=ubfactor,
                        temp_dir="logs/temp",
                        verbose=False
                    )
                    
                    # 将聚类结果转换为tensor
                    cell_clusters = torch.tensor(assigned_parts, dtype=torch.long)
                    
                    # 统计每个cluster的大小
                    unique_clusters, cluster_counts = torch.unique(cell_clusters, return_counts=True)
                    print(f"Clustering completed. Cluster sizes: min={cluster_counts.min().item()}, max={cluster_counts.max().item()}, mean={cluster_counts.float().mean().item():.1f}")
                    
                except Exception as e:
                    print(f"Hmetis clustering failed: {e}")
                    print("Using random clustering as fallback...")
                    cell_clusters = torch.randint(0, num_clusters, (num_cells,))
                    
            except Exception as e:
                print(f"Error during clustering: {e}")
                print("Using random clustering as fallback...")
                cell_clusters = torch.randint(0, 512, (num_cells,))
            
            # ==================== 生成聚类后的数据（包含hard_macro） ====================
            
            # 为每个cluster创建数据
            cluster_positions = []
            cluster_sizes = []
            
            # 计算每个cluster的位置和尺寸
            for cluster_id in range(self.num_clusters):
                # 获取属于该cluster的所有cell的索引
                cluster_cell_indices = torch.where(cell_clusters == cluster_id)[0]
                num_cells_in_cluster = len(cluster_cell_indices)
                
                if num_cells_in_cluster > 0:
                    # 计算cluster的质心位置
                    cluster_cells_positions = cell_positions[cluster_cell_indices]
                    centroid = cluster_cells_positions.mean(dim=0)
                    
                    # 计算cluster的总面积（所有cell的面积之和）
                    cluster_cells_sizes = cell_sizes[cluster_cell_indices]
                    total_area = (cluster_cells_sizes[:, 0] * cluster_cells_sizes[:, 1]).sum()
                    
                    # 将总面积转换为一个合理的尺寸（假设为正方形）
                    side_length = torch.sqrt(total_area).item()
                    cluster_x_size = min(side_length, 0.3)  # 限制最大尺寸
                    cluster_y_size = min(side_length, 0.3)
                    
                    # 如果cluster很小，设置最小尺寸
                    if cluster_x_size < 0.01:
                        cluster_x_size = 0.01
                    if cluster_y_size < 0.01:
                        cluster_y_size = 0.01
                    
                    cluster_positions.append(centroid)
                    cluster_sizes.append([cluster_x_size, cluster_y_size])
                else:
                    # 如果cluster为空，创建一个小型虚拟cluster
                    centroid = torch.rand(2) * 0.2 - 0.1  # 在中心附近随机位置
                    cluster_positions.append(centroid)
                    cluster_sizes.append([0.01, 0.01])
            
            # 转换为tensor
            if cluster_positions:
                cluster_positions = torch.stack(cluster_positions)
                cluster_sizes = torch.tensor(cluster_sizes, dtype=torch.float32)
            else:
                cluster_positions = torch.zeros((0, 2))
                cluster_sizes = torch.zeros((0, 2))
            
            # ==================== 创建cluster_map（原始id到聚类后id的映射） ====================
            # cluster_map的形状为[num_instances]，其中：
            # - 对于hard_macro: 映射到自身的id (0 到 actual_macros-1)
            # - 对于cell: 映射到 cluster_id + actual_macros (即从actual_macros开始)
            cluster_map = torch.full((num_instances,), -1, dtype=torch.long)
            
            # hard_macro映射到自身
            for i in range(actual_macros):
                cluster_map[i] = i
            
            # cell映射到cluster_id + actual_macros
            for i in range(num_cells):
                cell_global_idx = actual_macros + i
                cluster_map[cell_global_idx] = actual_macros + cell_clusters[i].item()
            
            # ==================== 创建cluster_data的边连接 ====================
            # 将原始边映射到聚类后的模块
            cluster_edges = []
            
            # 遍历所有原始边
            for i in range(edge_index.shape[1]):
                src_idx = edge_index[0, i].item()
                tgt_idx = edge_index[1, i].item()
                
                # 通过cluster_map映射到新的索引
                new_src_idx = cluster_map[src_idx]
                new_tgt_idx = cluster_map[tgt_idx]
                
                # 避免自环
                if new_src_idx != new_tgt_idx:
                    cluster_edges.append([new_src_idx, new_tgt_idx])
            
            # 转换为edge_index格式并去重（因为多个cell到同一个cluster的边会重复）
            if cluster_edges:
                cluster_edge_tensor = torch.tensor(cluster_edges, dtype=torch.long).t()
                
                # 去重：将边排序并去除重复
                # 先将边排序，使src <= tgt
                sorted_edges = torch.stack([torch.min(cluster_edge_tensor, dim=0)[0],
                                        torch.max(cluster_edge_tensor, dim=0)[0]])
                
                # 去除重复边
                unique_edges = torch.unique(sorted_edges, dim=1)
                
                # 恢复双向边（无向图需要两个方向）
                forward_edges = unique_edges
                reverse_edges = torch.stack([unique_edges[1], unique_edges[0]])
                cluster_edge_index = torch.cat([forward_edges, reverse_edges], dim=1)
            else:
                cluster_edge_index = torch.zeros((2, 0), dtype=torch.long)
            
            # ==================== 创建cluster_data ====================
            # 合并macro和cluster的位置和尺寸
            cluster_all_positions = torch.cat([macro_positions, cluster_positions], dim=0)
            cluster_all_sizes = torch.cat([macro_sizes, cluster_sizes], dim=0)
            
            # 创建cluster的terminal（只在中心位置）
            # 所有模块（macro和cluster）都只有一个terminal在中心
            cluster_num_modules = cluster_all_positions.shape[0]
            
            # 创建cluster的edge_attr（都是0，因为terminal在中心）
            cluster_edge_attr = torch.zeros((cluster_edge_index.shape[1], 4))
            
            # 创建cluster的mask：macro为True，cluster为False
            cluster_macro_mask = torch.ones(actual_macros, dtype=torch.bool)
            cluster_cluster_mask = torch.zeros(self.num_clusters, dtype=torch.bool)
            cluster_mask = torch.cat([cluster_macro_mask, cluster_cluster_mask], dim=0)
            
            # 创建cluster_data
            cluster_data = Data(
                x=cluster_all_sizes,
                edge_index=cluster_edge_index,
                edge_attr=cluster_edge_attr,
                is_ports=cluster_mask,
                cluster_map=cluster_map  # 存储原始id到聚类后id的映射
            )
            
            # ==================== 返回数据 ====================
            # 创建原始数据的mask
            macro_mask = torch.ones(actual_macros, dtype=torch.bool)
            cell_mask = torch.zeros(num_cells, dtype=torch.bool)
            mask = torch.cat([macro_mask, cell_mask], dim=0)
        
            # 创建原始data，并添加cluster_map
            data = Data(x=all_sizes, edge_index=edge_index, edge_attr=edge_attr, is_ports=mask)
            data.cluster_map = cluster_map
            
            # 在cluster_data中也存储cluster_map
            cluster_data.cluster_map = cluster_map
            
            if self.zero_edge_attr:
                edge_attr = 0 * edge_attr
                cluster_data.edge_attr = 0 * cluster_data.edge_attr
            
            # 返回四个值：原始位置、原始数据、聚类后位置、聚类后数据
            plot_sample(all_positions, data, f"data-gen/outputs/v2.61/pic/case_{idx}")
            plot_sample(cluster_all_positions, cluster_data, f"data-gen/outputs/v2.61/pic/case_{idx}_cluster")
            return all_positions, data, cluster_all_positions, cluster_data
        # ==================== 返回数据 ====================
        mask = torch.cat([placement.get_mask(), torch.zeros(num_cells, dtype=torch.bool)], dim=0)
        is_macros = torch.cat([torch.ones(actual_macros, dtype=torch.bool), torch.zeros(num_cells, dtype=torch.bool)], dim=0)
        # if self.zero_edge_attr:
        #     edge_attr = 0 * edge_attr
        
        data = Data(x=all_sizes, edge_index=edge_index, edge_attr=edge_attr, is_macros= is_macros, is_ports=mask, edge_pin_id = edge_pin_ids, numRow = num_rows)
        # print("==============================")
        # plot_sample(all_positions, data, f"data-gen/outputs/v2.61/pic/case_{idx}")
        return all_positions, data
    

    def generate_list(self, k, n, seed=None):
        assert n >= k, "n 必须 >= k，否则无法保证每个数至少出现一次"
        
        if seed is not None:
            random.seed(seed)

        # Step 1: 每个数至少一次
        result = list(range(k))
        extra = n - k

        # Step 2: 尽量让每个数出现两次
        indices = list(range(k))
        random.shuffle(indices)

        i = 0
        while extra > 0:
            result.append(indices[i % k])
            i += 1
            extra -= 1

        # Step 3: 打乱顺序
        # random.shuffle(result)
        return result


    def get_terminal_offsets(self, x_sizes, y_sizes, max_num_terminals, reference="center"):
        # NOTE here we assume reference point (for computing offset) is center of instance
        # NOTE outputs use whatever units are being used for sizes
        # x_sizes: (num_instances)
        # y_sizes: (num_instances)
        # max_num_terminals: int
        half_perim = (x_sizes + y_sizes)
                    
        terminal_locations = get_distribution("uniform", {"low": 0, "high": half_perim}).sample((max_num_terminals,)) # (max_term, num_instances)
        terminal_flip = get_distribution("bernoulli", {"probs": 0.5}).sample((max_num_terminals, x_sizes.shape[0])) # (max_term, num_instances)
        terminal_flip = (2 * terminal_flip) - 1

        x_sizes = x_sizes.unsqueeze(dim=0)  # 加一个维度 [xx] -> [1, xx]
        y_sizes = y_sizes.unsqueeze(dim=0)
        boundary_offset_x = torch.clamp(terminal_locations, torch.zeros_like(x_sizes), x_sizes) - (x_sizes/2)  # 限制在模块界限内，中心点为坐标原点
        boundary_offset_y = torch.clamp(terminal_locations-x_sizes, torch.zeros_like(y_sizes), y_sizes) - (y_sizes/2)

        boundary_offset_x = terminal_flip * boundary_offset_x  # 随机取正负号
        boundary_offset_y = terminal_flip * boundary_offset_y

        boundary_offset = torch.stack((boundary_offset_x, boundary_offset_y), dim=-1).movedim(1, 0) # (num_instances, max_term, 2)
        
        # for some components, we want terminals on the interior, not the boundary
        if self.interior_terminals_dist is not None:
            sizes = torch.stack((x_sizes, y_sizes), dim=-1).squeeze(dim=0) # (num_instances, 2)
            gm_size = torch.sqrt(x_sizes * y_sizes).squeeze(dim=0)
            is_terminal_interior = get_distribution(**self.interior_terminals_dist).sample(gm_size).view(gm_size.shape[0], 1, 1)
            if self.interior_terminals_loc == "uniform":
                interior_offset = get_distribution("uniform", {"low": -sizes/2, "high": sizes/2}).sample((max_num_terminals,)) # (max_term, num_instances, 2)
                interior_offset = interior_offset.moveaxis(0, 1)
            elif self.interior_terminals_loc == "center":
                interior_offset = torch.zeros_like(boundary_offset)
            else:
                raise NotImplementedError
            terminal_offset = is_terminal_interior * interior_offset + (1-is_terminal_interior) * boundary_offset
        else:
            terminal_offset = boundary_offset
        
        if reference == "bottom_left":
            terminal_offset[:,:,0] += x_sizes/2
            terminal_offset[:,:,1] += y_sizes/2
        return terminal_offset

    def get_terminal_distances(self, terminal_positions, norm_order=1):
        # given global terminal positions (V, T, 2)
        # compute and return pairwise L1 distances between terminals
        # optionally, can specify x for Lx distance using norm_order
        if norm_order == "inf":
            norm_order = float(norm_order)
        V, T, _ = terminal_positions.shape
        t_pos_1 = terminal_positions.view(V, T, 1, 1, 2)
        t_pos_2 = terminal_positions.view(1, 1, V, T, 2)
        delta_pos = t_pos_1 - t_pos_2 # (V, T, V, T, 2)  # 存储了 (x, y) 坐标差，delta_pos[3, 5, 10, 7] 就是 模块3的第5个端口 与 模块10的第7个端口 的坐标差 (dx, dy)
        distance = torch.norm(delta_pos, p=norm_order, dim=-1) # (V, T, V, T)
        return distance

    def process_edge_matrix(self, edge_exists, is_source, num_terminals):
        # edge_existence tensor (V, T, V, T)
        # is_source int tensor (V, T) (0=sink, 1=source)
        # num terminals int tensor (V)
        # process  as follows:
        # remove all edges ending in a source
        # remove all edges originating from a sink
        # remove all edges between same instance
        # remove all edges starting or ending in nonexistent terminal
        V, T, _, _ = edge_exists.shape
        assert is_source.shape == edge_exists.shape[:2]
        assert num_terminals.shape == (V,)

        # generate terminal filter
        terminal_filter = torch.zeros((V, T))
        for i, num_terminal in enumerate(num_terminals):
            terminal_filter[i, :num_terminal] = 1

        source_filter = (terminal_filter * is_source).view(V, T, 1, 1)
        sink_filter = (terminal_filter * (1-is_source)).view(1, 1, V, T)
        self_edge_filter = (1-torch.eye(V)).view(V, 1, V, 1)
        
        edges = edge_exists * source_filter
        edges = edges * sink_filter
        edges = edges * self_edge_filter
        return edges

    def connect_isolated_instances(self, edge_matrix, terminal_distances):
        # generate edges (IN-PLACE!) for disconnected instances
        # edge matrix has shape (V_src, T_src, V_dest, T_dest)
        # algorithm: for each instance with 0 degree, we connect edge from 0th terminal to closest available terminal
        # so that 0th terminal will become a sink
        # and we find the closest terminal that is:
        # - on a different vertex
        # - has out-degree
        V, T, _, _ = edge_matrix.shape
        out_degree = edge_matrix.sum(dim=(2,3)) # per terminal  # [V,T,V,T] [0,1,2,3]
        in_degree = edge_matrix.sum(dim=(0,1,3)) # per instance
        degree = out_degree.sum(dim=-1) + in_degree
        max_dist = 10+terminal_distances.max()
        for i in range(V):
            if degree[i] == 0: # isolated instance
                distances = terminal_distances[i, 0, :, :] # (V, T)
                distances = torch.where(out_degree > 0, distances, max_dist) # connect to existing nets only
                
                min_idx = torch.argmin(distances)
                instance_idx = min_idx // T
                terminal_idx = min_idx % T

                # connect to netlist
                edge_matrix[instance_idx, terminal_idx, i, 0] = 1

    def generate_edge_list(self, edge_exists, terminal_offsets):
        V, T, _, _ = edge_exists.shape
        edges = torch.nonzero(edge_exists) # (E, 4:v,t,v,t) int64
        edge_index_forward = edges[:,(0,2)] # 源模块到目标模块 (E, 2)
        edge_index_reverse = edges[:,(2,0)]

        edge_attr_source = terminal_offsets[edges[:,0], edges[:,1], :] # (E, 2)
        edge_attr_sink = terminal_offsets[edges[:,2], edges[:,3], :] # (E, 2)
        edge_attr_forward = torch.concat((edge_attr_source, edge_attr_sink), dim=-1)  # 每条边的属性是 源端口偏移 + 目标端口偏移
        edge_attr_reverse = torch.concat((edge_attr_sink, edge_attr_source), dim=-1) # (E, 4)
        
        # create undirected edge index and attr
        edge_index = torch.concat((edge_index_forward, edge_index_reverse), dim=0).T # (2, E)
        edge_attr = torch.concat((edge_attr_forward, edge_attr_reverse), dim=0) # (E, 4)
        
        # return copy
        return edge_index.clone(), edge_attr.clone()

class V3(V2):
    def sample(self):
        # Generate stop density
        stop_density = get_distribution(**self.stop_density_dist).sample()

        # Generate instance sizes
        aspect_ratio = get_distribution(**self.aspect_ratio_dist).sample((self.max_instance,))
        long_size = get_distribution(**self.instance_size_dist).sample((self.max_instance,))
        short_size = aspect_ratio * long_size
        long_x = get_distribution("bernoulli", {"probs": 0.5}).sample((self.max_instance,))

        x_sizes = long_x * long_size + (1-long_x) * (short_size)
        y_sizes = (1-long_x) * long_size + (long_x) * (short_size)

        # sort by area, descending order
        areas = x_sizes * y_sizes
        _, indices = torch.sort(areas, descending=True)
        x_sizes = x_sizes[indices]
        y_sizes = y_sizes[indices]

        # place samples individually
        placement = Placement()
        density = 0
        for (x_size, y_size) in zip(x_sizes, y_sizes):
            x_size = float(x_size)
            y_size = float(y_size)
            dist_params = {"low": torch.tensor([(x_size/2)-1.0, (y_size/2)-1.0]), "high": torch.tensor([1.0-(x_size/2), 1.0-(y_size/2)])}
            candidate_dist =  get_distribution("uniform", dist_params)
            
            for attempt_num in range(self.max_attempts_per_instance):
                candidate_pos = candidate_dist.sample()
                # print(instance_idx, attempt_num, candidate_pos)
                if placement.check_legality(candidate_pos[0].item(), candidate_pos[1].item(), x_size, y_size):
                    placement.commit_instance(candidate_pos[0].item(), candidate_pos[1].item(), x_size, y_size)
                    break
            density += (x_size * y_size)/4.0
            if density >= stop_density:
                break
        
        positions = placement.get_positions()
        sizes = placement.get_sizes()
        num_instances = positions.shape[0]

        # sample number of terminals
        instance_area = sizes[:, 0] * sizes[:, 1]
        num_terminals = get_distribution(**self.num_terminals_dist).sample(instance_area).int()
        num_terminals = torch.clip(num_terminals, min=1, max=32)
        max_num_terminals = torch.max(num_terminals)
        terminal_offsets = self.get_terminal_offsets(sizes[:,0], sizes[:,1], max_num_terminals, reference="center")

        # generate edges
        terminal_positions = positions.unsqueeze(dim=1) + terminal_offsets # (V, T, 2)
        terminal_distances = self.get_terminal_distances(terminal_positions)
        # scale terminal distances by sizes of vertices
        scaled_terminal_distances = terminal_distances / self.get_terminal_dist_scale(instance_area)
        edge_exists = get_distribution(**self.edge_dist).sample(scaled_terminal_distances) # (V, T, V, T)
        is_source = get_distribution(**self.source_terminal_dist).sample((num_instances, max_num_terminals))

        # delete edges between same instance, among other things
        edge_exists = self.process_edge_matrix(edge_exists, is_source, num_terminals)
        self.connect_isolated_instances(edge_exists, terminal_distances)

        # convert to edge list and generate attributes
        edge_index, edge_attr = self.generate_edge_list(edge_exists, terminal_offsets)
        mask = placement.get_mask()

        data = Data(x=sizes, edge_index=edge_index, edge_attr=edge_attr, is_ports=mask)
        
        return positions, data

    def get_terminal_dist_scale(self, instance_area):
        V, = instance_area.shape
        instance_scale = torch.sqrt(instance_area) # (V)
        source_scale = instance_scale.view(V, 1, 1, 1)
        dest_scale = instance_scale.view(1, 1, V, 1)
        p = 0.65
        dist_scale = torch.exp(p * torch.log(source_scale) + (1-p) * torch.log(dest_scale))
        return dist_scale

class V2_MaxSize(V2):
    def __init__(
            self, 
            cap_mode = "naive",
            size_upper_limit = 0.1,
            **kwargs,
        ):
        self.cap_mode = cap_mode
        self.size_upper_limit = size_upper_limit
        super().__init__(**kwargs)

    def sample(self):
        # Generate
        positions, data = super().sample()
        # clamp edge_attr and component sizes
        if self.cap_mode == "naive":
            data.x = torch.clamp(data.x, max = self.size_upper_limit)
            data.edge_attr = torch.clamp(data.edge_attr, max = self.size_upper_limit/2, min = -self.size_upper_limit/2)
        else:
            raise NotImplementedError
        return positions, data

class Flora():
    """
    Re-implementation of algorithm from Flora/GraphPlanner papers
    """
    def __init__(
            self,
            grid_size,
            density_dist,
            neighbors_dist,
            connectivity_dist,
            **kwargs
            ):
        self.grid_size = grid_size
        self.density_dist = density_dist
        self.neighbors_dist = neighbors_dist
        self.connectivity_dist = connectivity_dist
    
    def sample(
            self,
            size_dist_timer=None, # optional timers
            place_timer=None,
            terminal_timer=None,
            edge_timer=None,
            ):
        
        # We can precompute and cache a distance stencil maybe Too complicated, not worth the time

        # Generating object sizes and positions
        object_size = 2.0/self.grid_size        
        density = get_distribution(**self.density_dist).sample()
        grid_occupancy = torch.bernoulli(torch.full((self.grid_size, self.grid_size), density)).bool()

        x_coords = torch.linspace(-1+object_size/2, 1-object_size/2, self.grid_size)
        y_coords = torch.linspace(-1+object_size/2, 1-object_size/2, self.grid_size)
        coord_grid = torch.stack(torch.meshgrid(x_coords, y_coords, indexing="ij"), dim=-1) # (G, G, 2)

        obj_coords = coord_grid[grid_occupancy, :] # (V, 2)
        V, _ = obj_coords.shape
        obj_sizes = torch.full((V, 2), object_size)

        # Generating edge distributions TODO implement hierarchical gaussian
        num_neighbors = torch.round(torch.clip(get_distribution(**self.neighbors_dist).sample((V,)), min=1, max=V)).to(dtype=torch.int64)
        connectivity = torch.round(torch.clip(get_distribution(**self.connectivity_dist).sample((V, num_neighbors.max())), min=1, max=256)).to(dtype=torch.int64)
        
        # compute distances
        distances = torch.linalg.norm(obj_coords.view(V, 1, 2) - obj_coords.view(1, V, 2), dim=-1) # (V, V)
        
        _, distance_indices = torch.sort(distances, dim=1) # (V, V) each row is sorted

        # figuring out number of pairwise edges
        pairwise_edges = torch.zeros((V, V), dtype=torch.int64)
        for i in range(V):
            obj_neighbors = num_neighbors[i]
            obj_connectivity, _ = torch.sort(connectivity[i, :obj_neighbors]) # slice before sorting
            pairwise_edges[i, distance_indices[i, 1:obj_neighbors+1]] = obj_connectivity
        pairwise_edges = torch.floor((pairwise_edges + pairwise_edges.T)/2)

        # generating edge indices
        unique_edges, edge_counts = tgu.dense_to_sparse(torch.triu(pairwise_edges, diagonal=0))
        edge_indices = []
        for i, count in enumerate(edge_counts):
            count = count.int().item()
            edge_indices.append(unique_edges[:,i:i+1].repeat(1, count)) # (2, ...)
        edge_indices = torch.concatenate(edge_indices, dim=1) # (2, E_u)
        edge_indices = torch.concatenate((edge_indices, torch.flip(edge_indices, dims=(0,))), dim=1) # (2, E)
        _, E = edge_indices.shape
        edge_attr = torch.zeros((E, 4), dtype=torch.float32)

        mask = torch.zeros((V,)).bool()

        data = Data(x=obj_sizes, edge_index=edge_indices, edge_attr=edge_attr, is_ports=mask)
        return obj_coords, data

class Placement:
    def __init__(self, x = None, sizes = None, mask = None):
        # initializes empty placement, unless x, sizes, and mask are specified
        # x is predicted placements (V, 2)
        # attr is width height (V, 2)
        self.insts = []
        self.x = []
        self.y = []
        self.x_size = []
        self.y_size = []
        self.is_port = []
        if (x is not None) and (sizes is not None) and (mask is not None):
            for size, loc, is_ports in zip(sizes, x, mask):
                if not is_ports:
                    self.insts.append(
                        shapely.box(loc[0], loc[1], loc[0] + size[0], loc[1] + size[1])
                    )
            self.x = list(x[:,0])
            self.y = list(x[:,1])
            self.x_size = list(sizes[:,0])
            self.y_size = list(sizes[:,1])
            self.is_port = list(mask)

        self.chip = shapely.box(-1, -1, 1, 1)
        self.eps = 1e-8

    def check_legality(self, x_pos, y_pos, x_size, y_size, score=False):
        # checks legality of current placement (or optionally current placement with candidate)
        # x_pos, y_pos, x_size, y_size are floats
        # assumes given positions are center of instance
        # returns float with legality of placement (1 = bad, 0 = legal), or bool if score=False
        insts = self.insts + [shapely.box(x_pos - x_size/2, y_pos - y_size/2, x_pos + x_size/2, y_pos + y_size/2)]
        # insts包括现有模块和新的模块  shapely.box(...) 返回的是一个 Polygon 对象 i.area：返回多边形的面积

        insts_area = sum([i.area for i in insts])
        insts_overlap = shapely.intersection(shapely.unary_union(insts), self.chip).area  # 把所有矩形合并成一个多边形
        # intersection 判断是否与另一个几何体相交  union(other) 返回并集

        if score:
            return insts_overlap/insts_area
        else:
            return abs(insts_overlap - insts_area) < self.eps  #  eps = 1e-8
    
    def commit_instance(self, x_pos, y_pos, x_size, y_size, is_port=False):
        # adds instance with specified params without checking if legal or not
        # assumes coordinates specified are center of instance
        if not is_port:
            self.insts.append(shapely.box(x_pos - x_size/2, y_pos - y_size/2, x_pos + x_size/2, y_pos + y_size/2))
        self.x.append(x_pos)
        self.y.append(y_pos)
        self.x_size.append(x_size)
        self.y_size.append(y_size)
        self.is_port.append(is_port)

    def get_density(self):
        insts_area = sum([i.area for i in self.insts])
        density = insts_area/self.chip.area
        return density
    
    def get_positions(self):
        # returns tensor(V, 2) of x,y placements
        positions = torch.stack((torch.tensor(self.x), torch.tensor(self.y)), dim=-1)
        return positions

    def get_sizes(self):
        # returns tensor(V, 2) of x,y sizes
        sizes = torch.stack((torch.tensor(self.x_size), torch.tensor(self.y_size)), dim=-1)
        return sizes
    
    def get_mask(self):
        mask = torch.tensor(self.is_port)
        return mask
    
class FastPlacement:
    # TODO
    def __init__(self):
        return

    def check_legality(self, pos, size):
        return
    
    def commit_instance(self, pos, size, is_port=False):
        return
    
    def get_positions(self):
        return
    
    def get_sizes(self):
        return
    
    def get_mask(self):
        return

def plot_placement(positions, sizes, name = "debug_placement"):
    picture = utils.visualize(positions, sizes)
    utils.debug_plot_img(picture, name)

def plot_sample(x, cond, name = "debug_placement", plot_edges=False):
    picture = utils.visualize_placement(x, cond, plot_edges=plot_edges)
    utils.debug_plot_img(picture, name)