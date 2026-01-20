import torch
import guidance
import os

import torch.distributions as dist

from utils import visualize_placement, debug_plot_img, hpwl_fast, check_legality_new

import torch.utils.checkpoint as checkpoint

import math

class DDPO():
    
    def __init__(self, model, batch_size, ema_factor=0.99, eta = 1.0, warmup_size=200, chunk_size = 50):
        """
        model: nn.Module with reverse_samples() function that can output log probs
        reward_fn: callable
        batch_size: int
        """
        self.model = model
        # self.reward_fn = reward_fn
        self.batch_size = batch_size
        self.chunk_size = chunk_size  # 新增：分块大小


        self.ema_factor = ema_factor
        self.warmup_size = warmup_size

        self.rew_buffer = None
        self.ema_mean = None
        self.ema_std = None

        self.eta = eta

        # 动态权重 根据每个case设置
        self.case_weights = {}
        self.case_stats = {}
        self.hpwl_log_ref = 4.0 

    
    # 新的方法只需要reward数值，不需要计算图
    def loss_old_1114(self, x, cond, save_folder, step, print_every, idx = 0):
        # Note: here x is only used because reverse sampling requires it for port positions
        # Don't need intermediates
        # log_prob is (T, B) tensor with gradients
        # idx 用于控制绘图
        _num_timesteps = self.model.max_diffusion_steps
        # 确保 x_cond_input 和 cond 都在正确的设备上
        x_0 = self.model.reverse_samples_ddpo(self.batch_size, x, cond, num_timesteps = _num_timesteps, output_log_prob = False)
        
        # debug  绘制图像  100次画一次
        step_int = int(step)
        if step_int % print_every == 0 and idx == 0:
            with torch.no_grad():
                image = visualize_placement(x[0], cond, plot_pins=True, plot_edges=False, img_size=(2048, 2048))
                debug_plot_img(image, os.path.join(save_folder, f"{step_int}_origin"))

                image0 = visualize_placement(x_0[0], cond, plot_pins=True, plot_edges=False, img_size=(2048, 2048))
                debug_plot_img(image0, os.path.join(save_folder, f"{step_int}_placed"))


        # x_0 需要 detach，因为奖励函数不应该通过 x_0 回传梯度到生成过程。
        # 奖励函数只对最终结果评分。
        x_0_detached = x_0.detach() 
        
        # reward_fn 应该能够处理 x_0_detached 和 cond
        # 确保 reward_fn 的输出是 (B,) 且在正确的设备上
        # reward 应该 detach，因为我们不想奖励模型对奖励函数的内部参数求导。
        reward, legality, hpwl = self.reward_fn(x_0_detached, cond) # (B,)
        reward = reward.detach()
        legality = legality.detach()
        hpwl = hpwl.detach()

        self.update_moving_averages(reward)
        advantage = (reward - self.ema_mean) / self.ema_std
        # print(f"advantage:{advantage}")
        
        # 3. 关键修正：使用标准扩散loss，但加权
        B = x.shape[0]
        t = torch.randint(0, _num_timesteps, (B,), device=x.device)

        # 标准扩散loss（带梯度）
        diffusion_loss, metrics = self.model.loss(x, cond, t)

        # 4. RL加权：高奖励的样本降低loss权重，低奖励的样本增加loss权重
        # 注意：advantage是标准化后的，范围在[-3, 3]左右
        rl_weight = 1.0 - 0.2 * advantage  # 调整系数0.1控制RL强度
        # print(f"rl_weight: {rl_weight}")
        rl_weight = torch.clamp(rl_weight, 0.5, 1.5)  # 防止权重太极端

        loss = (diffusion_loss * rl_weight).mean()

        # trajectory_log_prob = log_prob_per_trajectory 
        # 损失函数：- E[ sum(log_prob) * advantage ]
        # loss = -torch.mean(trajectory_log_prob * advantage) 
        # loss = -torch.mean(trajectory_log_prob) 

        metrics = {
            "reward": reward.mean().cpu().item(), 
            "reward_ema_mean": self.ema_mean.cpu().item(),
            "reward_ema_std": self.ema_std.cpu().item(),
            "loss": loss.detach().cpu().item(), # 添加 loss 到 metrics，便于跟踪
            "legality": legality.mean().item(),
            "hpwl": hpwl.mean().item()
        }
        return loss, metrics

    ######### 真的DDPO #########
    def loss_old_1115(self, x, cond, save_folder, step, print_every, DEBUG = False, idx = 0):
        # 1. 生成样本并获取正确的log_prob
        x_0, trajectory_log_prob = self.model.reverse_samples_ddpo(
            self.batch_size, x, cond, output_log_prob=True
        )
        if DEBUG:
            # 2. 检查轨迹log_prob的梯度
            print(f"✅ trajectory_log_prob: {trajectory_log_prob}")
            print(f"✅ trajectory_log_prob requires_grad: {trajectory_log_prob.requires_grad}")
            print(f"✅ trajectory_log_prob grad_fn: {trajectory_log_prob.grad_fn}")

        # 2. 绘制图像
        step_int = int(step)
        if step_int % print_every == 0 and idx == 0:
            with torch.no_grad():
                image = visualize_placement(x[0], cond, plot_pins=True, plot_edges=False, img_size=(2048, 2048))
                debug_plot_img(image, os.path.join(save_folder, f"{step_int}_origin"))

                image0 = visualize_placement(x_0[0], cond, plot_pins=True, plot_edges=False, img_size=(2048, 2048))
                debug_plot_img(image0, os.path.join(save_folder, f"{step_int}_placed"))
        
        # 3. 计算奖励 - 确保reward不破坏计算图
        with torch.no_grad():  # 奖励计算不参与梯度
            x_0_detached = x_0.detach()
            reward, legality, hpwl = self.reward_fn(x_0_detached, cond)
            reward = reward.detach()
        
        self.update_moving_averages(reward)
        advantage = (reward - self.ema_mean) / (self.ema_std + 1e-8)
        
        # 4. 数值稳定性保护
        advantage = torch.clamp(advantage, -5.0, 5.0)  # 限制在合理范围
        trajectory_log_prob = torch.clamp(trajectory_log_prob, -10.0, 10.0)
        
        if DEBUG:
            print(f"✅ Advantage: {advantage}")
            print(f"✅ Advantage requires_grad: {advantage.requires_grad}")  # 应该是False
        
        # 5. DDPO loss - 策略梯度
        # L = -E[ log_prob * advantage ]
        policy_loss = -(trajectory_log_prob * advantage.detach()).mean()

        if DEBUG:
            print(f"✅ Policy loss: {policy_loss.item()}")
            print(f"✅ Policy loss requires_grad: {policy_loss.requires_grad}")
            print(f"✅ Policy loss grad_fn: {policy_loss.grad_fn}")
        
        # 6. 可选：添加熵正则化防止策略坍缩
        entropy = -trajectory_log_prob.mean()
        entropy_weight = 0.01
        total_loss = policy_loss - entropy_weight * entropy
        
        metrics = {
            "reward": reward.mean().cpu().item(),
            "reward_ema_mean": self.ema_mean.cpu().item(),
            "reward_ema_std": self.ema_std.cpu().item(),
            "loss": total_loss.detach().cpu().item(),
            "policy_loss": policy_loss.detach().cpu().item(),
            "entropy": entropy.detach().cpu().item(),
            "legality": legality.mean().item(),
            "hpwl": hpwl.mean().item(),
            "advantage_mean": advantage.mean().cpu().item(),
            "log_prob_mean": trajectory_log_prob.mean().cpu().item()
        }
        
        return total_loss, metrics
    ############################

    def loss_old_1120(self, x, cond, save_folder, step, dram_step, DEBUG = False, idx = 0, _total_timesteps = 30):
        total_timesteps = _total_timesteps # self.model.max_diffusion_steps
        device = next(self.model.parameters()).device
        all_log_probs = []
        current_x = self.model._epsilon_dist.sample((self.batch_size, cond.x.shape[0], self.model.input_shape[1])).squeeze(dim=-1)
        current_x = current_x.to(device).requires_grad_(True) # <--- 重点：在这里设置 requires_grad=True

        mask = self.model.get_mask(x, cond)
        if mask is not None:
            mask = mask.to(device)
            x_device = x.to(device).requires_grad_(True)  # 让 x 也保留梯度
            # 用 masked_scatter 替代 torch.where，更安全地保留梯度
            current_x = current_x.masked_scatter(mask, x_device)
        xt = x
        # 关键修复：使用噪声调度器的时间步序列
        if int(step) == 0:  # 只要第一次才调用，节省时间
            self.model._noise_scheduler.set_timesteps(total_timesteps)
        timesteps = self.model._noise_scheduler.timesteps  # 这应该是正确的时间步序列
        MEM = False
        if MEM:
            mem = torch.cuda.memory_allocated() / 1024**3
            node_size = len(x[0])
            print(f"DEBUG: Memory before loop: {mem:.2f} MB node_size: {node_size}")
        # 按照噪声调度器的时间步进行采样
        for i, (t_current, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            if MEM:
                nowMen = torch.cuda.memory_allocated() / 1024**3
                print(f"i: {i},t_current:{t_current.item():.4f} node_size: {node_size}")
                print(f"DEBUG: Allocated memory: {nowMen:.2f} GB Increase memory: {nowMen-mem:.2f} GB Ratio(increase/node_size):{1000*(nowMen-mem)/(node_size):.4f}")
                mem = nowMen
            
            t_vec = t_current.expand(self.batch_size).to(device)
        
            eps_pred = self.model(current_x, cond, t_vec)
            # 添加钩子，强制保留梯度（用于调试）
            # def grad_hook(grad):
            #     print(f"eps_pred 梯度范数: {grad.norm().item()}")
            #     return grad
            # eps_pred.register_hook(grad_hook)

            # def model_forward_for_checkpoint(x_input, cond_input, t_vec_input):
            #     # 确保这里只调用模型，不要有其他副作用
            #     return self.model(x_input, cond_input, t_vec_input)
            # eps_pred = checkpoint.checkpoint(model_forward_for_checkpoint, current_x, cond, t_vec, use_reentrant=False)

            alpha_t = self.model._noise_scheduler.alpha(t_current)
            sigma_t = self.model._noise_scheduler.sigma(t_current).to(device)  # 方案3 要用到


            alpha_next = self.model._noise_scheduler.alpha(t_next)
            sigma_next = self.model._noise_scheduler.sigma(t_next)
            eta = self.model._noise_scheduler.eta(t_current, t_next)  #.to(device) # 方案2 要用到
            
            deterministic_part = (current_x - sigma_t * eps_pred) / alpha_t

            noise = torch.randn_like(current_x)  # 标准正态分布 均值（mean） = 0 标准差（std） = 1  99.7% 的数值落在 -3 到 3 之间
            next_x = alpha_next * deterministic_part + \
                    torch.sqrt(torch.clamp(sigma_next**2 - eta**2, min=1e-8)) * eps_pred + \
                    eta * noise
            
            # 计算真实噪声
            # eps_true = (current_x - alpha_t * next_x) / sigma_t
            eps_true = (next_x - alpha_next * deterministic_part - eta * noise) / torch.sqrt(torch.clamp(sigma_next**2 - eta**2, min=1e-8))
            # eps_true = (next_x - alpha_next * x) / sigma_next
            
            # 动作分布 以eps_pred为均值，sigma_t为标准差
            action_distribution = dist.Normal(eps_pred, sigma_t)

            # debug
            if DEBUG:
                print(f"t_current: {t_current}")
                print(f"sigma_t: {sigma_t}")
                diff_sq_mean = (eps_true - eps_pred).pow(2).mean().item()
                print(f"(eps_true - eps_pred).pow(2).mean(): {diff_sq_mean}")

            log_prob = action_distribution.log_prob(eps_true)  # 计算高斯分布下的对数概率  -1/2((eps_true-eps_pred)**2/sigma_t**2 + 2 log(sigma_t) + log(2pi))
            # 数值稳定，归一化
            log_prob = log_prob - log_prob.max(dim = -1, keepdim=True)[0]  # 每行归一化到最大为0
            ###########################################################
            
            log_prob = log_prob.sum(dim=[1, 2])
            if DEBUG:
                print(f"log_prob_current_step min: {log_prob.min().item():.4f}, max: {log_prob.max().item():.4f}")
            log_prob = torch.clamp(log_prob, min=-1000.0, max=0.0) # 这是一个尝试值
            if DEBUG:
                print(f"after clamp log_prob_current_step min: {log_prob.min().item():.4f}, max: {log_prob.max().item():.4f}")

            all_log_probs.append(log_prob)
            # current_x = next_x.detach()
            current_x = next_x
        
        # 最终样本和奖励计算
        x_0 = current_x.detach()
        reward, legality, hpwl = self.reward_fn(self.batch_size, x_0, xt, cond)
        # reward, legality, hpwl = self.reward_fn(x_0, cond)
        # reward = reward.detach()

        self.update_moving_averages(reward)
        advantage = (reward - self.ema_mean) / (self.ema_std + 1e-8)
        advantage = torch.clamp(advantage, -2.0, 2.0)  
        
        log_prob_tensor = torch.stack(all_log_probs)  # (T, B)
        trajectory_log_prob = log_prob_tensor.sum(dim=0)  # sum是原论文中的写法 DDPOSF
        trajectory_log_prob = torch.clamp(trajectory_log_prob, -1000.0, 1000.0)

        # 计算损失
        loss = -torch.mean(trajectory_log_prob * advantage)

        # if DEBUG:
        # graph_ok = self._check_computation_graph_detailed(loss, log_prob_tensor, advantage, trajectory_log_prob)
        # if not graph_ok:
        #     print("❌ 计算图断裂，梯度无法传播！")

        # loss = -torch.mean(trajectory_log_prob * advantage)

        if DEBUG:
            self.comprehensive_gradient_check(int(step), loss, trajectory_log_prob, advantage)

        if DEBUG:
            self._check_training_stability(int(step), loss, advantage, trajectory_log_prob)
            print(f"最终损失: {loss.item():.6f}")
            print(f"优势函数 requires_grad: {advantage.requires_grad}")
            print(f"scaled_reward requires_grad: {reward.requires_grad}")

        # 调试：绘制图像
        step_int = int(step)
        if step_int % dram_step == 0 and idx == 0:
            with torch.no_grad():
                image = visualize_placement(x[0], cond, plot_pins=True, plot_edges=False, img_size=(2048, 2048))
                debug_plot_img(image, os.path.join(save_folder, f"{step_int}_origin"))

                image0 = visualize_placement(x_0[0], cond, plot_pins=True, plot_edges=False, img_size=(2048, 2048))
                debug_plot_img(image0, os.path.join(save_folder, f"{step_int}_placed"))
        
        metrics = {
            "reward": reward.detach().mean().cpu().numpy(),
            "reward_ema_mean": self.ema_mean.detach().cpu().numpy(),
            "reward_ema_std": self.ema_std.detach().cpu().numpy(),
            "legality": legality.mean().item(),
            "hpwl": hpwl.mean().item()
        }

        return loss, metrics
    
    def loss_old_1121_doubao(
            self, x, cond, save_folder, step, dram_step, 
            DEBUG = False, idx = 0, _total_timesteps = 30, eta = 0.0):
        # eta 是权重
        total_timesteps = _total_timesteps # self.model.max_diffusion_steps
        device = next(self.model.parameters()).device
        all_log_probs = []
        
        batch_size = self.batch_size
        # 随机采样时间步 t（0 到 total_timesteps-1）
        t_xt = torch.tensor([total_timesteps - 1] * batch_size, device=device)  # 固定为 T-1，避免随机长度
        # 获取 t 对应的 α_t 和 σ_t
        alpha_t = self.model._noise_scheduler.alpha(t_xt)
        sigma_t = self.model._noise_scheduler.sigma(t_xt)
        # 采样客观真实噪声 eps_true
        eps_true_global = torch.randn_like(x, device=device)  # 客观真实噪声（标签）
        # 生成 xt（正向过程加噪）
        xt = alpha_t.unsqueeze(-1).unsqueeze(-1) * x + sigma_t.unsqueeze(-1).unsqueeze(-1) * eps_true_global
        current_x = xt.requires_grad_(True)    # # 后续反向过程用 xt 作为初始输入 current_x，开启梯度追踪
        

        # mask 处理（保持不变，但确保梯度追踪）
        mask = self.model.get_mask(xt, cond)
        if mask is not None:
            mask = mask.to(device)
            x0_device = x.to(device).requires_grad_(True)
            current_x = current_x * (~mask).float() + x0_device * mask.float()
        current_x = current_x.to(device)

        if int(step) == 0:
            self.model._noise_scheduler.set_timesteps(total_timesteps)
        timesteps = self.model._noise_scheduler.timesteps  # 格式：[T-1, T-2, ..., 0]，长度=total_timesteps

        MEM = False
        if MEM:
            mem = torch.cuda.memory_allocated() / 1024**3
            node_size = len(x[0])
            print(f"DEBUG: Memory before loop: {mem:.2f} MB node_size: {node_size}")
        # 按照噪声调度器的时间步进行采样
        for i, (t_current, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            if MEM:
                nowMen = torch.cuda.memory_allocated() / 1024**3
                print(f"i: {i},t_current:{t_current.item():.4f} node_size: {node_size}")
                print(f"DEBUG: Allocated memory: {nowMen:.2f} GB Increase memory: {nowMen-mem:.2f} GB Ratio(increase/node_size):{1000*(nowMen-mem)/(node_size):.4f}")
                mem = nowMen
            
            t_vec = t_current.expand(self.batch_size).to(device)
        
            eps_pred = self.model(current_x, cond, t_vec)
        
            # def model_forward_for_checkpoint(x_input, cond_input, t_vec_input):
            #     # 确保这里只调用模型，不要有其他副作用
            #     return self.model(x_input, cond_input, t_vec_input)
            # eps_pred = checkpoint.checkpoint(model_forward_for_checkpoint, current_x, cond, t_vec, use_reentrant=False)

            # 注册梯度钩子，验证 eps_pred 是否有梯度（核心调试）
            # if DEBUG and i == 0:
            #     eps_pred.retain_grad()
            #     self.eps_pred_for_debug = eps_pred  # 保存用于后续检查

            alpha_t = self.model._noise_scheduler.alpha(t_current).to(device)  
            sigma_t = self.model._noise_scheduler.sigma(t_current).to(device)  
            # sigma_t = torch.clamp(sigma_t, min=0.1, max=1.0)  # 强制sigma_t≥0.1，避免梯度稀释/爆炸
            alpha_next = self.model._noise_scheduler.alpha(t_next)
            sigma_next = self.model._noise_scheduler.sigma(t_next)
            eta = self.model._noise_scheduler.eta(t_current, t_next) 

            # 扩展系数到 (B,1,1)，适配批量数据
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1).to(device)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1).to(device)  
            alpha_next = alpha_next.unsqueeze(-1).unsqueeze(-1).to(device)  
            sigma_next = sigma_next.unsqueeze(-1).unsqueeze(-1).to(device)  
            eta = eta.unsqueeze(-1).unsqueeze(-1).to(device)  
            
            deterministic_part = (current_x - sigma_t * eps_pred) / alpha_t
            noise = torch.randn_like(current_x,  device=device)  # 标准正态分布 均值（mean） = 0 标准差（std） = 1  99.7% 的数值落在 -3 到 3 之间
            next_x = alpha_next * deterministic_part + \
                    torch.sqrt(torch.clamp(sigma_next**2 - eta**2, min=1e-8)) * eps_pred + \
                    eta * noise
            
            # 真实噪声
            eps_true = eps_true_global
            # 动作分布 以eps_pred为均值，sigma_t为标准差
            action_distribution = dist.Normal(eps_pred, sigma_t)
            # debug
            if DEBUG:
                print(f"t_current: {t_current}")
                print(f"sigma_t: {sigma_t}")
                diff_sq_mean = (eps_true - eps_pred).pow(2).mean().item()
                print(f"(eps_true - eps_pred).pow(2).mean(): {diff_sq_mean}")
            log_prob = action_distribution.log_prob(eps_true)  # 计算高斯分布下的对数概率  -1/2((eps_true-eps_pred)**2/sigma_t**2 + 2 log(sigma_t) + log(2pi))
            # 数值稳定，归一化
            log_prob = log_prob - log_prob.max(dim = -1, keepdim=True)[0]  # 每行归一化到最大为0
            log_prob = log_prob.sum(dim=[1, 2])
            if DEBUG:
                print(f"log_prob_current_step min: {log_prob.min().item():.4f}, max: {log_prob.max().item():.4f}")
            log_prob = torch.clamp(log_prob, min=-1e5, max=1e5) # 这是一个尝试值
            if DEBUG:
                print(f"after clamp log_prob_current_step min: {log_prob.min().item():.4f}, max: {log_prob.max().item():.4f}")
                log_prob_std = log_prob.std().item()
                print(f"i: {i}, log_prob 标准差: {log_prob_std:.6f}")
                if log_prob_std < 1e-3:
                    print("⚠️ log_prob 接近常数，梯度会为0！")
            all_log_probs.append(log_prob)
            # current_x = next_x.detach()
            current_x = next_x
        
        # 最终样本和奖励计算
        x_0 = current_x.detach()
        reward, legality, hpwl = self.reward_fn(self.batch_size, x_0, x, cond)
        # reward, legality, hpwl = self.reward_fn(x_0, cond)
        # reward = reward.detach()

        self.update_moving_averages(reward)
        advantage = (reward - self.ema_mean) / (self.ema_std + 1e-8)
        advantage = torch.clamp(advantage, -2.0, 2.0).detach()  # 强制detach，避免梯度污染
        
        log_prob_tensor = torch.stack(all_log_probs)  # (T, B)
        trajectory_log_prob = log_prob_tensor.sum(dim=0)  # sum是原论文中的写法 DDPOSF
        # trajectory_log_prob = torch.clamp(trajectory_log_prob, -1000.0, 1000.0)

        # 计算损失
        # loss = -torch.mean(trajectory_log_prob * advantage)
        loss = -torch.mean(trajectory_log_prob * advantage) / 200.0  # 缩小200倍，避免梯度爆炸

        # 7. 梯度调试（设备一致）
        if DEBUG:
            loss.backward(retain_graph=True)
            # 检查eps_pred梯度
            if hasattr(self, 'eps_pred_for_debug') and self.eps_pred_for_debug.grad is not None:
                eps_pred_grad_norm = self.eps_pred_for_debug.grad.norm().item()
                print(f"✅ eps_pred 梯度范数: {eps_pred_grad_norm:.6f}（设备：{self.eps_pred_for_debug.grad.device}）")
                if eps_pred_grad_norm < 1e-10:
                    print("⚠️ eps_pred 梯度接近0，尝试放大log_prob系数")
            else:
                print("❌ eps_pred 无梯度")
            # 检查模型参数梯度
            model_grad_norm = 0.0
            has_model_grad = 0
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    param_grad_norm = param.grad.norm().item()
                    model_grad_norm += param_grad_norm ** 2
                    if param_grad_norm > 1e-10:
                        has_model_grad += 1
                        # 打印前5个有梯度的参数（避免输出过多）
                        if has_model_grad <= 5:
                            print(f"✅ 参数 {name} 梯度范数: {param_grad_norm:.6f}")
            model_grad_norm = model_grad_norm ** 0.5
            print(f"✅ 模型总梯度范数: {model_grad_norm:.6f}")
            print(f"✅ 有显著梯度的参数: {has_model_grad}/{len(list(self.model.parameters()))}")
            # 清除梯度
            self.model.zero_grad()

        if DEBUG:
            self.comprehensive_gradient_check(int(step), loss, trajectory_log_prob, advantage)

        if DEBUG:
            self._check_training_stability(int(step), loss, advantage, trajectory_log_prob)
            print(f"最终损失: {loss.item():.6f}")
            print(f"优势函数 requires_grad: {advantage.requires_grad}")
            print(f"scaled_reward requires_grad: {reward.requires_grad}")

        # 调试：绘制图像
        step_int = int(step)
        if step_int % dram_step == 0 and idx == 0:
            with torch.no_grad():
                image = visualize_placement(x[0], cond, plot_pins=True, plot_edges=False, img_size=(2048, 2048))
                debug_plot_img(image, os.path.join(save_folder, f"{step_int}_origin"))

                image0 = visualize_placement(x_0[0], cond, plot_pins=True, plot_edges=False, img_size=(2048, 2048))
                debug_plot_img(image0, os.path.join(save_folder, f"{step_int}_placed"))
        
        metrics = {
            "reward": reward.detach().mean().cpu().numpy(),
            "reward_ema_mean": self.ema_mean.detach().cpu().numpy(),
            "reward_ema_std": self.ema_std.detach().cpu().numpy(),
            "legality": legality.mean().item(),
            "hpwl": hpwl.mean().item(),
            "advantage": advantage.std().item()   # 正常范围<2.0   过大→增大 ema_decay（如 0.99→0.999）
        }
        # loss 正常范围 10 ~ 100  过大→增大损失缩放分母；过小→减小分母    
        # 模型总梯度范数（裁剪后）	0.5~5.0	过大→降低 log_prob 放大系数；过小→增大  在train_graph中修改
        return loss, metrics

    def loss(
            self, x, cond, save_folder, step, dram_step, 
            DEBUG = False, idx = 0, _total_timesteps = 30, eta = 0.0):
        # eta 是权重
        total_timesteps = _total_timesteps # self.model.max_diffusion_steps
        device = next(self.model.parameters()).device
        all_log_probs = []
        
        batch_size = self.batch_size

        # mask 处理（保持不变，但确保梯度追踪）
        mask = self.model.get_mask(x, cond)
        if mask is not None:
            mask = mask.to(device)
            x0_device = x.to(device).requires_grad_(True)
            current_x = current_x * (~mask).float() + x0_device * mask.float()
        current_x = current_x.to(device)

        if int(step) == 0:
            self.model._noise_scheduler.set_timesteps(total_timesteps)
        timesteps = self.model._noise_scheduler.timesteps  # 格式：[T-1, T-2, ..., 0]，长度=total_timesteps

        MEM = False
        if MEM:
            mem = torch.cuda.memory_allocated() / 1024**3
            node_size = len(x[0])
            print(f"DEBUG: Memory before loop: {mem:.2f} MB node_size: {node_size}")
        # 按照噪声调度器的时间步进行采样
        for i, (t_current, t_minus_one) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            if MEM:
                nowMen = torch.cuda.memory_allocated() / 1024**3
                print(f"i: {i},t_current:{t_current.item():.4f} node_size: {node_size}")
                print(f"DEBUG: Allocated memory: {nowMen:.2f} GB Increase memory: {nowMen-mem:.2f} GB Ratio(increase/node_size):{1000*(nowMen-mem)/(node_size):.4f}")
                mem = nowMen
            
            t_vec = t_current.expand(self.batch_size).to(device)
        
            eps_pred_t = self.model(current_x, cond, t_vec)
        
            # def model_forward_for_checkpoint(x_input, cond_input, t_vec_input):
            #     # 确保这里只调用模型，不要有其他副作用
            #     return self.model(x_input, cond_input, t_vec_input)
            # eps_pred_t = checkpoint.checkpoint(model_forward_for_checkpoint, current_x, cond, t_vec, use_reentrant=False)


            alpha_t = self.model._noise_scheduler.alpha(t_current).to(device)  
            sigma_t = self.model._noise_scheduler.sigma(t_current).to(device)  
            # sigma_t = torch.clamp(sigma_t, min=0.1, max=1.0)  # 强制sigma_t≥0.1，避免梯度稀释/爆炸
            alpha_t_minus_one = self.model._noise_scheduler.alpha(t_minus_one)
            sigma_t_minus_one = self.model._noise_scheduler.sigma(t_minus_one)
            eta = self.model._noise_scheduler.eta(t_current, t_minus_one) 
            
            # 随机变量的权重
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            # α_t**0.5 = alpha_t  (1 - α_t)**0.5 = sigma_t
            # so: σ_t = sqrt((sigma_t-1)**2/sigma_t**2)*(1-alpha_t**2/alpha_t-1**2))
            variance =  ((sigma_t_minus_one/sigma_t)**2 * (1 - (alpha_t/alpha_t_minus_one)**2))
            std_dev_t = eta * variance**(0.5)

            # compute "direction pointing to x_t"
            # direction = sqrt(1 - α_t−1 - σ_t**2) * eps_pred_t
            # so: direction = sqrt(1 - alpha_t-1**2 - σ_t**2) * eps_pred_t
            pred_sample_direction = (1 - alpha_t_minus_one**2 - std_dev_t**2)**(0.5) *eps_pred_t
            pred_original_sample = (current_x - sigma_t * eps_pred_t) / alpha_t

            # compute x_t without "random noise" 
            # 没有随机噪声的 x_t-1
            prev_sample_mean = alpha_t_minus_one * pred_original_sample + pred_sample_direction

            variance_noise = torch.randn_like(current_x,  device=device)  # 标准正态分布 均值（mean） = 0 标准差（std） = 1  99.7% 的数值落在 -3 到 3 之间

            # 有随机噪声的 x_t-1
            prev_sample = prev_sample_mean + std_dev_t * variance_noise
            
            # 计算高斯分布下的对数概率   
            # -1/2((eps_true-eps_pred)**2/sigma_t**2 + 2 log(sigma_t) + log(2pi))
            log_prob = (
                -((prev_sample.detach() - prev_sample_mean)**2) / (2 * (std_dev_t**2))
                - torch.log(std_dev_t)
                - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
            )
            # mean along all but batch dimension
            log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

            if DEBUG:
                print(f"after clamp log_prob_current_step min: {log_prob.min().item():.4f}, max: {log_prob.max().item():.4f}")
                log_prob_std = log_prob.std().item()
                print(f"i: {i}, log_prob 标准差: {log_prob_std:.6f}")
                if log_prob_std < 1e-3:
                    print("⚠️ log_prob 接近常数，梯度会为0！")
            all_log_probs.append(log_prob)
            # current_x = next_x.detach()
            current_x = prev_sample

        
        # 最终样本和奖励计算
        x_0 = current_x.detach()
        reward, legality, hpwl = self.reward_fn(self.batch_size, x_0, x, cond)
        # reward, legality, hpwl = self.reward_fn(x_0, cond)
        # reward = reward.detach()

        self.update_moving_averages(reward)
        advantage = (reward - self.ema_mean) / (self.ema_std + 1e-8)
        advantage = torch.clamp(advantage, -2.0, 2.0).detach()  # 强制detach，避免梯度污染
        
        log_prob_tensor = torch.stack(all_log_probs)  # (T, B)
        trajectory_log_prob = log_prob_tensor.sum(dim=0)  # sum是原论文中的写法 DDPOSF
        # trajectory_log_prob = torch.clamp(trajectory_log_prob, -1000.0, 1000.0)

        # 计算损失
        # loss = -torch.mean(trajectory_log_prob * advantage)
        loss = -torch.mean(trajectory_log_prob * advantage) / 200.0  # 缩小200倍，避免梯度爆炸

        # 7. 梯度调试（设备一致）
        if DEBUG:
            loss.backward(retain_graph=True)
            # 检查eps_pred梯度
            if hasattr(self, 'eps_pred_for_debug') and self.eps_pred_for_debug.grad is not None:
                eps_pred_grad_norm = self.eps_pred_for_debug.grad.norm().item()
                print(f"✅ eps_pred 梯度范数: {eps_pred_grad_norm:.6f}（设备：{self.eps_pred_for_debug.grad.device}）")
                if eps_pred_grad_norm < 1e-10:
                    print("⚠️ eps_pred 梯度接近0，尝试放大log_prob系数")
            else:
                print("❌ eps_pred 无梯度")
            # 检查模型参数梯度
            model_grad_norm = 0.0
            has_model_grad = 0
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    param_grad_norm = param.grad.norm().item()
                    model_grad_norm += param_grad_norm ** 2
                    if param_grad_norm > 1e-10:
                        has_model_grad += 1
                        # 打印前5个有梯度的参数（避免输出过多）
                        if has_model_grad <= 5:
                            print(f"✅ 参数 {name} 梯度范数: {param_grad_norm:.6f}")
            model_grad_norm = model_grad_norm ** 0.5
            print(f"✅ 模型总梯度范数: {model_grad_norm:.6f}")
            print(f"✅ 有显著梯度的参数: {has_model_grad}/{len(list(self.model.parameters()))}")
            # 清除梯度
            self.model.zero_grad()

        if DEBUG:
            self.comprehensive_gradient_check(int(step), loss, trajectory_log_prob, advantage)

        if DEBUG:
            self._check_training_stability(int(step), loss, advantage, trajectory_log_prob)
            print(f"最终损失: {loss.item():.6f}")
            print(f"优势函数 requires_grad: {advantage.requires_grad}")
            print(f"scaled_reward requires_grad: {reward.requires_grad}")

        # 调试：绘制图像
        step_int = int(step)
        if step_int % dram_step == 0 and idx == 0:
            with torch.no_grad():
                image = visualize_placement(x[0], cond, plot_pins=True, plot_edges=False, img_size=(2048, 2048))
                debug_plot_img(image, os.path.join(save_folder, f"{step_int}_origin"))

                image0 = visualize_placement(x_0[0], cond, plot_pins=True, plot_edges=False, img_size=(2048, 2048))
                debug_plot_img(image0, os.path.join(save_folder, f"{step_int}_placed"))
        
        metrics = {
            "reward": reward.detach().mean().cpu().numpy(),
            "reward_ema_mean": self.ema_mean.detach().cpu().numpy(),
            "reward_ema_std": self.ema_std.detach().cpu().numpy(),
            "legality": legality.mean().item(),
            "hpwl": hpwl.mean().item(),
            "advantage": advantage.std().item()   # 正常范围<2.0   过大→增大 ema_decay（如 0.99→0.999）
        }
        # loss 正常范围 10 ~ 100  过大→增大损失缩放分母；过小→减小分母    
        # 模型总梯度范数（裁剪后）	0.5~5.0	过大→降低 log_prob 放大系数；过小→增大  在train_graph中修改
        return loss, metrics


##############################################
################# 用于debug ##################
    def _check_parameter_changes(self, step):
        """检查模型参数是否发生变化"""
        total_change = 0
        max_change = 0
        changed_params = []
        
        for name, param in self.model.named_parameters():
            if name in self.initial_params:
                initial = self.initial_params[name]
                change = torch.abs(param.data - initial).sum().item()
                total_change += change
                max_change = max(max_change, change)

                # 更新参数
                # self.initial_params[name] = param.data
                
                if change > 0:
                    changed_params.append((name, change))
        
        print(f"=== 参数变化检查 (Step {step}) ===")
        print(f"总变化: {total_change:.6f}")
        print(f"最大变化: {max_change:.6f}")
        print(f"变化的参数数量: {len(changed_params)}/{len(list(self.model.named_parameters()))}")
        
        if changed_params:
            print("变化最大的参数:")
            for name, change in sorted(changed_params, key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {name}: {change:.6f}")
        else:
            print("⚠️  警告：没有检测到参数变化！")
        print("=" * 50)
    
    def _check_gradients(self, step):
        """检查梯度是否存在"""
        total_grad = 0
        has_gradient_params = []
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad += grad_norm
                if grad_norm > 0:
                    has_gradient_params.append((name, grad_norm))
        
        print(f"=== 梯度检查 (Step {step}) ===")
        print(f"总梯度范数: {total_grad:.6f}")
        print(f"有梯度的参数数量: {len(has_gradient_params)}/{len(list(self.model.named_parameters()))}")
        
        if has_gradient_params:
            print("梯度最大的参数:")
            for name, grad_norm in sorted(has_gradient_params, key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {name}: {grad_norm:.6f}")
        else:
            print("❌ 错误：没有检测到梯度！")
        print("=" * 50)

    def _check_detailed_gradients(self, step):
        """更详细的梯度检查"""
        print(f"=== 详细梯度检查 (Step {step}) ===")
        
        total_grad = 0
        has_significant_grad = 0
        grad_details = []
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                total_grad += grad_norm
                
                # 检查是否有显著梯度
                if abs(grad_mean) > 1e-10 or abs(grad_std) > 1e-10:
                    has_significant_grad += 1
                    grad_details.append((name, grad_norm, grad_mean, grad_std))
        
        print(f"总梯度范数: {total_grad:.10f}")
        print(f"有显著梯度的参数: {has_significant_grad}/{len(list(self.model.named_parameters()))}")
        
        if grad_details:
            print("梯度最大的参数:")
            for name, norm, mean, std in sorted(grad_details, key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {name}: 范数={norm:.10f}, 均值={mean:.10f}, 标准差={std:.10f}")
        else:
            print("❌ 没有检测到显著梯度")
        
        print("=" * 50)

    def _check_computation_graph_detailed(self, loss, log_probs_tensor, advantage, trajectory_log_prob):
        """详细的计算图检查"""
        print("=== 详细计算图检查 ===")
        
        # 检查loss的计算图
        print(f"1. loss.requires_grad: {loss.requires_grad}")
        print(f"2. loss.grad_fn: {loss.grad_fn}")
        
        if loss.grad_fn is None:
            print("❌ loss没有计算图连接！")
            return False
        
        # 检查各个组件的计算图
        print(f"3. advantage.requires_grad: {advantage.requires_grad}")
        print(f"4. trajectory_log_prob.requires_grad: {trajectory_log_prob.requires_grad}")
        
        if log_probs_tensor is not None:
            print(f"5. log_probs_tensor.requires_grad: {log_probs_tensor.requires_grad}")
        
        # 检查模型参数
        model_has_grad = any(p.requires_grad for p in self.model.parameters())
        print(f"6. 模型有可训练参数: {model_has_grad}")
        
        # 追溯计算图
        print("7. 计算图追溯:")
        current_fn = loss.grad_fn
        visited = set()
        
        for depth in range(20):  # 增加追溯深度
            if current_fn is None:
                print(f"    在深度{depth}计算图结束")
                break
            
            if current_fn in visited:
                print(f"    在深度{depth}检测到循环，停止追溯")
                break
            visited.add(current_fn)
            
            fn_name = type(current_fn).__name__
            print(f"    深度{depth}: {fn_name}")
            
            # 检查是否有next_functions
            if hasattr(current_fn, 'next_functions'):
                next_fns = current_fn.next_functions
                for i, (next_fn, _) in enumerate(next_fns):
                    if next_fn is not None:
                        next_fn_name = type(next_fn).__name__
                        print(f"      下一个{i}: {next_fn_name}")
                        
                        # 如果遇到关键操作，特别标注
                        if 'Backward' in next_fn_name:
                            print(f"        ⚡ 关键反向操作: {next_fn_name}")
            
            # 移动到下一个函数
            if hasattr(current_fn, 'next_functions') and current_fn.next_functions:
                current_fn = current_fn.next_functions[0][0]
            else:
                current_fn = None
        
        return True

    def _check_optimizer_state(self, optimizer, step):
        """检查优化器状态"""
        print(f"=== 优化器状态检查 (Step {step}) ===")
        
        for i, param_group in enumerate(optimizer.param_groups):
            print(f"参数组 {i}:")
            print(f"  学习率: {param_group['lr']}")
            print(f"  参数数量: {len(param_group['params'])}")
            
            # 检查优化器动量
            if 'momentum_buffer' in optimizer.state[param_group['params'][0]]:
                momentum_norm = optimizer.state[param_group['params'][0]]['momentum_buffer'].norm().item()
                print(f"  动量范数: {momentum_norm:.10f}")
        
        print("=" * 50)

    def _check_training_stability(self, step, loss, advantage, trajectory_log_prob):
        """检查训练稳定性"""
        print(f"=== 训练稳定性检查 (Step {step}) ===")
        
        # 检查损失值
        loss_value = loss.item()
        if abs(loss_value) > 1000:
            print(f"⚠️  警告：损失值过大: {loss_value:.2f}")
        
        # 检查优势函数
        adv_mean = advantage.mean().item()
        adv_std = advantage.std().item()
        print(f"优势函数 - 均值: {adv_mean:.3f}, 标准差: {adv_std:.3f}")
        
        # 检查log_prob
        log_prob_mean = trajectory_log_prob.mean().item()
        print(f"log_prob均值: {log_prob_mean:.8f}")
        
        # 检查梯度与损失的比例
        if hasattr(self, 'last_loss') and self.last_loss is not None:
            loss_change = abs(loss_value - self.last_loss)
            print(f"损失变化: {loss_change:.6f}")
        self.last_loss = loss_value
        
        print("=" * 50)

    def check_grad_scaler_issue(self, loss, optim, grad_scaler):
        """检查梯度缩放器相关问题"""
        print("=== 梯度缩放器检查 ===")
        
        # 手动进行反向传播，不使用grad_scaler
        print("1. 尝试不使用grad_scaler的反向传播...")
        optim.zero_grad()
        loss.backward()
        
        # 检查梯度
        total_grad_manual = 0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                total_grad_manual += param.grad.norm().item()
        
        print(f"2. 手动反向传播后的梯度范数: {total_grad_manual:.10f}")
        
        # 清理梯度
        optim.zero_grad()
        
        # 使用grad_scaler进行反向传播
        print("3. 使用grad_scaler进行反向传播...")
        scaled_loss = grad_scaler.scale(loss)
        scaled_loss.backward()
        
        # 检查缩放后的梯度
        total_grad_scaled = 0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                total_grad_scaled += param.grad.norm().item()
        
        print(f"4. grad_scaler反向传播后的梯度范数: {total_grad_scaled:.10f}")
        
        # 检查缩放因子
        print(f"5. grad_scaler缩放因子: {grad_scaler.get_scale()}")
        
        return total_grad_manual, total_grad_scaled

    def test_gradient_flow(self, x, cond):
        """测试梯度流是否正常"""
        print("=== 梯度流测试 ===")
        
        # 创建一个简单的测试输入
        test_batch_size = 2
        device = next(self.model.parameters()).device
        
        # 简单的前向传播
        with torch.enable_grad():
            current_x = self.model._epsilon_dist.sample((test_batch_size, 10, 2)).squeeze(dim=-1).to(device)
            t_vec = torch.tensor(0.5, device=device).expand(test_batch_size)
            
            # 模型预测
            eps_pred = self.model(current_x, cond, t_vec)  # 不使用cond
            
            # 简单的log_prob计算
            noise = torch.randn_like(current_x)
            action_distribution = dist.Normal(eps_pred, 1.0)
            log_prob = action_distribution.log_prob(noise)
            log_prob = log_prob.sum(dim=[1, 2]) if len(log_prob.shape) == 3 else log_prob.sum()
            
            # 计算损失
            fake_advantage = torch.tensor([1.0, -1.0], device=device)  # 模拟优势函数
            test_loss = -torch.mean(log_prob * fake_advantage)
            
            # 反向传播
            test_loss.backward()
            
            # 检查梯度
            total_grad = 0
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad += grad_norm
                    if grad_norm > 1e-10:
                        print(f"  参数 {name} 有梯度: {grad_norm:.6f}")
            
            print(f"测试总梯度范数: {total_grad:.10f}")
            
            # 清理梯度
            self.model.zero_grad()
        
        return total_grad > 0

    def comprehensive_gradient_check(self, step, loss, trajectory_log_prob, advantage):
        """综合梯度检查"""
        print("=== 综合梯度检查 ===")
        
        checks_passed = 0
        total_checks = 4
        
        # 检查1: loss需要梯度
        if loss.requires_grad:
            print("✓ 检查1: loss需要梯度")
            checks_passed += 1
        else:
            print("❌ 检查1: loss不需要梯度")
        
        # 检查2: trajectory_log_prob需要梯度
        if trajectory_log_prob.requires_grad:
            print("✓ 检查2: trajectory_log_prob需要梯度")
            checks_passed += 1
        else:
            print("❌ 检查2: trajectory_log_prob不需要梯度")
        
        # 检查3: advantage不需要梯度（这是正常的）
        if not advantage.requires_grad:
            print("✓ 检查3: advantage不需要梯度（正常）")
            checks_passed += 1
        else:
            print("⚠️ 检查3: advantage需要梯度（可能有问题）")
        
        # 检查4: 计算图连接
        if trajectory_log_prob.grad_fn is not None:
            print("✓ 检查4: trajectory_log_prob有计算图连接")
            checks_passed += 1
        else:
            print("❌ 检查4: trajectory_log_prob没有计算图连接")
        
        print(f"通过检查: {checks_passed}/{total_checks}")
        
        if checks_passed >= 3:
            print("✓ 梯度流基本正常")
            return True
        else:
            print("❌ 梯度流存在问题")
            return False

    def update_moving_averages(self, new_reward):
        # new reward has shape (B)
        # if self.rew_buffer is None:
        #     self.rew_buffer = new_reward
        # buff_len = self.rew_buffer.shape[0]
        # if buff_len < self.warmup_size:
        #     self.rew_buffer = torch.cat((self.rew_buffer, new_reward), dim=0)
        #     self.ema_mean = self.rew_buffer.mean()
        #     self.ema_std = self.rew_buffer.std()
        # else:
        #     self.ema_mean = self.ema_factor * self.ema_mean + (1-self.ema_factor) * new_reward.mean()
        #     self.ema_std = self.ema_factor * self.ema_std + (1-self.ema_factor) * new_reward.std()

        new_reward = new_reward.detach()
        """改进的EMA更新, 避免数值不稳定"""
        if self.rew_buffer is None:
            self.rew_buffer = new_reward.cpu()
            self.ema_mean = new_reward.mean().detach()
            self.ema_std = new_reward.std().detach()
            return
        
        buff_len = self.rew_buffer.shape[0]
        if buff_len < self.warmup_size:
            self.rew_buffer = torch.cat((self.rew_buffer, new_reward.detach().cpu()), dim=0)
            self.ema_mean = self.rew_buffer.mean()
            self.ema_std = self.rew_buffer.std()
        else:
            # 使用更稳定的EMA更新
            batch_mean = new_reward.mean().detach()
            batch_std = new_reward.std().detach()
            
            self.ema_mean = self.ema_factor * self.ema_mean + (1 - self.ema_factor) * batch_mean
            self.ema_std = self.ema_factor * self.ema_std + (1 - self.ema_factor) * batch_std
            
            # 确保std不为0
            self.ema_std = torch.max(self.ema_std, torch.tensor(1e-6, device=self.ema_std.device))

    ##########################################
    ######## 参照 DDPO 官方代码实现 ###########
    @torch.no_grad()
    def sample_with_logprob(self, x, cond, timesteps, eta = None):
        device = x.device
        batch_size = self.batch_size
        batch_shape = (batch_size, cond.x.shape[0], self.model.input_shape[1])
        
        current_x = self.model._epsilon_dist.sample(batch_shape).squeeze(dim = -1) # (B, C, H, W)
        # 去噪循环
        all_x_list = [current_x]
        all_x0_pre = []
        all_log_probs = []
        
        for i, (t_current, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            t_vec = t_current.expand(self.batch_size).to(device)
            eps_pred = self.model(current_x, cond, t_vec)
            # 后续可以加guidance...
            # compute the previous noisy sample x_t -> x_t-1
            # 单步去噪
            variance_noise = self.model._epsilon_dist.sample(batch_shape).squeeze(dim = -1) if i<(len(timesteps)-2) else torch.zeros_like(current_x)
            current_x, log_prob, x0_pre = self.ddim_step_with_logprob(
                eps_pred, current_x, t_current, t_prev,
                variance_noise, eta
            )
            current_x = torch.clamp(current_x, -1, 1) # 保持在界内  坐标范围在-1到1
            all_x_list.append(current_x)
            all_x0_pre.append(x0_pre)  # 从T到0
            all_log_probs.append(log_prob)
        
        x0 = current_x  # 最终去噪图
        all_x0_pre[-1] = x0  # 最后一个应该直接赋值x0即可，而不是x0_pre
        # 后处理...

        return x0, all_x_list, all_log_probs, all_x0_pre

    def ddim_step_with_logprob(
            self, eps_pred, xt, t_current, t_prev, 
            variance_noise, # 随机噪声
            eta = None,
            use_clipped_model_output = False,
            prev_sample = None # xt-1
        ):
        # 本函数中的alpha与beta与ddpo官方代码中的不同
        # α_t**0.5 = alpha_t  (1 - α_t)**0.5 = beta_t
        # 计算alpha, beta等参数
        device = eps_pred.device
        alpha_t = self.model._noise_scheduler.alpha(t_current)
        alpha_t_prev = self.model._noise_scheduler.alpha(t_prev)
        beta_t = self.model._noise_scheduler.beta(t_current)
        beta_t_prev = self.model._noise_scheduler.beta(t_prev)

        # ========== 修复维度不匹配 ==========
        alpha_t = self._left_broadcast(alpha_t, xt.shape).to(device)
        alpha_t_prev = self._left_broadcast(alpha_t_prev, xt.shape).to(device)
        beta_t = self._left_broadcast(beta_t, xt.shape).to(device)
        beta_t_prev = self._left_broadcast(beta_t_prev, xt.shape).to(device)
        
        # alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1).to(device)
        # alpha_t_prev = alpha_t_prev.unsqueeze(-1).unsqueeze(-1).to(device)
        # beta_t = beta_t.unsqueeze(-1).unsqueeze(-1).to(device)
        # beta_t_prev = beta_t_prev.unsqueeze(-1).unsqueeze(-1).to(device)
        

        # 计算预测的x0 
        # 公式： xt = x0*alpha_t + eps_pred*beta_t
        x0_pre = (xt - beta_t * eps_pred) / alpha_t

        # 考虑x0_pre是否需要裁剪
        # x0_pre = x0_pre.clamp(-1, 1)  # 坐标范围 (-1, 1)

        # 计算 variance: "sigma_t(η)" 
        # 原公式: σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        # 代入我们的：σ_t = (beta_t-1)/beta_t)*sqrt((1-(alpha_t/alpha_t-1)**2))
        variance =  (beta_t_prev/(beta_t)) * torch.sqrt(1 - torch.square(alpha_t/alpha_t_prev))  # = self.model._noise_scheduler.eta
        # eta = 1.0 就是 DDPM  eta = 0.0 就是 DDIM
        std_dev_t = eta * variance # σ_t
        std_dev_t = self._left_broadcast(std_dev_t, xt.shape).to(xt.device)

        # 因为可能发生了clamp，会导致出现差异, 保证一致性
        if use_clipped_model_output:
            eps_pred = (xt - x0_pre * alpha_t) / beta_t
        
        # 计算去噪方向
        # 原公式: direction = sqrt(1 - α_t−1 - σ_t**2) * eps_pred_t
        # 代入我们的: direction = sqrt(beta_t-1**2 - σ_t**2) * eps_pred_t
        pred_xt_prev_direction = torch.sqrt(torch.clip(torch.square(beta_t_prev) - torch.square(std_dev_t), min=0)) * eps_pred  
        

        # 计算 x_t-1 不考虑随机噪声
        xt_prev_mean = alpha_t_prev * x0_pre + pred_xt_prev_direction

        # 计算 x_t-1
        if prev_sample is None:
            xt_pre = xt_prev_mean + std_dev_t * variance_noise
        else:
            xt_pre = prev_sample

        # print(f"std_dev_t:{std_dev_t[0][0][0]}")
        # 计算高斯分布下的对数概率
        # -1/2((eps_true-eps_pred)**2/sigma_t**2 + 2 log(sigma_t) + log(2pi))
        log_prob = (
            -((xt_pre.detach() - xt_prev_mean)**2) / (2 * (std_dev_t**2))
            - torch.log(std_dev_t)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        return xt_pre.type(xt.dtype), log_prob, x0_pre
    
    def _left_broadcast(self, t, shape):
        assert t.ndim <= len(shape)
        return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)

    @torch.no_grad()
    def get_one_reward(self, x, cond, baseline_hpwl, _hpwl_weight=1, _legality_weight=3, outLandH = False):
        ''' 计算一个case的reward 用于sampling和eval '''
        current_hpwl = hpwl_fast(x, cond, normalized_hpwl=True)
        legality_temp = check_legality_new(x, None, cond, cond.is_ports, score=True)
        hpwl_ratio_temp = 1.0 -  (current_hpwl) / (baseline_hpwl) # 1.8 
        # hpwl_ratio_temp = max(hpwl_ratio_temp, 0.4)  # 保底 防止对hpwl不优化了
        # hpwl_ratio_temp = min(hpwl_ratio_temp, 1.0)  # 补：匹配注释的上限1，避免异常值
        reward = _hpwl_weight * hpwl_ratio_temp  + _legality_weight * legality_temp
        reward = torch.tensor([reward]).to(x.device)
        if outLandH:
            legality_temp = torch.tensor([legality_temp]).to(x.device)
            hpwl_ratio_temp = torch.tensor([hpwl_ratio_temp]).to(x.device)
            return reward, legality_temp, hpwl_ratio_temp
        return reward

    @torch.no_grad()
    def get_reward(self, x0, cond, xt, x0_pre_list, intermediate=False,_hpwl_weight=1, _legality_weight=3): 
        ''' x0_pre_list 最后一个是x0 前面是由时间步 T...0 # batch_size 只能为1 '''
        x0_detached = x0.detach()   # (B, V, 2)
        xt_detached = xt.detach()   # (B, V, 2)
        baseline_hpwl = hpwl_fast(xt_detached[0], cond, normalized_hpwl=True)
        intermediate_rewards = []
        diff = None
        if intermediate:
            for x0_pre in x0_pre_list:
                intermediate_rewards.append(self.get_one_reward(x0_pre[0], cond, baseline_hpwl, _hpwl_weight, _legality_weight))
            # 用差值表示reward
            intermediate_rewards = torch.cat(intermediate_rewards, dim=0)
            diff = torch.zeros_like(intermediate_rewards)
            diff[1:] = intermediate_rewards[1:] - intermediate_rewards[:-1]
            diff[0] = 0
        ########################### 计算x0
        reward_x0, legal_x0, hpwl_x0 = self.get_one_reward(x0_detached[0], cond, baseline_hpwl, _hpwl_weight, _legality_weight, outLandH=True)
        ###########################
        
        return reward_x0, legal_x0, hpwl_x0, diff # acc_reward(intermediate_rewards)
    

def acc_reward(intermediate_rewards, gamma = 0.99):
    # intermediate_rewards: shape [N]
    for i in range(intermediate_rewards.size(0) - 2, -1, -1):
        intermediate_rewards[i] = intermediate_rewards[i] + gamma * intermediate_rewards[i + 1] 
    return intermediate_rewards

def legality_reward(x, cond):
    # 理论最小值是 0：表示完全合法（无重叠，无出界）
    # 最大值是不确定的：可以是非常大的正数，取决于模块数量、尺寸以及不合法程度。
    return -guidance.legality_guidance_potential(x, cond) # better legality means better reward

def hpwl_reward(x, cond):
    # 理论最小值是 0：当所有连接点都重合时
    # 最大值是不确定的：取决于芯片尺寸和连接点的分布
    return -guidance.hpwl_guidance_potential(x, cond) # lower hpwl means higher reward

def get_reward_fn(legality_weight, hpwl_weight):
    def reward_fn(x, cond):
        # 确保x不需要梯度（detach）
        x_detached = x.detach()

        # 计算可微的奖励组件
        legality_potential = guidance.legality_guidance_potential(x_detached, cond) # legality_guidance_potential (不合法性能量，越大越不合法)
        hpwl_potential = guidance.hpwl_guidance_potential(x_detached, cond)    # hpwl_guidance_potential (线长，越大越不理想)
 
        
        reward = legality_weight * (-legality_potential) + hpwl_weight * (-hpwl_potential)
        # 分离不需要梯度的部分用于指标
        legality_detached = legality_potential.detach()
        hpwl_detached = hpwl_potential.detach()

        return reward, legality_detached, hpwl_detached
    return reward_fn

def get_reward_fn_ddpo(legality_weight, hpwl_weight, reward_version, scale_factor):
    def reward_fn_norm_new(x0, cond, xt, batch_size, baseline_hpwl = None, taget_legal = 0.93):
        x0_detached = x0.detach()  # (B, V, 2)
        xt_detached = xt.detach()

        hpwl_ratios = []
        legalitys = []
        # if baseline_hpwl == None:
        baseline_hpwl = hpwl_fast(xt_detached[0], cond, normalized_hpwl=True)
        # baseline_legal = check_legality_new(xt_detached[0], None, cond, cond.is_ports, score=True)

        for i in range(batch_size):
            # 先计算合法率（必须先判断合法率，再决定HPWL权重）
            legality_temp = check_legality_new(x0_detached[i], None, cond, cond.is_ports, score=True)
            legalitys.append(legality_temp)
            # HPWL计算：保留激励，但新增平衡约束
            current_hpwl = hpwl_fast(x0_detached[i], cond, normalized_hpwl=True)
            # hpwl_ratio_temp = (baseline_hpwl- current_hpwl)/ baseline_hpwl
            # hpwl_ratio_temp = max(hpwl_ratio_temp, -1.0)  
            # hpwl_ratio_temp = min(hpwl_ratio_temp, 1.0)  
            hpwl_ratio_temp = 1.8 - current_hpwl / baseline_hpwl
            # 强制下限  范围由 0.6 - 1 放大到0.0 - 1.0
            hpwl_ratio_temp = (hpwl_ratio_temp - 0.6) / 0.4
            hpwl_ratio_temp = max(0.0, min(1.0, hpwl_ratio_temp))
            # hpwl_ratio_temp = - current_hpwl / x0_detached[0].shape[0]  # 线长比上顶点数
            hpwl_ratios.append(hpwl_ratio_temp)

        legalitys = torch.tensor(legalitys).to(x0.device)
        hpwl_ratios = torch.tensor(hpwl_ratios).to(x0.device)
        # print(f"hpwl_ratios: {hpwl_ratios[0].item()} legalitys: {legalitys[0].item()} ratio: {hpwl_ratios[0].item()/ legalitys[0].item()}")

    
        # 计算最终奖励（HPWL权重动态变化）
        reward =  hpwl_weight * hpwl_ratios + legality_weight * legalitys # torch.clamp(taget_legal - legalitys, min=0)
        return reward, legalitys, hpwl_ratios

    def reward_fn_norm(x0, cond, xt, batch_size, _hpwl_weight=1, _legality_weight=1, _target_legality=0.95):
        # 需要原始数据xt  和最终方案x0
        x0_detached = x0.detach()   # (B, V, 2)
        xt_detached = xt.detach()   # (B, V, 2)

        # hpwl ratio
        hpwl_ratios = []
        legalitys = []
        baseline_hpwl = hpwl_fast(xt_detached[0], cond, normalized_hpwl=True)
        for i in range(batch_size):
            current_hpwl, _ = hpwl_fast(x0_detached[i], cond, normalized_hpwl=False)
            # 优化率最高范围0.2的 (baseline - current) / baseline + 0.8  限制上限1  防止恶化合法化率
            hpwl_ratio_temp =  1.8 -  (current_hpwl) / (baseline_hpwl + 1e-8)  
            hpwl_ratio_temp = max(hpwl_ratio_temp, 0.6)  # 保底 0.6 防止对hpwl不优化了
            hpwl_ratio_temp = min(hpwl_ratio_temp, 1.0)  # 补：匹配注释的上限1，避免异常值
            hpwl_ratios.append(hpwl_ratio_temp)
            legality_temp = check_legality_new(x0_detached[i], xt_detached[i], cond, cond.is_ports, score=True)
            legalitys.append(legality_temp)

        legalitys = torch.tensor(legalitys).to(x0.device)
        hpwl_ratios = torch.tensor(hpwl_ratios).to(x0.device)

        illegal_penalty = -0.1
        very_illegal_penalty = -0.2

        legal_bonus = torch.where(
            legalitys >= _target_legality,  # 达到目标：额外奖励
            legalitys + 0.05,
            torch.where(
                legalitys >= _target_legality - 0.05,  # 底线内：正常奖励
                legalitys,
                torch.where(
                    legalitys >= _target_legality - 0.1,  # 轻度非法（0.85~0.90）：轻度惩罚
                    legalitys + illegal_penalty,
                    legalitys + very_illegal_penalty  # 严重非法（<0.85）：重度惩罚
                )
            )
        )

        reward = _hpwl_weight * hpwl_ratios + _legality_weight * legal_bonus

        return reward, legalitys, hpwl_ratios
    
    def reward_fn(x, cond, xt, batch_size):
        # 确保x不需要梯度（detach）
        x_detached = x.detach()

        # 计算可微的奖励组件
        legality_potential = guidance.legality_guidance_potential(x_detached, cond) # legality_guidance_potential (不合法性能量，越大越不合法)
        hpwl_potential = guidance.hpwl_guidance_potential(x_detached, cond)    # hpwl_guidance_potential (线长，越大越不理想)
 
        
        reward = legality_weight * (-legality_potential) + hpwl_weight * (-hpwl_potential)
        # 分离不需要梯度的部分用于指标
        legality_detached = legality_potential.detach()
        hpwl_detached = hpwl_potential.detach()

        return reward, legality_detached, hpwl_detached
    if reward_version == "norm":
        return reward_fn_norm
    else:
        return reward_fn

