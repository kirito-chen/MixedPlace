import utils
import torch
import hydra
import models
import ddpo
from omegaconf import OmegaConf, open_dict
import common
import os
import time



from tqdm import tqdm  # 导入tqdm
from datetime import datetime 
import matplotlib.pyplot as plt
import numpy as np

import signal
import sys

from statTracker import PerPromptStatTracker

# 全局变量存储训练进度
dram_step_reward = []
dram_step_legal = [] 
dram_step_hpwl = []
dram_step_loss = []
dram_step_ppo_loss = []
dram_step_mse_loss = []
dram_val_reward = []
log_dir_global = ""  # 需要在训练开始前设置


@hydra.main(version_base=None, config_path="configs", config_name="config_graph")
def main(cfg):
    global log_dir_global, dram_step_reward, dram_step_legal, dram_step_hpwl, dram_step_loss, dram_step_ppo_loss, dram_step_mse_loss, dram_val_reward
    # 初始化全局变量
    dram_step_reward.clear()
    dram_step_legal.clear()
    dram_step_hpwl.clear()
    dram_step_loss.clear()
    dram_step_ppo_loss.clear()
    dram_step_mse_loss.clear()
    dram_val_reward.clear()

    # Preliminaries
    OmegaConf.set_struct(cfg, True)  # 冻结配置结构，不允许在程序运行时动态添加新的配置项
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
   
    # 如果未指定时间，则用当前时间
    current_time = cfg.time_stamp or datetime.now().strftime("%y%m%d%H%M")
    # 目录结构：{log_dir}/.{method}/{task}.{method}.{seed}.{time}
    method_dir = os.path.join(cfg.log_dir, f"{cfg.task}.{cfg.method}.{cfg.seed}")
    log_dir = os.path.join(method_dir, f"{current_time}")
    log_dir_global = log_dir # 用于绘图
    sample_dir = os.path.join(log_dir, "samples")
    checkpointer = common.Checkpointer(os.path.join(log_dir, "latest.ckpt"))
    os.makedirs(method_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    print(f"saving checkpoints to: {log_dir}")
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Preparing dataset
    train_set, val_set = utils.load_graph_data(cfg.task, augment = cfg.augment, train_data_limit = cfg.train_data_limit, val_data_limit = cfg.val_data_limit)
    sample_shape = train_set[0][0].shape
    
    dataloader = utils.GraphDataLoader(train_set, val_set, cfg.batch_size, cfg.val_batch_size, device)
    with open_dict(cfg):
        if cfg.family in ["cond_diffusion", "continuous_diffusion", "self_cond_diffusion", "skip_diffusion", "guided_diffusion", "skip_guided_diffusion"]:
            cfg.model.update({
                "num_classes": cfg.num_classes,
                "input_shape": tuple(sample_shape),
                "device": device,
            })
        elif cfg.family in ["mixed_diffusion"]:
            cfg.model.update({
                "device": device,
            })
        else:
            raise NotImplementedError

    # Preparing model, optimizer, and grad scaler (for AMP)
    model_types = {
        "cond_diffusion": models.CondDiffusionModel,
        "continuous_diffusion": models.ContinuousDiffusionModel, # Use this!
        "self_cond_diffusion": models.SelfCondDiffusionModel,
        "mixed_diffusion": models.ChipDiffusionModel,
        "skip_diffusion": models.SkipDiffusionModel,
        "guided_diffusion": models.GuidedDiffusionModel,
        "skip_guided_diffusion": models.SkipGuidedDiffusionModel,
    }
    print(f"Debug-- cfg.family:{cfg.family}")
    if cfg.implementation == "custom":
        model = model_types[cfg.family](**cfg.model).to(device)
    else:
        raise NotImplementedError
    # optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.ddpo.adam_beta1, cfg.ddpo.adam_beta2),
        weight_decay=cfg.ddpo.adam_weight_decay,
        eps=cfg.ddpo.adam_epsilon,
    )
    num_epochs =  cfg.num_epochs
    # if cfg.all_sample_num:
    #     num_epochs = int(cfg.all_sample_num / (cfg.num_batches_per_epoch* cfg.actual_batch))
    
    # 绑定余弦退火调度器
    # T_max：学习率完成一个余弦周期的epoch数；eta_min：最小学习率（避免降到0）
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer=optim,
    #     T_max=num_epochs // 2,  # 比如150 epoch，75个epoch完成半周期（常用）
    #     eta_min=cfg.lr * 0.01  # 最小LR=初始LR的1%（5e-5→5e-7）
    # )
    grad_scaler = torch.cuda.amp.GradScaler(enabled = (device == "cuda"))
    train_metrics = common.Metrics()
    print(f"########### cfg.mode:{cfg.mode}")
    ddpo_model = ddpo.DDPO(
        model, 
        # ddpo.get_reward_fn_ddpo(cfg.ddpo.legality_weight, cfg.ddpo.hpwl_weight, cfg.reward_version, cfg.scale_factor), 
        cfg.batch_size,
        cfg.ddpo.ema_factor,
        cfg.eta,
        )
    
    ################### 计算模型此时获得的布局的线长 baseline_hpwl_dict ###################
    # baseline_hpwl_dict = {}
    # baseline_hpwl_dict = utils.calcul_baseline_hpwl(dataloader, model)

    stat_tracker = PerPromptStatTracker(
        buffer_size=cfg.per_prompt_stat_tracking.stat_tracker_buffer_size,
        min_count=cfg.per_prompt_stat_tracking.stat_tracker_min_count
    )
    

    # Prepare logger
    num_params = sum([param.numel() for param in model.parameters()])
    with open_dict(cfg):  # for eval/debugging
        cfg.update({
            "num_params": num_params,
            "train_dataset": dataloader.get_train_size(),
            "val_dataset": dataloader.get_val_size(),
        })
    outputs = [
        common.logger.TerminalOutput(cfg.logger.filter),
    ]
    if cfg.logger.get("wandb", False):
        wandb_run_name = f"{cfg.task}.{cfg.method}.{cfg.seed}.{current_time}"
        outputs.append(common.logger.WandBOutput(wandb_run_name, cfg))
    step = common.Counter()
    logger = common.Logger(step, outputs)
    utils.save_cfg(cfg, os.path.join(log_dir, "config.yaml"))

    # Load checkpoint if exists
    # print(OmegaConf.to_yaml(cfg))
    print(f"model has {num_params} params")
    print(f"ddpo has {cfg.total_timesteps} total_timesteps")
    load_checkpoint(checkpointer, cfg, step, model, optim, grad_scaler)#, scheduler=scheduler)

    # Start training
    print(f"==== Start Training on Device: {device} ====")
    
    first_epoch = 0
    global_step = 0
    total_timesteps = cfg.total_timesteps
    batch_size = cfg.batch_size
    actual_batch = cfg.actual_batch
    lambda_mse = cfg.mse_loss_weight  # e.g. 0.1
    
    eta = cfg.eta  # 1.0 为 DDIM 没有额外的随机性
    best_reward = -1e2
    best_min = -1e2
    best_max = 1e12
    best_epoch = -1
    best_val = 0
    best_val_epoch = -1

    lambda_lagrange = 1.0    # initial multipler
    eta_lambda = 1e-2        # lr for lambda
    ent_coef = cfg.ent_coef 

    eval_time = 0

    global_reward_mean = None
    global_reward_std = None
    ema_alpha = cfg.ddpo.ema_factor # EMA 更新系数

    # 1. 外层 epoch 循环：先赋值给变量（方便调用set_postfix）
    epoch_pbar = tqdm(
        range(first_epoch, num_epochs), # num_epochs = 100
        desc=f"Epoch",
        position=0,
        leave=True,  # 保留外层进度条，避免刷屏
        dynamic_ncols=True  # 自适应终端宽度
    )

    for epoch in epoch_pbar:
        # use for draw
        legality_list = []
        hpwl_list = []
        # 2. 采样 sample
        samples = []

        # hpwl_w, legality_w, target_legal = get_dynamic_params(epoch, num_epochs)
        
        sampling_pbar = tqdm(
                range(cfg.num_batches_per_epoch),  # num_batches_per_epoch = 4
                desc=f"Epoch {epoch}: sampling",
                position=1,
                leave=False,  # 内层循环结束后自动清除，不干扰外层
                dynamic_ncols=True
            )
        model._noise_scheduler.set_timesteps(total_timesteps)
        for i in sampling_pbar:
            # 2.1 获取数据，时间步
            timesteps = model._noise_scheduler.timesteps
            x_list_batch = []
            cond_batch = []
            log_probs_batch =[]
            reward_batch = []
            legal_batch = []
            prompt_ids_batch = []
            intermediate_rewards_batch = []
            model.eval()
            with torch.no_grad():
                for _ in range(actual_batch):
                    # x, cond = dataloader.get_batch("train")
                    (x, cond), idx = dataloader.get_batch_and_idx("train")
                    # 2.2 记录模型预测 xt-x0, log_probs
                    x0, x_list, log_probs, x0_pre_list = ddpo_model.sample_with_logprob(x, cond, timesteps, eta)
                    # x_list最后一个就是x0, 第一个就是纯噪声  x_list, log_probs本身都是列表
                    
                    # 2.3 计算reward
                    # baseline_hpwl = baseline_hpwl_dict.get(idx, None)
                    # rewards, legality, hpwl = ddpo_model.reward_fn(x0, cond, x, batch_size, hpwl_w, legality_w, target_legal)  # 后续可以用异步计算
                    rewards, legality, hpwl, intermediate_rewards = ddpo_model.get_reward(x0, cond, x, x0_pre_list, cfg.intermediate, cfg.ddpo.hpwl_weight, cfg.ddpo.legality_weight)
                    rewards = rewards.to(device)
                    x_list = torch.stack(x_list, dim = 1)
                    log_probs = torch.stack(log_probs, dim = 1)

                    x_list_batch.append(x_list)
                    cond_batch.append(cond)
                    log_probs_batch.append(log_probs)

                    intermediate_rewards_batch.append(intermediate_rewards)

                    # 下面这些本身是batch个数的一维tensor
                    reward_batch.append(rewards)
                    prompt_ids_batch.append(idx)
                    legal_batch.append(legality)
                    # draw 
                    legality_list.append(legality.mean())
                    hpwl_list.append(hpwl.mean())
                
            # 更新采样进度条后缀（实时显示）
            curr_legal_mean = sum(legality_list) / len(legality_list) if legality_list else 0.0
            curr_hpwl_mean = sum(hpwl_list) / len(hpwl_list) if hpwl_list else 0.0
            sampling_pbar.set_postfix({
                "legal_mean": f"{curr_legal_mean:.4f}",  # 保留4位小数，清晰易读
                "hpwl_mean": f"{curr_hpwl_mean:.4f}"
            })
            
            timesteps_after = timesteps.repeat(batch_size, 1)
            timesteps_batch = []
            x_list_new = []
            next_x_list_new = []
            intermediate_rewards_list = []
            for j in range(actual_batch):
                x_list_new.append(x_list_batch[j][:,:-1])
                next_x_list_new.append(x_list_batch[j][:,1:])
                # 这里用 clone() 复制 tensor（避免多个元素指向同一个 tensor 对象，可选）
                new_timesteps = timesteps_after.clone()  
                timesteps_batch.append(new_timesteps)
                intermediate_rewards_list.append(intermediate_rewards_batch[j])

            samples.append(
                {
                    "timesteps": timesteps_batch,
                    "x_list": x_list_new,    # 时间步t前的x
                    "next_x_list": next_x_list_new,  # 时间步t后的x
                    "log_probs": log_probs_batch,
                    "rewards": reward_batch,
                    "cond": cond_batch,
                    "legalitys": legal_batch,  # 用作loss惩罚项
                    "prompt_ids": prompt_ids_batch ,
                    "intermediate_rewards": intermediate_rewards_list  # T-0 每一步到最终路径的回报
                }
            )
        # 关闭采样进度条（避免残留）
        sampling_pbar.close()
            
        # 计算reward
        # reward_combined = torch.cat([s["rewards"] for s in samples], dim=0)
        all_rewards = []
        all_prompt_ids = []
        for s in samples:
            for reward_batch in s["rewards"]:
                all_rewards.append(reward_batch)
            for prompt_ids_batch in s["prompt_ids"]:    
                all_prompt_ids.append(prompt_ids_batch)
        rewards = torch.cat(all_rewards, dim=0)
        prompt_ids = torch.cat(all_prompt_ids, dim=0)
        rewards_mean = rewards.mean()
        rewards_std = rewards.std()

        

        
        ################## 4. 计算优势函数  ##################
        ###################### 方案1   
        # EMA更新
        # if global_reward_mean is None:
        #     global_reward_mean = rewards_mean
        #     global_reward_std = rewards_std
        # else:
        #     global_reward_mean = ema_alpha * global_reward_mean + (1 - ema_alpha) * rewards_mean
        #     global_reward_std = ema_alpha * global_reward_std + (1 - ema_alpha) * rewards_std
        # for s in samples:
        #     s["advantages"] = []  # 新增 advantages 键
        #     for i in range(len(s["rewards"])):
        #         # 直接修改原 rewards 的值（标准化）
        #         advantage = (s["rewards"][i] - global_reward_mean) / (global_reward_std + 1e-8)
        #         # 将修改后的 rewards 复制一份到 advantages（结构完全对齐）
        #         # 用 .clone() 避免张量共享内存（后续修改一个不会影响另一个，可选但推荐）
        #         s["advantages"].append(advantage.clone())
        #         # all_advantages.append(advantage)
        #     del s["rewards"]

        
        if cfg.intermediate == False:
            ###################### 方案2 
            if cfg.per_prompt_stat_tracking.stat_tracking:
                advantages_np = stat_tracker.update(prompt_ids, rewards.cpu().numpy())
            else:
                advantages_np = (rewards.cpu().numpy() - rewards_mean.cpu().numpy()) / rewards_std.cpu().numpy()
            advantages_tensor = torch.as_tensor(advantages_np).to(rewards.device)
            start_idx = 0
            for s in samples:
                batch_size_s = len(s["rewards"]) # Get the batch size from this 's'
                temp = advantages_tensor[start_idx : start_idx + batch_size_s]
                s["advantages"] = [
                    temp[i].unsqueeze(0) # 将每个标量优势值转换为一个 (1,) 的一维张量
                    for i in range(temp.shape[0])
                ]
                start_idx += batch_size_s
                
                # 清理不再需要的键
                del s["rewards"] 
                del s["prompt_ids"]
                del s["intermediate_rewards"]
        else:
            ###################### 方案3 求及时回报平均
            # ------------------- 第一步：提取并扁平化所有intermediate_rewards -------------------
            all_intermediate_rewards = []  # 存储所有样本的50步奖励，最终形状(total_batch, 50)
            total_batch = 0
            
            # 遍历每个batch采样结果
            for sample_dict in samples:
                # 提取当前batch的intermediate_rewards_list（len=actual_batch）
                intermediate_rewards_list = sample_dict["intermediate_rewards"]
                for reward_tensor in intermediate_rewards_list:
                    # 确保奖励是50步的CUDA张量，形状(50,) → 扩展为(1,50)后加入列表
                    if len(reward_tensor.shape) == 1:
                        reward_tensor = reward_tensor.unsqueeze(0)  # (1,50)
                    all_intermediate_rewards.append(reward_tensor)
                    total_batch += 1
            
            # 拼接为(total_batch, 50)的CUDA张量
            reward_t = torch.cat(all_intermediate_rewards, dim=0).to(device)  # (total_batch, 50)
            # ------------------- 第二步：计算累积回报G_t（倒序） -------------------
            batch_size_total, num_timesteps = reward_t.shape  # num_timesteps=50
            G_t = torch.zeros_like(reward_t, device=device)
            G_t[:, -1] = reward_t[:, -1]  # 最后一步G_t = 最后一步奖励
            
            # 倒序计算累积回报（t从48到0）
            gamma = 1 # 50步短时序无折扣，固定1.0
            for t in range(num_timesteps-2, -1, -1):
                G_t[:, t] = reward_t[:, t] + gamma * G_t[:, t+1]

            # ------------------- 第三步：简化版GAE计算优势函数 -------------------
            # A_t = G_t - 同timestep所有样本的平均G_t（无价值网络，用批次平均替代）
            avg_G_t = torch.mean(G_t, dim=0)  # (50,)，每个timestep的批次平均累积回报
            A_t = G_t - avg_G_t.unsqueeze(0)  # (total_batch, 50)，原始优势值

            # ------------------- 第四步：优势函数归一化（稳定梯度） -------------------
            A_mean = torch.mean(A_t)
            A_std = torch.std(A_t) + 1e-8  # 避免除零
            A_t_norm = (A_t - A_mean) / A_std  # (total_batch, 50)，归一化后的优势值

            # ------------------- 第五步：将优势函数回填到samples结构中 -------------------
            idx = 0  # 遍历A_t_norm的索引
            for sample_dict in samples:
                actual_batch = len(sample_dict["intermediate_rewards"])
                # 提取当前batch对应的优势函数 (actual_batch, 50)
                batch_advantages = A_t_norm[idx:idx+actual_batch]
                idx += actual_batch
                
                # 将batch_advantages拆分为actual_batch个(50,)的张量，匹配intermediate_rewards_list结构
                advantages_list = [batch_advantages[i] for i in range(actual_batch)]
                sample_dict["advantages"] = advantages_list

                del sample_dict["rewards"]
                del sample_dict["prompt_ids"]
                del sample_dict["intermediate_rewards"]



        train_metrics.add(
            {
                "epoch": epoch,
                "reward_mean": rewards_mean,
            }
        )
        if best_reward < rewards_mean:
            best_reward = rewards_mean
            best_epoch = epoch
            checkpointer.save(os.path.join(log_dir, f"train_best.ckpt"))
        # draw
        dram_step_reward.append((epoch, rewards_mean))
        legality_mean = sum(legality_list) / len(legality_list)
        dram_step_legal.append((epoch, legality_mean))
        hpwl_mean = sum(hpwl_list) / len(hpwl_list)
        dram_step_hpwl.append((epoch, hpwl_mean))

        epoch_pbar.set_postfix({
            "reward": f"{rewards_mean:.3f}",    # 当前epoch奖励均值
            "legal": f"{legality_mean:.3f}",    # 当前epoch合法性均值
            "hpwl": f"{hpwl_mean:.3f}",         # 当前epoch hpwl均值
            "best": f"{best_reward:.3f} {best_epoch} val:{best_min:.3f} {best_val_epoch}", # 最优奖励（可选，方便监控）
            # "best": f"val:{best_min:.3f} {best_val_epoch}"
        })
        tqdm.write(
            f"[Epoch {epoch:04d}] "
            f"reward={rewards_mean:.3f} | "
            f"legal={legality_mean:.3f} | "
            f"hpwl={hpwl_mean:.3f} | "
            f"best={best_reward:.3f} (epoch {best_epoch}) val:{best_min:.3f} (epoch {best_val_epoch})"
        )

        # 3. 训练
        accum_step = 0  # 累积步数计数器（初始为0）
        accumulated_loss_value = 0.0  # 仅累积loss的纯数值（无梯度，不占显存）
        for inner_epoch in range(cfg.num_inner_epochs):  # num_inner_epochs = 1
            # 3.1 重排 samples 消除序列相关性
            perm = torch.randperm(cfg.num_batches_per_epoch, device="cpu")
            
            # 3.2 训练 samples_batched
            model.train()
            # 这里只会运行一次
            training_pbar = tqdm(
                range(len(samples)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=1,  # 采样循环已关闭，复用position=1，避免重叠
                leave=False,  # 训练结束后清除，不刷屏
                dynamic_ncols=True  # 自适应终端宽度
            )

            for i in training_pbar:
                index = perm[i]
                sample = samples[index]

                sample_loss_list = []  # 用于tqdm更新
                
                # 4. 训练 每个timestep
                time_perm = torch.randperm(total_timesteps, device="cpu")
                legal_diff_list = []
                for j in range(total_timesteps):
                    time_idx = time_perm[j]
                    # 5. 获取eps log_prob
                    log_probs = []
                    for k in range(actual_batch):
                        x_current = sample["x_list"][k][:, time_idx]
                        x_next = sample["next_x_list"][k][:, time_idx]
                        t_current = sample["timesteps"][k][:, time_idx].to(device)
                        cond_t = sample["cond"][k]
                        eps_pred = model(
                            x_current,
                            cond_t,
                            t_current
                        )

                        batch_shape = (batch_size, sample["cond"][k].x.shape[0], model.input_shape[1])
                        # 额外的噪声
                        variance_noise = model._epsilon_dist.sample(batch_shape).squeeze(dim = -1) if j < (len(timesteps)-2) else torch.zeros_like(sample["x_list"][k][:, time_idx])
                        _, log_prob, _ = ddpo_model.ddim_step_with_logprob(
                            eps_pred,
                            x_current,
                            t_current,
                            sample["timesteps"][k][:,time_idx + 1].to(device),
                            variance_noise,
                            eta = eta,
                            prev_sample = x_next,
                        )
                        log_probs.append(log_prob)

                    # 5.1 ppo logic
                    advantages = torch.clamp(
                        # sample["advantages"],
                        # torch.cat(sample["intermediate_rewards"][time_idx], dim=0),
                        torch.stack([tensor[time_idx] for tensor in sample["advantages"]]) if cfg.intermediate else torch.cat(sample["advantages"], dim=0),
                        -cfg.adv_clip_max,
                        cfg.adv_clip_max,
                    )
                    log_probs = torch.cat(log_probs, dim=0)
                    ratio = torch.exp(log_probs - torch.cat(sample["log_probs"], dim=0)[:, time_idx])
                    unclipped_loss = -advantages * ratio
                    clipped_loss = -advantages * torch.clamp(
                        ratio,
                        1.0 - cfg.clip_range,
                        1.0 + cfg.clip_range,
                    )
                    ppo_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                    # 4. Compute constraint penalty
                    final_legal = torch.cat(sample["legalitys"], dim=0).mean()
                    if cfg.legal_gate and final_legal < cfg.ddpo.legal_target:
                        loss = ppo_loss * 0.0
                    else:
                        loss = ppo_loss

                    # legal_diff = cfg.ddpo.legal_target - torch.cat(sample["legalitys"], dim=0)
                    # deficit = torch.clamp(legal_diff, min=0).squeeze()
                    # legal_diff_list.append(deficit)
                    # penalty_loss = lambda_lagrange * deficit
                    # entropy = - torch.mean(log_probs)
                    # loss = torch.mean(ppo_loss + penalty_loss - ent_coef * entropy)

                    # 5.2 反向传播和step
                    loss = loss / cfg.gradient_accumulation_steps  #平均
                    
                    grad_scaler.scale(loss).backward()
                    accum_step += 1 # 累积步数+1
                    accumulated_loss_value += loss.detach().cpu().item()  # 取数值累加到变量

                    # ========== 计算实时均值，更新tqdm后缀 ==========
                    loss_item = loss.detach().cpu().item()
                    sample_loss_list.append(loss_item)

                    if accum_step % cfg.gradient_accumulation_steps == 0:
                        grad_scaler.unscale_(optim)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = cfg.max_grad_norm)
                        grad_scaler.step(optim)
                        grad_scaler.update()
                        optim.zero_grad()
                        dram_step_loss.append((global_step, accumulated_loss_value))
                        # dram_step_ppo_loss.append((global_step, ppo_loss))
                        # dram_step_mse_loss.append((global_step, mse_loss_step))
                        # 重置累积变量
                        global_step += 1
                        accum_step = 0
                        accumulated_loss_value = 0.0
                        # Update lambda
                        # lambda_lagrange = max(
                        #     0,
                        #     lambda_lagrange + eta_lambda * torch.stack(legal_diff_list).mean().item()
                        # )
                        # lambda_lagrange = min(lambda_lagrange, 1.0)
                        # legal_diff_list = []

                        


                # 手动处理剩余梯度
                if accum_step >= cfg.gradient_accumulation_steps // 2:
                    grad_scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
                    grad_scaler.step(optim)
                    grad_scaler.update()
                    optim.zero_grad()
                    dram_step_loss.append((global_step, float(loss.detach().cpu().item())))
                    # dram_step_ppo_loss.append((global_step, ppo_loss))
                    # dram_step_mse_loss.append((global_step, mse_loss_step))
                    global_step += 1
                    # lambda_lagrange = max(
                    #     0,
                    #     lambda_lagrange + eta_lambda * torch.stack(legal_diff_list).mean().item()
                    # )
                    # lambda_lagrange = min(lambda_lagrange, 1.0)

                # 重置累积变量
                accumulated_loss_value = 0.0
                accum_step = 0

                # 更新tqdm
                curr_sample_loss_mean = sum(sample_loss_list) / len(sample_loss_list)
                training_pbar.set_postfix({
                    "avg_loss": f"{curr_sample_loss_mean:.6f}",  # 该sample的累计均值
                })

        # 评估hpwl
        if (cfg.eval_every > 0) and epoch > 0 and epoch % cfg.eval_every == 0:
            eval_time += 1
            num = 50
            if  eval_time % 5 == 0:
                num = 400 # 每过五次做一次全数据测评
            val_logs = utils.validate_ddpo(dataloader, model, ddpo_model, cfg.ddpo.hpwl_weight, cfg.ddpo.legality_weight, val_size=num) # num是进行验证的数据集的数量
            # val = val_logs["hpwl_mean"]
            hpwl_mean_valid = val_logs["reward"]    

            dram_val_reward.append((epoch, hpwl_mean_valid))

            if hpwl_mean_valid > best_min:   # hpwl_mean_valid 越大越好
                best_min = hpwl_mean_valid
                best_val_epoch = epoch
                checkpointer.save(os.path.join(log_dir, f"best.ckpt"))
        

        # --------------------- 调度器更新（epoch结束后调用） ---------------------
        # scheduler.step()  # 每轮epoch更新一次LR

        if epoch != 0 and epoch % cfg.save_freq == 0:
            checkpointer.save(os.path.join(log_dir, f"epoch_{epoch}.ckpt"))
    # 最后一个
    checkpointer.save() # save latest checkpoint

    plot_hpwl_curve(dram_step_reward, "reward",log_dir)
    plot_hpwl_curve(dram_step_legal, "legal_ratio",log_dir)
    plot_hpwl_curve(dram_step_hpwl, "hpwl_ratio",log_dir)
    plot_hpwl_curve(dram_step_loss, "loss",log_dir)
    plot_hpwl_curve(dram_val_reward, "val_reward",log_dir)
    # plot_hpwl_curve(dram_step_ppo_loss, "ppo_loss",log_dir)
    # plot_hpwl_curve(dram_step_mse_loss, "mse_loss",log_dir)

def load_checkpoint(checkpointer, cfg, step, model, optim, grad_scaler, scheduler=None):
    register_dict = {
        "step": step,
        "model": model,
        "optim": optim,
        "grad_scaler": grad_scaler,
    }
    if scheduler is not None:  # 仅 DDPO 模式传入 scheduler，其他模式不传
        register_dict["scheduler"] = scheduler
    checkpointer.register(register_dict)
    if cfg.mode == "train":
        checkpointer.load(
            path_override = None if (cfg.from_checkpoint == "none" or cfg.from_checkpoint is None) 
            else os.path.join(cfg.log_dir, cfg.from_checkpoint)
        )
    elif cfg.mode in ["finetune", "ddpo"]:
        if cfg.from_checkpoint :
            if "epoch_" in cfg.from_checkpoint:  # 判断是断点续训
                loaded = checkpointer.load(
                    path_override = os.path.join(cfg.log_dir, cfg.from_checkpoint),
                    # 去掉 filter_keys，加载所有注册的对象（包括 scheduler）
                )
            else:
                loaded = checkpointer.load(
                    path_override = os.path.join(cfg.log_dir, cfg.from_checkpoint),
                    filter_keys = ["model"],
                )
        # Try to resume existing run
        else:
            loaded = checkpointer.load()  # 读取当前路径的latest.ckpt !!!!!!!!!!!!!!
        # 微调的时候将step重新设置为0
        # step.value = 0
        if not loaded:
            # No existing run, so load pre-trained model only
            loaded = checkpointer.load(
                path_override = os.path.join(cfg.log_dir, cfg.from_checkpoint),
                filter_keys = ["model"],
            )
            if not loaded:
                print("WARNING Failed to load checkpoint for finetuning. Training from scratch instead.")
    else:
        raise NotImplementedError

def plot_hpwl_curve(dram_data, ylabel, log_dir):
    """
    dram_data: list of (step, value) tuples
    """
    f"{log_dir}/reward_curve.png"
    save_path = f"{log_dir}/_{ylabel}.png"
    save_txt = f"{log_dir}/_{ylabel}.csv"

    # 保存为 txt
    with open(save_txt, "w") as f:
        for s, r in dram_data:
            f.write(f"{s}, {r}\n")

    if len(dram_data) == 0:
        print("dram_data is empty!")
        return

    # 解包 step 和 hpwl


    steps = []
    hpwls = []

    for idx, item in enumerate(dram_data):
        step_raw, value_raw = item
        try:
            if isinstance(value_raw, torch.Tensor):
                # CUDA tensor → CPU tensor → 标量
                value = value_raw.cpu().item()
            elif isinstance(value_raw, (np.ndarray, np.generic)):
                # numpy 数组 → 标量
                value = value_raw.item()
            elif isinstance(value_raw, (int, float)):
                # 已是标量，直接使用
                value = value_raw
            else:
                # 其他类型尝试转换为 float
                value = float(value_raw)
        except Exception as e:
            print(f"Warning: 无法处理第 {idx} 个元素的 value（值：{value_raw}，类型：{type(value_raw)}），错误：{e}，跳过该元素")
            continue
        steps.append(step_raw)
        hpwls.append(value)

    title=f"{ylabel} vs Training Steps"
    # 绘制折线图
    plt.figure(figsize=(8, 5))
    plt.plot(steps, hpwls, linewidth=2, color='blue', marker='o', markersize=4, label=ylabel)
    plt.xlabel("Training Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存图片
    plt.savefig(save_path, dpi=300)
    print(f"{ylabel} curve saved to {save_path}")

def signal_handler(sig, frame):
    """处理中断信号"""
    print("\n" + "="*50)
    print("检测到中断信号, 正在保存训练进度...")
    print("="*50)
    
    # 保存当前模型
    
    try:
        print("正在绘制训练进度图...")
        plot_hpwl_curve(dram_step_reward, "reward", log_dir_global)
        plot_hpwl_curve(dram_step_legal, "legal_ratio",log_dir_global)
        plot_hpwl_curve(dram_step_hpwl, "hpwl_ratio",log_dir_global)
        plot_hpwl_curve(dram_step_loss, "loss",log_dir_global)
        plot_hpwl_curve(dram_val_reward, "val_reward",log_dir_global)
        # plot_hpwl_curve(dram_step_ppo_loss, "ppo_loss",log_dir_global)
        # plot_hpwl_curve(dram_step_mse_loss, "mse_loss",log_dir_global)
        print("✓ 训练进度图已保存")
    except Exception as e:
        print(f"✗ 绘图时发生错误: {e}")
    print("程序安全退出")
    sys.exit(6)

def get_dynamic_params_new(current_epoch, total_epochs=200):  # epoch增加到200
    explore_end = int(total_epochs * 1/3)   # 0~66epoch
    transition_end = int(total_epochs * 2/3)# 67~132epoch
    
    # 探索期：合法权重显著高于HPWL，强制重视合法率
    hpwl_explore = 1.2    # 进一步降低HPWL权重
    legality_explore = 2.2# 大幅提高合法权重（远超HPWL）
    target_explore = 0.95 # 固定目标为0.95（区间中点）
    
    # 稳定期：合法权重仍高于HPWL，保持约束
    hpwl_stable = 1.1     
    legality_stable = 2.3
    target_stable = 0.95  

    if current_epoch <= explore_end:
        return hpwl_explore, legality_explore, target_explore
    elif current_epoch <= transition_end:
        ratio = (current_epoch - explore_end) / (transition_end - explore_end)
        hpwl_weight = hpwl_explore - ratio * (hpwl_explore - hpwl_stable)
        legality_weight = legality_explore + ratio * (legality_stable - legality_explore)
        target_legality = target_explore  # 目标固定，不插值
        return hpwl_weight, legality_weight, target_legality
    else:
        return hpwl_stable, legality_stable, target_stable

def get_dynamic_params(current_epoch, total_epochs=150):
    """
    动态生成权重和目标合法性，适配"前期探线长+后期保合法"需求
    :param current_epoch: 当前训练epoch（从0开始）
    :param total_epochs: 总训练epoch（默认150，可按需调整）
    :return: hpwl_weight, legality_weight, target_legality
    """
    # 阶段分界（可按需微调，比如把过渡阶段拉长/缩短）
    explore_end = int(total_epochs * 1/3)   # 探索期：0~50epoch（总150）
    transition_end = int(total_epochs * 4/5)# 过渡期：51~120epoch
    # stable期：101~150epoch
    
    # 基础参数配置
    # 探索期：高HPWL权重，低合法权重，宽松目标
    hpwl_explore =  1.5  # 2.0
    legality_explore = 1.2 # 1.0
    target_explore = 0.90
    
    # 稳定期：低HPWL权重，高合法权重，严格目标
    hpwl_stable = 1.2 # 1.0
    legality_stable = 2.0 # 3.0
    target_stable = 0.93
    
    # 动态计算逻辑
    if current_epoch <= explore_end:
        # 探索期：固定参数
        return hpwl_explore, legality_explore, target_explore
    elif current_epoch <= transition_end:
        # 过渡期：线性插值（平滑过渡，避免突变）
        ratio = (current_epoch - explore_end) / (transition_end - explore_end)
        hpwl_weight = hpwl_explore - ratio * (hpwl_explore - hpwl_stable)
        legality_weight = legality_explore + ratio * (legality_stable - legality_explore)
        target_legality = target_explore + ratio * (target_stable - target_explore)
        return hpwl_weight, legality_weight, target_legality
    else:
        # 稳定期：固定参数
        return hpwl_stable, legality_stable, target_stable


if __name__=="__main__":
    # 在主程序开始前注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill命令
    main()
