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

# 全局变量存储训练进度
dram_step_reward = []
dram_step_legal = [] 
dram_step_hpwl = []
dram_step_loss = []
log_dir_global = ""  # 需要在训练开始前设置


@hydra.main(version_base=None, config_path="configs", config_name="config_graph")
def main(cfg):
    global log_dir_global, dram_step_reward, dram_step_legal, dram_step_hpwl, dram_step_loss
    # 初始化全局变量
    dram_step_reward.clear()
    dram_step_legal.clear()
    dram_step_hpwl.clear()
    dram_step_loss.clear()

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
    try:
        os.makedirs(method_dir)
    except FileExistsError:
        pass
    try:
        os.makedirs(method_dir)
        os.makedirs(log_dir)
    except FileExistsError:
        pass
    try:
        os.makedirs(sample_dir)
    except FileExistsError:
        pass
    print(f"saving checkpoints to: {log_dir}")
    torch.manual_seed(cfg.seed)

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
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    grad_scaler = torch.cuda.amp.GradScaler(enabled = (device == "cuda"))
    train_metrics = common.Metrics()
    print(f"########### cfg.mode:{cfg.mode}")
    ddpo_model = ddpo.DDPO(
        model, 
        ddpo.get_reward_fn_ddpo(cfg.ddpo.legality_weight, cfg.ddpo.hpwl_weight, cfg.reward_version), 
        cfg.batch_size,
        cfg.ddpo.ema_factor,
        cfg.eta,
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
        wandb_run_name = f"{cfg.task}.{cfg.method}.{cfg.seed}"
        outputs.append(common.logger.WandBOutput(wandb_run_name, cfg))
    step = common.Counter()
    logger = common.Logger(step, outputs)
    utils.save_cfg(cfg, os.path.join(log_dir, "config.yaml"))

    # Load checkpoint if exists
    # print(OmegaConf.to_yaml(cfg))
    print(f"model has {num_params} params")
    print(f"ddpo has {cfg.total_timesteps} total_timesteps")
    load_checkpoint(checkpointer, cfg, step, model, optim, grad_scaler)

    # Start training
    print(f"==== Start Training on Device: {device} ====")
    
    first_epoch = 0
    global_step = 0
    total_timesteps = cfg.total_timesteps
    batch_size = cfg.batch_size
    eta = cfg.eta  # 1.0 为 DDIM 没有额外的随机性
    best_reward = -float('inf')
    # 1. 外层 epoch
    for epoch in tqdm(
        range(first_epoch, cfg.num_epochs), # num_epochs = 100
        desc=f"Epoch",
        position=0,
    ):
        # use for draw
        legality_list = []
        hpwl_list = []
        # 2. 采样 sample
        samples = []
        for i in tqdm(
            range(cfg.num_batches_per_epoch),  # num_batches_per_epoch = 4
            desc=f"Epoch {epoch}: sampling",
            position=1,
        ):
            # 2.1 获取数据，时间步
            x, cond = dataloader.get_batch("train")
            model._noise_scheduler.set_timesteps(total_timesteps)
            timesteps = model._noise_scheduler.timesteps

            # 2.2 记录模型预测 xt-x0, log_probs
            x0, x_list, log_probs = ddpo_model.sample_with_logprob(x, cond, timesteps, eta)
            x_list = torch.stack(x_list, dim = 1)
            log_probs = torch.stack(log_probs, dim = 1)
            
            # 2.3 计算reward 
            rewards, legality, hpwl = ddpo_model.reward_fn(x0, cond, x, batch_size)  # 后续可以用异步计算
            rewards = rewards.to(device)
            # rewards, legality, hpwl = ddpo_model.reward_fn(x0, cond)  # 后续可以用异步计算
            samples.append(
                {
                    "timesteps": timesteps.repeat(batch_size, 1),
                    "x_list": x_list[:,:-1],    # 时间步t前的x
                    "next_x_list": x_list[:,1:],  # 时间步t后的x
                    "log_probs": log_probs,
                    "rewards": rewards,
                    "cond": cond # .repeat(batch_size, 1)
                }
            )
            # draw 
            legality_list.append(legality.mean())
            hpwl_list.append(hpwl.mean())
            
        # 计算reward
        reward_combined = torch.cat([s["rewards"] for s in samples], dim=0)
        rewards = reward_combined
        rewards_mean = rewards.mean()
        train_metrics.add(
            {
                "epoch": epoch,
                "reward_mean": rewards_mean,
            }
        )
        if best_reward < rewards_mean:
            best_reward = rewards_mean
            # print(f"saving best model in epoch {epoch}")
            checkpointer.save(os.path.join(log_dir, f"best.ckpt"))
        # draw
        dram_step_reward.append((epoch, rewards_mean))
        legality_mean = sum(legality_list) / len(legality_list)
        dram_step_legal.append((epoch, legality_mean))
        hpwl_mean = sum(hpwl_list) / len(hpwl_list)
        dram_step_hpwl.append((epoch, hpwl_mean))

        # 计算优势函数
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        advantages_split = torch.split(advantages, split_size_or_sections=batch_size, dim=0)
        for s, adv in zip(samples, advantages_split):
            s["advantages"] = adv  # adv 形状与 s["reward"] 完全一致（如 [2, 30]）
            del s["rewards"]


        # 3. 训练
        accum_step = 0  # 累积步数计数器（初始为0）
        for inner_epoch in range(cfg.num_inner_epochs):  # num_inner_epochs = 1
            # 3.1 重排 samples 消除序列相关性
            perm = torch.randperm(cfg.num_batches_per_epoch, device="cpu")
            
            # 3.2 训练 samples_batched
            model.train()
            # 这里只会运行一次
            for i in tqdm(
                range(len(samples)),
                # list(enumerate(samples)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=1,
            ):
                index = perm[i]
                sample = samples[index]
                cond = sample["cond"].to(device)
                # timesteps = sample["timesteps"]
                
                # 4. 训练 每个timestep
                time_perm = torch.randperm(total_timesteps, device="cpu")
                for j in tqdm(
                    range(total_timesteps),
                    desc="Timestep",
                    position=2,
                    leave=False,
                ):
                    time_idx = time_perm[j]
                    # 5. 获取eps log_prob
                    eps_pred = model(
                        torch.cat([sample["x_list"][:, time_idx]]),
                        cond,
                        torch.cat([sample["timesteps"][:, time_idx]]).to(device)
                    )
                    batch_shape = (batch_size, cond.x.shape[0], model.input_shape[1])
                    variance_noise = model._epsilon_dist.sample(batch_shape).squeeze(dim = -1) if i<(len(timesteps)-2) else torch.zeros_like(sample["x_list"][:, time_idx])
                    _, log_prob = ddpo_model.ddim_step_with_logprob(
                        eps_pred,
                        sample["x_list"][:, time_idx],
                        sample["timesteps"][:,time_idx].to(device),
                        sample["timesteps"][:,time_idx + 1].to(device),
                        variance_noise,
                        eta = eta,
                        prev_sample = sample["next_x_list"][:, time_idx],
                    )
                    # 5.1 ppo logic
                    advantages = torch.clamp(
                        sample["advantages"],
                        -cfg.adv_clip_max,
                        cfg.adv_clip_max,
                    )
                    ratio = torch.exp(log_prob - sample["log_probs"][:, time_idx])
                    unclipped_loss = -advantages * ratio
                    clipped_loss = -advantages * torch.clamp(
                        ratio,
                        1.0 - cfg.clip_range,
                        1.0 + cfg.clip_range,
                    )
                    loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                    # 5.2 反向传播和step
                    grad_scaler.scale(loss).backward()
                    accum_step += 1 # 累积步数+1

                    if accum_step % cfg.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 
                            max_norm = cfg.max_grad_norm
                        )
                        grad_scaler.step(optim)
                        grad_scaler.update()
                        optim.zero_grad()
                        dram_step_loss.append((global_step, loss))
                        global_step += 1
                        accum_step = 0
                # 手动处理剩余梯度
                if accum_step != 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
                    grad_scaler.step(optim)
                    grad_scaler.update()
                    optim.zero_grad()
                    dram_step_loss.append((global_step, loss))
                    global_step += 1
                    accum_step = 0

        if epoch != 0 and epoch % cfg.save_freq == 0:
            checkpointer.save(os.path.join(log_dir, f"epoch_{epoch}.ckpt"))
    # 最后一个
    # checkpointer.save() # save latest checkpoint

    plot_hpwl_curve(dram_step_reward, "reward",log_dir)
    plot_hpwl_curve(dram_step_legal, "legal_ratio",log_dir)
    plot_hpwl_curve(dram_step_hpwl, "hpwl_ratio",log_dir)
    plot_hpwl_curve(dram_step_loss, "loss",log_dir)

def load_checkpoint(checkpointer, cfg, step, model, optim, grad_scaler):
    checkpointer.register({
            "step": step,
            "model": model,
            "optim": optim,
            "grad_scaler": grad_scaler,
        })
    if cfg.mode == "train":
        checkpointer.load(
            path_override = None if (cfg.from_checkpoint == "none" or cfg.from_checkpoint is None) 
            else os.path.join(cfg.log_dir, cfg.from_checkpoint)
        )
    elif cfg.mode in ["finetune", "ddpo"]:
        # Try to resume existing run
        if cfg.from_checkpoint :
            loaded = checkpointer.load(
                path_override = os.path.join(cfg.log_dir, cfg.from_checkpoint),
                filter_keys = ["model"],
            )
        # Try to resume existing run
        else:
            loaded = checkpointer.load()  # 读取当前路径的latest.ckpt !!!!!!!!!!!!!!
        # 微调的时候将step重新设置为0
        step.value = 0
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
    dram_data: list of (step, value, color) tuples
    """
    f"{log_dir}/reward_curve.png"
    save_path = f"{log_dir}/_{ylabel}.png"
    save_txt = f"{log_dir}/_{ylabel}.csv"

    # 保存为 csv
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
    # === 找到最大值并标红 ===
    max_idx = np.argmax(hpwls)
    plt.scatter(steps[max_idx], hpwls[max_idx], color='red', s=50, zorder=10)
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
        print("✓ 训练进度图已保存")
    except Exception as e:
        print(f"✗ 绘图时发生错误: {e}")
    print("程序安全退出")
    sys.exit(6)


if __name__=="__main__":
    # 在主程序开始前注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill命令
    main()