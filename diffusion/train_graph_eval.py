import utils
import torch
import hydra
import models
import ddpo
from omegaconf import OmegaConf, open_dict
import common
import os
import time
import csv



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
    cluster_baseline_hpwl_train = {}
    cluster_baseline_hpwl_train, cluster_baseline_hpwl_val = utils.calcul_baseline_hpwl(dataloader, model)

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
    print(f"==== Start Eval on Device: {device} ====")
    
    first_epoch = 0
    total_timesteps = cfg.total_timesteps
    actual_batch = cfg.actual_batch
    
    eta = cfg.eta  # 1.0 为 DDIM 没有额外的随机性


    csv_file = f"results_{os.path.splitext(os.path.basename(cfg.from_checkpoint))[0]}.csv"

    # 写表头
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "dp_HPWL", "time"])

    # 1. 外层 epoch 循环：先赋值给变量（方便调用set_postfix）
    num_samples = 500 # 测试500个数据集

    model._noise_scheduler.set_timesteps(total_timesteps)
    timesteps = model._noise_scheduler.timesteps
    model.eval()
    with torch.no_grad():
        for j in range(num_samples):  # 测试500个数据集
            start_time = time.time()   # 记录开始时间

            idx = torch.tensor([j])
            x, cond = dataloader.get_batch_with_idx("val", idx)
            # 2.2 记录模型预测 xt-x0, log_probs
            x0, x_list, log_probs, x0_pre_list = ddpo_model.sample_with_logprob(x, cond, timesteps, eta)
            # x_list最后一个就是x0, 第一个就是纯噪声  x_list, log_probs本身都是列表
            
            # 2.3 计算reward
            cluster_baseline_hpwl = cluster_baseline_hpwl_train.get(idx.item(), None)
            cluster_baseline_hpwl = torch.tensor(cluster_baseline_hpwl, device=device)
            dp_HPWL = ddpo_model.get_dp_hpwl(idx, x0, cond, x, x0_pre_list, cluster_baseline_hpwl, False, cfg.ddpo.hpwl_weight, cfg.ddpo.legality_weight)
            elapsed = time.time() - start_time
            print(f"idx:{j}, dp_HPWL:{dp_HPWL}")
            # tensor → float
            if torch.is_tensor(dp_HPWL):
                dp_HPWL = dp_HPWL.item()
            # 写入CSV
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([j, dp_HPWL, elapsed])
           



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
    # signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    # signal.signal(signal.SIGTERM, signal_handler)  # kill命令
    main()
