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

@hydra.main(version_base=None, config_path="configs", config_name="config_graph")
def main(cfg):
    # Preliminaries
    OmegaConf.set_struct(cfg, True)  # 冻结配置结构，不允许在程序运行时动态添加新的配置项
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
   
    # 如果未指定时间，则用当前时间
    current_time = cfg.time_stamp or datetime.now().strftime("%y%m%d%H%M")
    # 目录结构：{log_dir}/.{method}/{task}.{method}.{seed}.{time}
    method_dir = os.path.join(cfg.log_dir, f"{cfg.task}.{cfg.method}.{cfg.seed}")
    log_dir = os.path.join(method_dir, f"{current_time}")
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
    ###########################################################
    # 在模型定义之后，训练循环开始之前 
    # torch.compile 旨在将这类Python控制流图(for 循环, if/elif/else 条件判断)转换为高效的、可优化的计算图，
    # 从而减少Python解释器的开销, 并可能生成更优化的CUDA内核
    # model = torch.compile(model)                           !!!!!!!!!!报错
    # model = torch.compile(model, mode="reduce-overhead")   !!!!!!!!!!报错
    ###########################################################
    if cfg.mode == "ddpo":
        ddpo_model = ddpo.DDPO(
            model, 
            ddpo.get_reward_fn_ddpo(cfg.ddpo.legality_weight, cfg.ddpo.hpwl_weight), 
            cfg.batch_size,
            cfg.ddpo.ema_factor,
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
    ddpo_model.initial_params = {
        name: param.data.clone() 
        for name, param in ddpo_model.model.named_parameters()
    }

    # Start training
    print(f"==== Start Training on Device: {device} ====")
    model.train()

    t_0 = time.time()
    t_1 = time.time()
    best_hpwl = 1e12
    best_reward = -float('inf')
    dram_step_avg_reward = []
    dram_step_reward = []
    dram_step_legal = []
    dram_step_hpwl = []
    dram_step_advantage = []
    dram_step_loss = []
    dram_step_grad = []

    # 初始化滑动平均参数
    ema_alpha = 0.01       # 越大新 reward 权重越高
    avg_reward = 0.0      # 初始 EMA
    best_avg_reward = -float('inf')  # 记录最佳滑动平均 reward

    DEBUG = False
    with tqdm(total=cfg.train_steps, desc="Training Progress", dynamic_ncols=True) as pbar:
        while step < cfg.train_steps:
            optim.zero_grad() # 每次有效更新前清零
            if DEBUG and int(step) == 3:
                exit(1)
            x, cond = dataloader.get_batch("train")
            # x has (B, N, 2); netlist_data is a single graph in tg.Data format

            if cfg.mode != "ddpo":
                t = torch.randint(1, cfg.model.max_diffusion_steps + 1, [x.shape[0]], device = device)  # 1 - 1000
                loss, model_metrics = model.loss(x, cond, t)
            else:
                loss, model_metrics = ddpo_model.loss(x, cond, sample_dir, step, cfg.eval_every, DEBUG = DEBUG, idx = 0, _total_timesteps = cfg.total_timesteps)
                
            # 反向传播前检查参数变化（与上一步比较）
            if DEBUG:
                ddpo_model._check_parameter_changes(int(step))

            grad_scaler.scale(loss).backward()
            if DEBUG:
                print("=== 原始梯度检查（裁剪/反缩放前）===")
                total_raw_grad = 0.0
                has_raw_grad = 0
                for param in ddpo_model.model.parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        total_raw_grad += grad_norm
                        if grad_norm > 1e-10:
                            has_raw_grad += 1
                print(f"原始总梯度范数: {total_raw_grad:.10f}")
                print(f"有原始梯度的参数: {has_raw_grad}/{len(list(ddpo_model.model.parameters()))}")
            # 反向传播后立即执行梯度裁剪
            # 关键：启用梯度裁剪，优先设置为1.0
            max_grad_norm = 1. # 适度放宽，保留更多有效梯度
            grad_scaler.unscale_(optim)  # 必须先反缩放，再裁剪（之前的核心遗漏！）
            torch.nn.utils.clip_grad_norm_(ddpo_model.model.parameters(), max_grad_norm)
            # 模型总梯度范数（裁剪后）	0.5~5.0	过大→降低 log_prob 放大系数；过小→增大  修改 max_grad_norm
            clipped_grad_norm = 0.0
            for param in ddpo_model.model.parameters():
                if param.grad is not None:
                    clipped_grad_norm += param.grad.norm().item() ** 2
            clipped_grad_norm = clipped_grad_norm ** 0.5
            dram_step_grad.append((int(step), clipped_grad_norm))

            # 检查裁剪后的梯度范数
            if DEBUG:
                total_grad_norm = 0.0
                for param in ddpo_model.model.parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.norm().item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                print(f"裁剪后总梯度范数: {total_grad_norm:.2f}")
            # 后续优化器步骤
            grad_scaler.step(optim)
            grad_scaler.update()

            # 反向传播后检查梯度
            if DEBUG:
                ddpo_model._check_detailed_gradients(int(step))
                ddpo_model._check_optimizer_state(optim, int(step))

            # 记录累积后的 metrics
            train_metrics.add({"loss": loss.detach().cpu().item()})
            train_metrics.add(model_metrics)
            step.increment() # 这是一个有效训练步骤
            step_int = int(step)
            '''
            # 验证过慢
            if (step_int) % cfg.print_every == 0:
                t_2 = time.time()
                # x_val, cond_val = dataloader.get_batch("val")
                # train_logs = utils.validate(x, model, cond)
                val_logs = utils.validate_ddpo(dataloader, model, ddpo_model, val_size=32)

                logger.add({
                    "time_elapsed": t_2-t_0, 
                    "ms_per_step": 1000*(t_2-t_1)/cfg.print_every
                    })
                logger.add(train_metrics.result())
                logger.add(val_logs, prefix="val")
                # logger.add(train_logs, prefix="train")

                t_1 = t_2
                checkpointer.save() # save latest checkpoint

                # 使用 reward 判断 best checkpoint
                current_reward = val_logs["reward"]  #  validate() 里返回的 reward
                print(f"step {step_int} in eval hpwl, current_reward={current_reward:.2f}")
                dram_step_reward.append((step_int, current_reward)) # 记录数据用于后续绘图

                if current_reward > best_reward:
                    best_reward = current_reward
                    checkpointer.save(os.path.join(log_dir, f"best.ckpt"))
                    print(f"saving best_{step_int} model, best_reward={best_reward:.2f}")
                # cond_val.to(device="cpu")
            '''
            # ============================
            # 更新滑动平均 reward
            # ============================
            current_reward = model_metrics.get('reward', 0)
            if step == 1:
                avg_reward = current_reward  # EMA 初始化
            else:
                avg_reward = ema_alpha * current_reward + (1 - ema_alpha) * avg_reward

            # 保存 best ckpt
            if current_reward > best_reward:
                best_reward = current_reward
                checkpointer.save(os.path.join(log_dir, f"best.ckpt"))
                print(f"[Step {step_int}] Saving best.ckpt checkpoint, best_reward={best_reward:.4f}")
                
            dram_step_avg_reward.append((step_int, avg_reward)) # 记录数据用于后续绘图
            dram_step_legal.append((step_int, model_metrics.get('legality', 0)))
            dram_step_hpwl.append((step_int, model_metrics.get('hpwl', 0)))
            dram_step_reward.append((step_int, current_reward))
            dram_step_advantage.append((step_int, model_metrics.get('advantage', 0)))
            dram_step_loss.append((step_int,loss.item()))
            with open(f"{log_dir}/avg_reward_curve.csv", "a") as f:      # 以追加模式写入
                f.write(f"{step_int}, {avg_reward}\n")

            # 保存 best model（基于滑动平均 reward）
            # if avg_reward > best_avg_reward:
            #     best_avg_reward = avg_reward
            #     checkpointer.save(os.path.join(log_dir, f"best.ckpt"))
            #     print(f"[Step {step_int}] Saving best.ckpt checkpoint, avg_reward={best_avg_reward:.4f}")

            if (cfg.eval_every > 0) and (step_int) % cfg.eval_every == 0:
                print(f"saving model at step {step_int}")
                checkpointer.save(os.path.join(log_dir, f"step_{step_int}.ckpt"))
                print("generating evaluation report")
                t3 = time.time()
                # 时间过长就不生成了
                # utils.generate_report(cfg.eval_samples, dataloader, model, logger, policy = cfg.eval_policy)
                logger.write()
                t4 = time.time()
                print(f"generated report in {t4-t3:.3f} sec")
            
            # 更新进度条
            pbar.set_postfix(loss=loss.item(), reward=current_reward)
            pbar.update(1)  # 更新进度条

            cond.to(device="cpu")
    
    # 训练结束绘制hpwl变化曲线图
    plot_hpwl_curve(dram_step_avg_reward, "avg_reward", log_dir)
    plot_hpwl_curve(dram_step_reward, "reward",log_dir)
    plot_hpwl_curve(dram_step_legal, "legal",log_dir)
    plot_hpwl_curve(dram_step_hpwl, "hpwl",log_dir)
    # 监控超参
    plot_hpwl_curve(dram_step_advantage, "advantage",log_dir)
    plot_hpwl_curve(dram_step_grad, "grad",log_dir)
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
        loaded = checkpointer.load()
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
    steps = [item[0] for item in dram_data]
    hpwls = [item[1] for item in dram_data]

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

if __name__=="__main__":
    main()
