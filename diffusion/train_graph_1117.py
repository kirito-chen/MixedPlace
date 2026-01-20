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
    if cfg.mode == "ddpo":
        ddpo_model = ddpo.DDPO(
            model, 
            ddpo.get_reward_fn(cfg.ddpo.legality_weight, cfg.ddpo.hpwl_weight), 
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
    print(OmegaConf.to_yaml(cfg))
    print(f"model has {num_params} params")
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
    best_loss = 1e12
    DEBUG = False
    with tqdm(total=cfg.train_steps, desc="Training Progress", dynamic_ncols=True) as pbar:
        while step < cfg.train_steps:
            optim.zero_grad() # 每次有效更新前清零

            x, cond = dataloader.get_batch("train")
            # x has (B, N, 2); netlist_data is a single graph in tg.Data format

            if cfg.mode != "ddpo":
                t = torch.randint(1, cfg.model.max_diffusion_steps + 1, [x.shape[0]], device = device)  # 1 - 1000
                loss, model_metrics = model.loss(x, cond, t)
            else:
                loss, model_metrics = ddpo_model.loss(x, cond, sample_dir, step, cfg.print_every, DEBUG)
                
            '''
            # 临时：不使用grad_scaler，直接backward
            print("=== 测试：直接反向传播（不使用grad_scaler）===")
            loss.backward()
            # 检查梯度
            ddpo_model._check_detailed_gradients(int(step))
             # 梯度裁剪
            max_grad_norm = 1.0
            torch.nn.utils.clip_grad_norm_(ddpo_model.model.parameters(), max_grad_norm)

            # 优化器步骤
            optim.step()

            '''
            # 反向传播前检查参数变化（与上一步比较）
            if DEBUG:
                ddpo_model._check_parameter_changes(int(step))

            # # 反向传播与梯度处理完整流程
            # grad_scaler.scale(loss).backward()

            # # 1. 先恢复原始梯度（关键步骤）
            # grad_scaler.unscale_(optim)

            # # 2. 检查是否有梯度溢出
            # has_overflow = False
            # for param in ddpo_model.model.parameters():
            #     if param.grad is not None:
            #         if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
            #             has_overflow = True
            #             break
            # print(f"Step {int(step)}: 梯度溢出 = {has_overflow}")

            # # 3. 仅在无溢出时执行裁剪和更新
            # if not has_overflow:
            #     max_grad_norm = 1.0
            #     torch.nn.utils.clip_grad_norm_(ddpo_model.model.parameters(), max_grad_norm)
            #     # 计算裁剪后梯度范数
            #     total_grad_norm = torch.nn.utils.clip_grad_norm_(ddpo_model.model.parameters(), max_grad_norm)
            #     print(f"裁剪后总梯度范数: {total_grad_norm:.2f}")
            #     # 执行参数更新
            #     grad_scaler.step(optim)
            # else:
            #     print(f"❌ Step {int(step)}: 梯度溢出，跳过更新")
            #     # 手动跳过更新，避免错误传播
            #     grad_scaler.update()
            #     continue

            # # 4. 更新梯度缩放系数
            # grad_scaler.update()


           
            
            grad_scaler.scale(loss).backward()
            # 反向传播后立即执行梯度裁剪
            # 关键：启用梯度裁剪，优先设置为1.0
            max_grad_norm = 1.0
            torch.nn.utils.clip_grad_norm_(ddpo_model.model.parameters(), max_grad_norm)
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

            # 7. 调试：检查梯度
            # for name, param in model.named_parameters():
            #     print(f"[Grad Check] {name} requires_grad = {param.requires_grad}")
            #     if param.requires_grad and param.grad is not None:
            #         if param.grad.norm() > 0:
            #             print(f"[Grad Check] {name}: grad norm = {param.grad.norm().item():.6f}")
            #         else:
            #             print(f"[Grad Check] {name} grad = 0")

            # 记录累积后的 metrics
            train_metrics.add({"loss": loss.detach().cpu().item()})
            train_metrics.add(model_metrics)
            step.increment() # 这是一个有效训练步骤

            if (int(step)) % cfg.print_every == 0:
                t_2 = time.time()
                x_val, cond_val = dataloader.get_batch("val")
                train_logs = utils.validate(x, model, cond)
                val_logs = utils.validate(x_val, model, cond_val)

                logger.add({
                    "time_elapsed": t_2-t_0, 
                    "ms_per_step": 1000*(t_2-t_1)/cfg.print_every
                    })
                logger.add(train_metrics.result())
                logger.add(val_logs, prefix="val")
                logger.add(train_logs, prefix="train")

                t_1 = t_2

                checkpointer.save() # save latest checkpoint
                if val_logs["loss"] < best_loss:
                    best_loss = val_logs["loss"]
                    checkpointer.save(os.path.join(log_dir, f"best.ckpt"))
                    print(f"saving best_{int(step)} model")
                cond_val.to(device="cpu")

            if (cfg.eval_every > 0) and (int(step)) % cfg.eval_every == 0:
                print(f"saving model at step {int(step)}")
                checkpointer.save(os.path.join(log_dir, f"step_{int(step)}.ckpt"))
                print("generating evaluation report")
                t3 = time.time()
                # 时间过长就不生成了
                # utils.generate_report(cfg.eval_samples, dataloader, model, logger, policy = cfg.eval_policy)
                logger.write()
                t4 = time.time()
                print(f"generated report in {t4-t3:.3f} sec")
            
            # 更新进度条
            pbar.set_postfix(loss=loss.item(), reward=model_metrics.get('reward', 0), legality= model_metrics.get('legality', 0), hpwl= model_metrics.get('hpwl', 0))
            pbar.update(1)  # 更新进度条

            cond.to(device="cpu")


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


if __name__=="__main__":
    main()
