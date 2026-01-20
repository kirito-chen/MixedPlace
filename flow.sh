
# time_stamp=$(date +"%y%m%d%H%M")
time_stamp="2512221653"
task="v2.61"
program="train_graph.py"

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py method=eval_macro_only task=ispd2005 \
    from_checkpoint=$task.ddpo.61/$time_stamp/best.ckpt \
    legalizer@_global_=opt-adam guidance@_global_=opt num_output_samples=8 \
    model.grad_descent_steps=20 model.hpwl_guidance_weight=16e-4 legalization.alpha_lr=8e-3 \
    legalization.hpwl_weight=12e-5 legalization.legality_potential_target=0 \
    legalization.grad_descent_steps=20000 macros_only=True time_stamp=$time_stamp
    

time_stamp=$(date +"%y%m%d%H%M")
task="v2.61"
program="train_graph.py"

# === 第一步：训练 ===
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/$program \
    method=ddpo task=$task mode@_global_=ddpo \
    from_checkpoint=baseline/large-v2.ckpt time_stamp=$time_stamp

# 验证ddpo微调后的模型
if [ $? -eq 0 ]; then
    echo "✅ 训练完成，开始验证..."
    ### V1.61
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py task=v1.61 \
    method=eval from_checkpoint=$task.ddpo.61/$time_stamp/best.ckpt \
    legalizer@_global_=none guidance@_global_=none num_output_samples=40 \
    time_stamp=$time_stamp
    ### V2.61
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py task=v2.61 \
    method=eval from_checkpoint=$task.ddpo.61/$time_stamp/best.ckpt \
    legalizer@_global_=none guidance@_global_=none num_output_samples=40 \
    time_stamp=$time_stamp

    ########## Evaluating zero-shot on clustered IBM benchmark with guidance:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py method=eval_guided task=ibm.cluster512.v1 \
        from_checkpoint=$task.ddpo.61/$time_stamp/best.ckpt num_output_samples=18 time_stamp=$time_stamp

    ########## Macro-only evaluation for IBM and ISPD benchmarks:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py method=eval_macro_only \
        task=ibm.cluster512.v1 from_checkpoint=$task.ddpo.61/$time_stamp/best.ckpt \
        legalizer@_global_=opt-adam num_output_samples=18 model.grad_descent_steps=20 \
        model.hpwl_guidance_weight=16e-4 legalization.alpha_lr=8e-3 legalization.hpwl_weight=12e-5 \
        legalization.legality_potential_target=0 legalization.grad_descent_steps=20000 macros_only=True \
        time_stamp=$time_stamp
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py method=eval_macro_only task=ispd2005 \
        from_checkpoint=$task.ddpo.61/$time_stamp/best.ckpt \
        legalizer@_global_=opt-adam guidance@_global_=opt num_output_samples=8 \
        model.grad_descent_steps=20 model.hpwl_guidance_weight=16e-4 legalization.alpha_lr=8e-3 \
        legalization.hpwl_weight=12e-5 legalization.legality_potential_target=0 \
        legalization.grad_descent_steps=20000 macros_only=True time_stamp=$time_stamp
    

else
  echo "❌ 训练失败，跳过验证。"
fi

