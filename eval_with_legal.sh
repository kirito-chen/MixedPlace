
time_stamp="baseline_with_legal_no_guided"
from_checkpoint="baseline/large-v2.ckpt"

########## Evaluating zero-shot on clustered IBM benchmark with guidance:
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py method=eval_guided task=ibm.cluster512.v1 \
    from_checkpoint=$from_checkpoint guidance@_global_=none \
    num_output_samples=18 time_stamp=$time_stamp

########## Macro-only evaluation for IBM and ISPD benchmarks:
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py method=eval_macro_only \
    task=ibm.cluster512.v1 from_checkpoint=$from_checkpoint \
    guidance@_global_=none \
    legalizer@_global_=opt-adam num_output_samples=18 model.grad_descent_steps=20 \
    model.hpwl_guidance_weight=16e-4 legalization.alpha_lr=8e-3 legalization.hpwl_weight=12e-5 \
    legalization.legality_potential_target=0 legalization.grad_descent_steps=20000 macros_only=True \
    time_stamp=$time_stamp
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py method=eval_macro_only task=ispd2005 \
    from_checkpoint=$from_checkpoint legalizer@_global_=opt-adam guidance@_global_=none \
    num_output_samples=8 \
    model.grad_descent_steps=20 model.hpwl_guidance_weight=16e-4 legalization.alpha_lr=8e-3 \
    legalization.hpwl_weight=12e-5 legalization.legality_potential_target=0 \
    legalization.grad_descent_steps=20000 macros_only=True  time_stamp=$time_stamp

exit 1

time_stamp="2512062049_best_with_legal"
# from_checkpoint="v2.61.ddpo.61/2512042217/latest.ckpt"
# logs/diffusion_debug/v2.61.ddpo.61/2512062049-best
from_checkpoint="v2.61.ddpo.61/2512062049-best/best.ckpt"
# from_checkpoint="baseline/large-v2.ckpt"


# 带合法
########## Evaluating zero-shot on clustered IBM benchmark with guidance:
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py method=eval_guided task=ibm.cluster512.v1 \
    from_checkpoint=$from_checkpoint guidance@_global_=none \
    num_output_samples=18 time_stamp=$time_stamp

########## Macro-only evaluation for IBM and ISPD benchmarks:
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py method=eval_macro_only \
    task=ibm.cluster512.v1 from_checkpoint=$from_checkpoint \
    guidance@_global_=none \
    legalizer@_global_=opt-adam num_output_samples=18 model.grad_descent_steps=20 \
    model.hpwl_guidance_weight=16e-4 legalization.alpha_lr=8e-3 legalization.hpwl_weight=12e-5 \
    legalization.legality_potential_target=0 legalization.grad_descent_steps=20000 macros_only=True \
    time_stamp=$time_stamp
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py method=eval_macro_only task=ispd2005 \
    from_checkpoint=$from_checkpoint legalizer@_global_=opt-adam guidance@_global_=none \
    num_output_samples=8 \
    model.grad_descent_steps=20 model.hpwl_guidance_weight=16e-4 legalization.alpha_lr=8e-3 \
    legalization.hpwl_weight=12e-5 legalization.legality_potential_target=0 \
    legalization.grad_descent_steps=20000 macros_only=True  time_stamp=$time_stamp


