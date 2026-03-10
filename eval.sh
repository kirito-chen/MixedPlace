# time_stamp="2602281113"
# task="clusterWithBase"
# eval_type="val_best" 


# ########## Evaluating zero-shot on clustered IBM benchmark with guidance:
# CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py method=eval_guided task=ibm.cluster512.v1 \
#     from_checkpoint=$task.ddpo.61/$time_stamp/$eval_type.ckpt num_output_samples=18 time_stamp=${time_stamp}


time_stamp="260310"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py method=eval_guided task=clusterWithBase \
    from_checkpoint=baseline/large-v2.ckpt num_output_samples=500 time_stamp=${time_stamp}

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py method=eval_guided task=clusterWithBase \
    from_checkpoint=best/2602281113_val_best.ckpt num_output_samples=500 time_stamp=${time_stamp}