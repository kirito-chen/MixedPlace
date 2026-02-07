time_stamp="2602052140"
task="clusterWithBase"
eval_type="train_best"


########## Evaluating zero-shot on clustered IBM benchmark with guidance:
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py method=eval_guided task=ibm.cluster512.v1 \
    from_checkpoint=$task.ddpo.61/$time_stamp/$eval_type.ckpt num_output_samples=18 time_stamp=${time_stamp}_train_best
