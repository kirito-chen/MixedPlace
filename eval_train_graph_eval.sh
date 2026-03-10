# 用于测试测试集的
time_stamp="260310"
task="clusterWithBase"
program="train_graph_eval.py"

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/$program \
    method=ddpo task=$task mode@_global_=ddpo \
    from_checkpoint=baseline/large-v2.ckpt time_stamp=${time_stamp}_baseline

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/$program \
    method=ddpo task=$task mode@_global_=ddpo \
    from_checkpoint=best/2602281113_val_best.ckpt time_stamp=${time_stamp}_best