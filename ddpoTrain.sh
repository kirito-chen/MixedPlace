# CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/train_graph.py method=ddpo task=v1.61 mode@_global_=ddpo from_checkpoint=v1.61.train_small.61_oral/best.ckpt

# v1.61 训练DDPO
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/train_graph.py method=ddpo task=v2.61 \
    mode@_global_=ddpo from_checkpoint=baseline/large-v2.ckpt

# v2.61 训练DDPO
# CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/train_graph.py method=ddpo task=v2.61 \
#     mode@_global_=ddpo from_checkpoint=baseline/large-v2.ckpt