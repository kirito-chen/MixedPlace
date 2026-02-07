# python mixedBookshelf.py 

# python mixedCluster_parallel.py  


time_stamp="2601271702"
task="clusterWithBase"

########## Evaluating zero-shot on clustered IBM benchmark with guidance:
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py method=eval_guided task=ibm.cluster512.v1 \
    from_checkpoint=$task.ddpo.61/$time_stamp/best.ckpt num_output_samples=18 time_stamp=$time_stamp