

# list_type=("with_guided_and_legal" "no_guided_with_legal" "no_guided_and_legal")
list_from_checkpoint=("paper_model/2512222317_best.ckpt") # "baseline/large-v2.ckpt"

origin_time="2512222317"


for from_checkpoint in "${list_from_checkpoint[@]}"; do
    echo "==============================="
    echo "Running checkpoint: $from_checkpoint"
    echo "==============================="

    # 提取斜杠前的部分，例如：baseline/large-v2.ckpt → baseline
    prefix="${from_checkpoint%%/*}"
    # prefix=$origin_time
    

    time_stamp="final_${origin_time}_${prefix}"
    echo "---- Running: prefix=$prefix time_stamp=$time_stamp ckpt=$from_checkpoint ----"
    ####################### v1.6和v2.6只在这时候做
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py task=v1.61 method=eval \
        from_checkpoint=$from_checkpoint legalizer@_global_=none guidance@_global_=none \
        num_output_samples=2000 time_stamp=$time_stamp

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py task=v2.61 method=eval \
        from_checkpoint=$from_checkpoint legalizer@_global_=none guidance@_global_=none \
        num_output_samples=400 time_stamp=$time_stamp
        

done

