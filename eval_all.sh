

list_type=("no_guided_with_legal" ) #"no_guided_and_legal")  # "with_guided_and_legal"
list_from_checkpoint=("paper_model/2512201545_best.ckpt")

# list_from_checkpoint=("v2.61.ddpo.61/2512101521-best/best.ckpt" "v2.61.ddpo.61/2512101521-best/val_best.ckpt")
# list_type=("no_guided_and_legal")
# list_from_checkpoint=("ddpo_best_1206/best.ckpt")
origin_time="2512201545"


for from_checkpoint in "${list_from_checkpoint[@]}"; do
    echo "==============================="
    echo "Running checkpoint: $from_checkpoint"
    echo "==============================="

    # 提取斜杠前的部分，例如：baseline/large-v2.ckpt → baseline
    # prefix="${from_checkpoint%%/*}"
    prefix=$origin_time
    
    for type in "${list_type[@]}"; do
        
        # baseline → 只允许 with_guided_and_legal
        # if [[ "$prefix" == "baseline" && "$type" != "with_guided_and_legal" ]]; then
        #     continue
        # fi
        # # ddpo_best_1206 → 只允许 no_guided_and_legal
        # if [[ "$prefix" == "ddpo_best_1206" && "$type" != "no_guided_and_legal" ]]; then
        #     continue
        # fi

        time_stamp="b_2_${prefix}_${type}"
        echo "---- Running: prefix=$prefix  type=$type time_stamp=$time_stamp ckpt=$from_checkpoint ----"
        case $type in
            with_guided_and_legal)
                ########## Evaluating zero-shot on clustered IBM benchmark with guidance:
                CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py method=eval_guided task=ibm.cluster512.v1 \
                    from_checkpoint=$from_checkpoint num_output_samples=18 time_stamp=$time_stamp

                ########## Macro-only evaluation for IBM and ISPD benchmarks:
                CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py method=eval_macro_only \
                    task=ibm.cluster512.v1 from_checkpoint=$from_checkpoint \
                    legalizer@_global_=opt-adam num_output_samples=18 model.grad_descent_steps=20 \
                    model.hpwl_guidance_weight=16e-4 legalization.alpha_lr=8e-3 legalization.hpwl_weight=12e-5 \
                    legalization.legality_potential_target=0 legalization.grad_descent_steps=20000 macros_only=True \
                    time_stamp=$time_stamp
                CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py method=eval_macro_only task=ispd2005 \
                    from_checkpoint=$from_checkpoint legalizer@_global_=opt-adam guidance@_global_=opt num_output_samples=8 \
                    model.grad_descent_steps=20 model.hpwl_guidance_weight=16e-4 legalization.alpha_lr=8e-3 \
                    legalization.hpwl_weight=12e-5 legalization.legality_potential_target=0 \
                    legalization.grad_descent_steps=20000 macros_only=True  time_stamp=$time_stamp

                    ;;
            no_guided_with_legal)
                ########## Evaluating zero-shot on clustered IBM benchmark with guidance:
                # CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py method=eval_guided task=ibm.cluster512.v1 \
                #     from_checkpoint=$from_checkpoint num_output_samples=18 guidance@_global_=none time_stamp=$time_stamp

                # ########## Macro-only evaluation for IBM and ISPD benchmarks:
                # CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py method=eval_macro_only \
                #     task=ibm.cluster512.v1 from_checkpoint=$from_checkpoint \
                #     legalizer@_global_=opt-adam guidance@_global_=none \
                #     num_output_samples=18 model.grad_descent_steps=20 \
                #     legalization.alpha_lr=8e-3 legalization.hpwl_weight=12e-5 \
                #     legalization.legality_potential_target=0 legalization.grad_descent_steps=20000 macros_only=True \
                #     time_stamp=$time_stamp
                CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py method=eval_macro_only task=ispd2005 \
                    from_checkpoint=$from_checkpoint legalizer@_global_=opt-adam guidance@_global_=none num_output_samples=8 \
                    model.grad_descent_steps=20 legalization.alpha_lr=8e-3 \
                    legalization.hpwl_weight=12e-5 legalization.legality_potential_target=0 \
                    legalization.grad_descent_steps=20000 macros_only=True  time_stamp=$time_stamp
                    ;;
            no_guided_and_legal)
                ####################### v1.6和v2.6只在这时候做
                # CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py task=v1.61 method=eval \
                #     from_checkpoint=$from_checkpoint legalizer@_global_=none guidance@_global_=none \
                #     num_output_samples=40 time_stamp=$time_stamp

                # CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py task=v2.61 method=eval \
                #     from_checkpoint=$from_checkpoint legalizer@_global_=none guidance@_global_=none \
                #     num_output_samples=40 time_stamp=$time_stamp

                ########## Evaluating zero-shot on clustered IBM benchmark with guidance:
                CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py method=eval_guided task=ibm.cluster512.v1 \
                    from_checkpoint=$from_checkpoint legalizer@_global_=none guidance@_global_=none \
                    num_output_samples=18 time_stamp=$time_stamp

                ######### Macro-only evaluation for IBM and ISPD benchmarks:
                CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py method=eval_macro_only \
                    task=ibm.cluster512.v1 from_checkpoint=$from_checkpoint \
                    legalizer@_global_=none guidance@_global_=none \
                    num_output_samples=18 model.grad_descent_steps=20 \
                    model.hpwl_guidance_weight=16e-4  macros_only=True \
                    time_stamp=$time_stamp

                CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/eval.py method=eval_macro_only task=ispd2005 \
                    from_checkpoint=$from_checkpoint legalizer@_global_=none guidance@_global_=none num_output_samples=8 \
                    model.grad_descent_steps=20 model.hpwl_guidance_weight=16e-4  macros_only=True  time_stamp=$time_stamp
                    ;;
            *)
                echo "[WARN] 未知 time_stamp: $time_stamp"
                ;;
        esac
    done
done

