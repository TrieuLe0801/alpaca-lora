python finetune.py \
    --base_model 'medalpaca/medalpaca-7b' \
    --data_path 'medalpaca/medical_meadow_medqa' \
    --instruction_key 'instruction' \
    --input_key 'input' \
    --output_key 'output' \
    --output_dir './lora-alpaca' \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 1 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length \
    # --resume_from_checkpoint ./lora-alpaca/checkpoint-200