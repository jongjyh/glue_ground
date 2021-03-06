mkdir ckpt 2>/dev/null
export TASK_NAME=mrpc

CUDA_VISIBLE_DEVICES=0  WANDB_DISABLED=0 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python  -m debugpy --listen 5672 run_ans.py \
  --model_name_or_path roberta-base \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --max_eval_samples 1000 \
  --learning_rate 2e-5 \
  --num_train_epochs 30 \
  --output_dir ckpt/$TASK_NAME+lora_skip/ \
  --overwrite_output_dir  \
  --logging_steps 250 \
  --save_total_limit 1 \
  --evaluation_strategy steps \