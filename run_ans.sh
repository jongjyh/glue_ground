mkdir ckpt 2>/dev/null
export TASK_NAME=mrpc

CUDA_VISIBLE_DEVICES=2  WANDB_DISABLED=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m debugpy --listen 5678 --wait-for-client run_ans.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-4 \
  --num_train_epochs 5 \
  --output_dir ckpt/$TASK_NAME/ \
  --overwrite_output_dir  \
  --logging_steps 100 \
  --save_total_limit 1 \
  --evaluation_strategy steps \