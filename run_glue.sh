mkdir ckpt 2>/dev/null
export TASK_NAME=mnli

CUDA_VISIBLE_DEVICES=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m debugpy --listen 5678 --wait-for-client run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ckpt/$TASK_NAME/ \
  --overwrite_output_dir  \
  --save_total_limit 1