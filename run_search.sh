mkdir ckpt 2>/dev/null
export TASK_NAME=mrpc

CUDA_VISIBLE_DEVICES=2  WANDB_DISABLED=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -m debugpy --listen 5678 --wait-for-client run_ans.py \
  --model_name_or_path roberta-base \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --max_eval_samples 1000 \
  --learning_rate 2e-5 \
  --num_train_epochs 30 \
  --output_dir ckpt/$TASK_NAME+lora_random/ \
  --overwrite_output_dir  \
  --logging_steps 250 \
  --save_total_limit 1 \
  --evaluation_strategy steps \
  --do_search true \
  --metric eval/accuracy \
  --direction maximize \
  --n_trials 10 \