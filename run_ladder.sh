mkdir ckpt 2>/dev/null
export TASK_NAME=mrpc
lr=2e-4
wd=1e-1
ep=20
# run_name=normal_${TASK_NAME}_bert_lr${lr}_wd${wd}_ep${ep}
run_name=${TASK_NAME}_bert_lr${lr}_wd${wd}_ep${ep}
# CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=0 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python  run_ladder.py \
CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m debugpy --listen 5660 run_ladder.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate $lr \
  --num_train_epochs $ep \
  --output_dir ckpt/$run_name/ \
  --run_name $run_name \
  --weight_decay $wd \
  --overwrite_output_dir  \
  --evaluation_strategy steps \
  --save_total_limit 1 \
  --logging_steps 250 \
  --metric_for_best_model eval_accuracy  \
  --greater_is_better 1 \
