mkdir ckpt 2>/dev/null
export TASK_NAME=mnli
prune_module=0-11-output-intermediate
lr=2e-5
wd=1e-1
ep=2
l0_lamb=1e-1
tem=0.66666
drop_init=0.1
rand_w=false
max_grad_norm=1
a_lr=5e-3
# run_name=normal_${TASK_NAME}_bert_lr${lr}_wd${wd}_ep${ep}
run_name=${TASK_NAME}_bert_lr${lr}_wd${wd}_ep${ep}_prune${prune_module}_di${drop_init}_lamb${l0_lamb}_tem${tem}_rw${rand_w}_mgn${max_grad_norm}_alr${a_lr}
# CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=0 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m debugpy --listen 5660 run_glue.py \
CUDA_VISIBLE_DEVICES=3 WANDB_DISABLED=0 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python  run_glue.py \
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
  --prune_module $prune_module \
  --droprate_init $drop_init \
  --lamb $l0_lamb \
  --temperature $tem \
  --rand_w $rand_w \
  --max_grad_norm $max_grad_norm \
  --a_lr ${a_lr} \

