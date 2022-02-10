
MODEL="$1"
CONFIG="$2"
OUTPUT_SUFFIX="hpopt-1"


echo "Running model '$MODEL' with '$CONFIG'"
# TASKS: CoLA  MNLI  MRPC  QNLI  QQP  RTE  SNLI  SST-2  STS-B  WNLI

# usage: run_glue.py [-h] --model_name_or_path MODEL_NAME_OR_PATH
#                    [--config_name CONFIG_NAME]
#                    [--tokenizer_name TOKENIZER_NAME] [--cache_dir CACHE_DIR]
#                    --task_name TASK_NAME --data_dir DATA_DIR
#                    [--max_seq_length MAX_SEQ_LENGTH] [--overwrite_cache]
#                    --output_dir OUTPUT_DIR [--overwrite_output_dir]
#                    [--do_train] [--do_eval] [--do_predict]
#                    [--evaluate_during_training]
#                    [--per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE]
#                    [--per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE]
#                    [--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE]
#                    [--per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE]
#                    [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
#                    [--learning_rate LEARNING_RATE]
#                    [--weight_decay WEIGHT_DECAY] [--adam_epsilon ADAM_EPSILON]
#                    [--max_grad_norm MAX_GRAD_NORM]
#                    [--num_train_epochs NUM_TRAIN_EPOCHS]
#                    [--max_steps MAX_STEPS] [--warmup_steps WARMUP_STEPS]
#                    [--logging_dir LOGGING_DIR] [--logging_first_step]
#                    [--logging_steps LOGGING_STEPS] [--save_steps SAVE_STEPS]
#                    [--save_total_limit SAVE_TOTAL_LIMIT] [--no_cuda]
#                    [--seed SEED] [--fp16] [--fp16_opt_level FP16_OPT_LEVEL]
#                    [--local_rank LOCAL_RANK] [--tpu_num_cores TPU_NUM_CORES]
#                    [--tpu_metrics_debug] [--debug] [--dataloader_drop_last]
#                    [--eval_steps EVAL_STEPS] [--past_index PAST_INDEX]

# optional arguments:
#   -h, --help            show this help message and exit
#   --model_name_or_path MODEL_NAME_OR_PATH
#                         Path to pretrained model or model identifier from
#                         huggingface.co/models
#   --config_name CONFIG_NAME
#                         Pretrained config name or path if not the same as
#                         model_name
#   --tokenizer_name TOKENIZER_NAME
#                         Pretrained tokenizer name or path if not the same as
#                         model_name
#   --cache_dir CACHE_DIR
#                         Where do you want to store the pretrained models
#                         downloaded from s3
#   --task_name TASK_NAME
#                         The name of the task to train on: cola, mnli, mnli-mm,
#                         mrpc, sst-2, sts-b, qqp, qnli, rte, wnli
#   --data_dir DATA_DIR   The input data dir. Should contain the .tsv files (or
#                         other data files) for the task.
#   --max_seq_length MAX_SEQ_LENGTH
#                         The maximum total input sequence length after
#                         tokenization. Sequences longer than this will be
#                         truncated, sequences shorter will be padded.
#   --overwrite_cache     Overwrite the cached training and evaluation sets
#   --output_dir OUTPUT_DIR
#                         The output directory where the model predictions and
#                         checkpoints will be written.
#   --overwrite_output_dir
#                         Overwrite the content of the output directory.Use this
#                         to continue training if output_dir points to a
#                         checkpoint directory.
#   --do_train            Whether to run training.
#   --do_eval             Whether to run eval on the dev set.
#   --do_predict          Whether to run predictions on the test set.
#   --evaluate_during_training
#                         Run evaluation during training at each logging step.
#   --per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE
#                         Batch size per GPU/TPU core/CPU for training.
#   --per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE
#                         Batch size per GPU/TPU core/CPU for evaluation.
#   --per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE
#                         Deprecated, the use of `--per_device_train_batch_size`
#                         is preferred. Batch size per GPU/TPU core/CPU for
#                         training.
#   --per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE
#                         Deprecated, the use of `--per_device_eval_batch_size`
#                         is preferred.Batch size per GPU/TPU core/CPU for
#                         evaluation.
#   --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
#                         Number of updates steps to accumulate before
#                         performing a backward/update pass.
#   --learning_rate LEARNING_RATE
#                         The initial learning rate for Adam.
#   --weight_decay WEIGHT_DECAY
#                         Weight decay if we apply some.
#   --adam_epsilon ADAM_EPSILON
#                         Epsilon for Adam optimizer.
#   --max_grad_norm MAX_GRAD_NORM
#                         Max gradient norm.
#   --num_train_epochs NUM_TRAIN_EPOCHS
#                         Total number of training epochs to perform.
#   --max_steps MAX_STEPS
#                         If > 0: set total number of training steps to perform.
#                         Override num_train_epochs.
#   --warmup_steps WARMUP_STEPS
#                         Linear warmup over warmup_steps.
#   --logging_dir LOGGING_DIR
#                         Tensorboard log dir.
#   --logging_first_step  Log and eval the first global_step
#   --logging_steps LOGGING_STEPS
#                         Log every X updates steps.
#   --save_steps SAVE_STEPS
#                         Save checkpoint every X updates steps.
#   --save_total_limit SAVE_TOTAL_LIMIT
#                         Limit the total amount of checkpoints.Deletes the
#                         older checkpoints in the output_dir. Default is
#                         unlimited checkpoints
#   --no_cuda             Do not use CUDA even when it is available
#   --seed SEED           random seed for initialization
#   --fp16                Whether to use 16-bit (mixed) precision (through
#                         NVIDIA apex) instead of 32-bit
#   --fp16_opt_level FP16_OPT_LEVEL
#                         For fp16: Apex AMP optimization level selected in
#                         ['O0', 'O1', 'O2', and 'O3'].See details at
#                         https://nvidia.github.io/apex/amp.html
#   --local_rank LOCAL_RANK
#                         For distributed training: local_rank
#   --tpu_num_cores TPU_NUM_CORES
#                         TPU: Number of TPU cores (automatically passed by
#                         launcher script)
#   --tpu_metrics_debug   Deprecated, the use of `--debug` is preferred. TPU:
#                         Whether to print debug metrics
#   --debug               Whether to print debug metrics on TPU
#   --dataloader_drop_last
#                         Drop the last incomplete batch if it is not divisible
#                         by the batch size.
#   --eval_steps EVAL_STEPS
#                         Run an evaluation every X steps.
#   --past_index PAST_INDEX
#                         If >=0, uses the corresponding part of the output as
#                         the past state for next step.



TASK=CoLA; python3 text-classification/run_glue.py --model_name_or_path $MODEL \
	--config $CONFIG --tokenizer_name bert-base-uncased \
	--output_dir $MODEL/glue$OUTPUT_SUFFIX/$TASK \
	--task_name $TASK --data_dir /data21/lgalke/datasets/GLUE/glue_data/$TASK \
	--save_steps 10000 --save_total_limit 1 \
	--do_train --do_eval --num_train_epochs 70 

# TASK=MNLI; python3 text-classification/run_glue.py --model_name_or_path $MODEL \
# 	--config $CONFIG --tokenizer_name bert-base-uncased \
# 	--output_dir $MODEL/glue$OUTPUT_SUFFIX/$TASK --task_name $TASK \
# 	--data_dir /data21/lgalke/datasets/GLUE/glue_data/$TASK \
# 	--save_steps 10000 --save_total_limit 1 \
# 	--do_train --do_eval --num_train_epochs 9

# TASK=MRPC; python3 text-classification/run_glue.py --model_name_or_path $MODEL \
# 	--config $CONFIG --tokenizer_name bert-base-uncased \
# 	--output_dir $MODEL/glue$OUTPUT_SUFFIX/$TASK --task_name $TASK \
# 	--data_dir /data21/lgalke/datasets/GLUE/glue_data/$TASK \
# 	--save_steps 10000 --save_total_limit 1 \
# 	--do_train --do_eval

# TASK=QNLI; python3 text-classification/run_glue.py --model_name_or_path $MODEL \
# 	--config $CONFIG --tokenizer_name bert-base-uncased \
# 	--output_dir $MODEL/glue$OUTPUT_SUFFIX/$TASK --task_name $TASK \
# 	--data_dir /data21/lgalke/datasets/GLUE/glue_data/$TASK \
# 	--save_steps 10000 --save_total_limit 1 \
# 	--do_train --do_eval

# TASK=QQP; python3 text-classification/run_glue.py --model_name_or_path $MODEL \
# 	--config $CONFIG --tokenizer_name bert-base-uncased \
# 	--output_dir $MODEL/glue$OUTPUT_SUFFIX/$TASK --task_name $TASK \
# 	--data_dir /data21/lgalke/datasets/GLUE/glue_data/$TASK \
# 	--save_steps 10000 --save_total_limit 1 \
# 	--overwrite_output_dir \
# 	--do_train --do_eval --num_train_epochs 50

# TASK=RTE; python3 text-classification/run_glue.py --model_name_or_path $MODEL \
# 	--config $CONFIG --tokenizer_name bert-base-uncased \
# 	--output_dir $MODEL/glue$OUTPUT_SUFFIX/$TASK --task_name $TASK \
# 	--data_dir /data21/lgalke/datasets/GLUE/glue_data/$TASK \
# 	--save_steps 10000 --save_total_limit 1 \
# 	--do_train --do_eval --num_train_epochs 9

# TASK=SST-2; python3 text-classification/run_glue.py --model_name_or_path $MODEL \
# 	--config $CONFIG --tokenizer_name bert-base-uncased \
# 	--output_dir $MODEL/glue$OUTPUT_SUFFIX/$TASK --task_name $TASK \
# 	--data_dir /data21/lgalke/datasets/GLUE/glue_data/$TASK \
# 	--save_steps 10000 --save_total_limit 1 \
# 	--do_train --do_eval

# TASK=STS-B; python3 text-classification/run_glue.py --model_name_or_path $MODEL \
# 	--config $CONFIG --tokenizer_name bert-base-uncased \
# 	--output_dir $MODEL/glue$OUTPUT_SUFFIX/$TASK --task_name $TASK \
# 	--data_dir /data21/lgalke/datasets/GLUE/glue_data/$TASK \
# 	--save_steps 10000 --save_total_limit 1 \
# 	--do_train --do_eval --num_train_epochs 40

# TASK=WNLI; python3 text-classification/run_glue.py --model_name_or_path $MODEL \
# 	--config $CONFIG --tokenizer_name bert-base-uncased \
# 	--output_dir $MODEL/glue$OUTPUT_SUFFIX/$TASK --task_name $TASK \
# 	--data_dir /data21/lgalke/datasets/GLUE/glue_data/$TASK \
# 	--save_steps 10000 --save_total_limit 1 \
# 	--do_train --do_eval
