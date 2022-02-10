export NODE_RANK=0
export N_NODES=1

export N_GPU_NODE=2
export WORLD_SIZE=2
export MASTER_PORT=9999
export MASTER_ADDR=127.0.0.1

CONF="config/seq2mat_hybrid_bidirectional_sbertlike.json"
OUT="zoo/seq2mat_bidi_hybrid_100p"

pkill -f 'python -u distillation/train.py'

python -m torch.distributed.launch \
    --nproc_per_node=$N_GPU_NODE \
    --nnodes=$N_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    distillation/train.py \
        --force \
        --n_gpu $WORLD_SIZE \
        --student_type seq2mat \
        --student_config $CONF \
        --teacher_type bert \
        --teacher_name bert-base-uncased \
        --alpha_ce 0.5 --alpha_mlm 0.5 --alpha_cos 0.00 --alpha_clm 0.0 --mlm \
        --n_epoch 1 \
        --batch_size 128 \
        --gradient_accumulation_steps 8 \
        --freeze_pos_embs \
        --dump_path $OUT \
        --data_file data/TBC-EnWiki-simple.bert-base-uncased.pickle \
	--token_counts data/token_counts.bert-base-uncased.pickle \

# usage: train.py [-h] [--force] --dump_path DUMP_PATH --data_file DATA_FILE
#                 --student_type {distilbert,roberta,gpt2,seq2mat}
#                 --student_config STUDENT_CONFIG
#                 [--student_pretrained_weights STUDENT_PRETRAINED_WEIGHTS]
#                 --teacher_type {bert,roberta,gpt2} --teacher_name TEACHER_NAME
#                 [--temperature TEMPERATURE] [--alpha_ce ALPHA_CE]
#                 [--alpha_mlm ALPHA_MLM] [--alpha_clm ALPHA_CLM]
#                 [--alpha_mse ALPHA_MSE] [--alpha_cos ALPHA_COS] [--mlm]
#                 [--mlm_mask_prop MLM_MASK_PROP] [--word_mask WORD_MASK]
#                 [--word_keep WORD_KEEP] [--word_rand WORD_RAND]
#                 [--mlm_smoothing MLM_SMOOTHING] [--token_counts TOKEN_COUNTS]
#                 [--restrict_ce_to_mask] [--freeze_pos_embs]
#                 [--freeze_token_type_embds] [--n_epoch N_EPOCH]
#                 [--batch_size BATCH_SIZE] [--group_by_size]
#                 [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
#                 [--warmup_prop WARMUP_PROP] [--weight_decay WEIGHT_DECAY]
#                 [--learning_rate LEARNING_RATE] [--adam_epsilon ADAM_EPSILON]
#                 [--max_grad_norm MAX_GRAD_NORM]
#                 [--initializer_range INITIALIZER_RANGE] [--fp16]
#                 [--fp16_opt_level FP16_OPT_LEVEL] [--n_gpu N_GPU]
#                 [--local_rank LOCAL_RANK] [--seed SEED]
#                 [--log_interval LOG_INTERVAL]
#                 [--checkpoint_interval CHECKPOINT_INTERVAL]

# Training

# optional arguments:
#   -h, --help            show this help message and exit
#   --force               Overwrite dump_path if it already exists.
#   --dump_path DUMP_PATH
#                         The output directory (log, checkpoints, parameters,
#                         etc.)
#   --data_file DATA_FILE
#                         The binarized file (tokenized + tokens_to_ids) and
#                         grouped by sequence.
#   --student_type {distilbert,roberta,gpt2,seq2mat}
#                         The student type (DistilBERT, RoBERTa, Seq2mat).
#   --student_config STUDENT_CONFIG
#                         Path to the student configuration.
#   --student_pretrained_weights STUDENT_PRETRAINED_WEIGHTS
#                         Load student initialization checkpoint.
#   --teacher_type {bert,roberta,gpt2}
#                         Teacher type (BERT, RoBERTa).
#   --teacher_name TEACHER_NAME
#                         The teacher model.
#   --temperature TEMPERATURE
#                         Temperature for the softmax temperature.
#   --alpha_ce ALPHA_CE   Linear weight for the distillation loss. Must be >=0.
#   --alpha_mlm ALPHA_MLM
#                         Linear weight for the MLM loss. Must be >=0. Should be
#                         used in coonjunction with `mlm` flag.
#   --alpha_clm ALPHA_CLM
#                         Linear weight for the CLM loss. Must be >=0.
#   --alpha_mse ALPHA_MSE
#                         Linear weight of the MSE loss. Must be >=0.
#   --alpha_cos ALPHA_COS
#                         Linear weight of the cosine embedding loss. Must be
#                         >=0.
#   --mlm                 The LM step: MLM or CLM. If `mlm` is True, the MLM is
#                         used over CLM.
#   --mlm_mask_prop MLM_MASK_PROP
#                         Proportion of tokens for which we need to make a
#                         prediction.
#   --word_mask WORD_MASK
#                         Proportion of tokens to mask out.
#   --word_keep WORD_KEEP
#                         Proportion of tokens to keep.
#   --word_rand WORD_RAND
#                         Proportion of tokens to randomly replace.
#   --mlm_smoothing MLM_SMOOTHING
#                         Smoothing parameter to emphasize more rare tokens (see
#                         XLM, similar to word2vec).
#   --token_counts TOKEN_COUNTS
#                         The token counts in the data_file for MLM.
#   --restrict_ce_to_mask
#                         If true, compute the distilation loss only the [MLM]
#                         prediction distribution.
#   --freeze_pos_embs     Freeze positional embeddings during distillation. For
#                         student_type in ['roberta', 'gpt2'] only.
#   --freeze_token_type_embds
#                         Freeze token type embeddings during distillation if
#                         existent. For student_type in ['roberta'] only.
#   --n_epoch N_EPOCH     Number of pass on the whole dataset.
#   --batch_size BATCH_SIZE
#                         Batch size (for each process).
#   --group_by_size       If true, group sequences that have similar length into
#                         the same batch. Default is true.
#   --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
#                         Gradient accumulation for larger training batches.
#   --warmup_prop WARMUP_PROP
#                         Linear warmup proportion.
#   --weight_decay WEIGHT_DECAY
#                         Weight deay if we apply some.
#   --learning_rate LEARNING_RATE
#                         The initial learning rate for Adam.
#   --adam_epsilon ADAM_EPSILON
#                         Epsilon for Adam optimizer.
#   --max_grad_norm MAX_GRAD_NORM
#                         Max gradient norm.
#   --initializer_range INITIALIZER_RANGE
#                         Random initialization range.
#   --fp16                Whether to use 16-bit (mixed) precision (through
#                         NVIDIA apex) instead of 32-bit
#   --fp16_opt_level FP16_OPT_LEVEL
#                         For fp16: Apex AMP optimization level selected in
#                         ['O0', 'O1', 'O2', and 'O3'].See details at
#                         https://nvidia.github.io/apex/amp.html
#   --n_gpu N_GPU         Number of GPUs in the node.
#   --local_rank LOCAL_RANK
#                         Distributed training - Local rank
#   --seed SEED           Random seed
#   --log_interval LOG_INTERVAL
#                         Tensorboard logging interval.
#   --checkpoint_interval CHECKPOINT_INTERVAL
#                         Checkpoint interval.
