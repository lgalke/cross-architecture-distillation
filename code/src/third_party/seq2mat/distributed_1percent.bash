export NODE_RANK=0
export N_NODES=1

export N_GPU_NODE=4
export WORLD_SIZE=2
export MASTER_PORT=9999
export MASTER_ADDR=127.0.0.1

CONF="config/seq2mat_conv_6layer_attn.json"
OUT="zoo/seq2mat_6layer_attn_1p"

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
        --freeze_pos_embs \
        --dump_path $OUT \
        --data_file data/TBC-EnWiki-1percent.bert-base-uncased.pickle \
	--token_counts data/TBC-EnWiki-1percent.token_counts.bert-base-uncased.pickle
