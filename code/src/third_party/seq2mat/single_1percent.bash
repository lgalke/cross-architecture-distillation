
CONFIG="$1"

if [ -z $CONFIG ]; then
	echo "Please give config as first argument. Exiting."
	exit 1
fi

filename=$(basename $CONFIG)
MODEL="${filename%.*}"

echo "Training $MODEL with config $CONFIG:"

cat $CONFIG


python3 distillation/train.py \
	--student_type seq2mat --student_config "$CONFIG" \
	--teacher_type bert --teacher_name bert-base-uncased \
	--alpha_ce 5.0 --alpha_mlm 2.0 --alpha_cos 0.0 --alpha_clm 0.0 --mlm \
	--dump_path "zoo/$MODEL-1p-ce5-mlm2" \
	--data_file data/TBC-EnWiki-1percent.bert-base-uncased.pickle \
	--token_counts data/TBC-EnWiki-1percent.token_counts.bert-base-uncased.pickle --force
