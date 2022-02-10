CONFIG="$1"

if [ -z $CONFIG ]; then
	echo "Please give config as first argument. Exiting."
	exit 1
fi
python3 distillation/train.py --student_type seq2mat --student_config "$CONFIG" --teacher_type bert --teacher_name bert-base-uncased --alpha_ce 5.0 --alpha_mlm 2.0 --alpha_clm 0.0 --mlm --dump_path zoo/seq2mat_debug_v0 --data_file data/debug.bert-base-uncased.pickle --token_counts data/debug_token_counts.bert-base-uncased.pickle --force
