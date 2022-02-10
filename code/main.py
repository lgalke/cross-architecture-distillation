# settings.py
import logging as log
import pprint
from dotenv import load_dotenv
load_dotenv()
# !!!!! import this right from the start so it doesn't complain later on
from src.util.comet_ml_integration import is_cometml_available
if is_cometml_available():
    print("Using CometML!")
from src.pipeline import Pipeline
from src.parser.commandlineparser import CommandLineParser

# sbatch -p gpu_4 job.sh -t "rte" -c "mlp" --teacher "distillbert" -emb "hybrid" -pt

if __name__ == "__main__":
    parser = CommandLineParser()
    pipeline = Pipeline()
    args = parser.parse_args()
    configs = args.get("configs")

    pprint.pformat(configs)
    evaluations = []
    for config in configs:
        print("send task: {} on student {} pipeline".format(
            config.get("task"), config.get("student")))
        print("task_config:{}".format(config))
        evaluations.append(pipeline.run_task(config))
    print(evaluations)
