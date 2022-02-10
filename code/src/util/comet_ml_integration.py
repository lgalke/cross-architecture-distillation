import os
import importlib.util

try:
    import comet_ml
except ImportError:
    pass


_has_comet = importlib.util.find_spec("comet_ml") is not None and os.getenv("COMET_MODE", "").upper() != "DISABLED"

# permanently disable wandb, in distributed setup we might have it installed
os.environ['WANDB_DISABLED'] = '1'

def is_cometml_available():
    return _has_comet


def annotate_current_experiment(task_config, eval_result, step_name="student-distillation", other={}, tag=None):
    """
    step_name identifies at which stage of the distillation this training step was performed
      for fine-tuning BERT/teacher, use teacher-fintetuning
      for student training, use student-training
      it can be any arbitrary string though, just make sure to be consistent
      users can define a custom tag for a run, e.g. sunday_afternoon_mlp
    """
    if is_cometml_available():
        experiment = comet_ml.config.get_global_experiment()
        if experiment is None:
            return

        experiment.log_parameters(
            task_config
        )

        experiment.add_tag(step_name)
        experiment.add_tag(task_config['task'])
        experiment.add_tag(task_config['embeddings'])
        experiment.log_metrics(eval_result, prefix='best')
        # explicitly log cmdline learning rate to prevent comet from overwriting it
        # while we're at it, also log other parameters explicitly as hyperparameters
        experiment.log_parameters({
            'arg_learning_rate': task_config['learning_rate'],
            'embedding_type': task_config['embeddings'],
            'student_type': task_config['student']
        })
        experiment.log_metrics(other)

        if tag is not None:
            experiment.add_tag(tag)

        # log slurm info, if available
        if os.getenv("SLURM_JOB_ID"):
            experiment.log_others({'slurm_job_id': os.getenv('SLURM_JOB_ID', '')})

