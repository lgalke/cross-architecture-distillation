# for each student, provide task-specific TrainingArguments
# these should apply only to distillation
task_specific_training_args = {
    'mlp': {
        'qnli': {
            'per_device_train_batch_size': 16,
        },
        'qqp': {
            'per_device_train_batch_size': 128,
            # 'gradient_accumulation_steps': 2,
            'per_device_eval_batch_size': 64,
            # 'eval_accumulation_steps': 2,
        },
        'rte': {
            'per_device_train_batch_size': 64,
        },
        'mnli': {
            'per_device_train_batch_size': 128,
            # 'gradient_accumulation_steps': 2,
        }
    },
    'linear_probe': {
        'qnli': {
            'per_device_train_batch_size': 16,
        },
        'qqp': {
            'per_device_train_batch_size': 16,
            #'gradient_accumulation_steps': 2,
            'per_device_eval_batch_size': 64,
            # 'eval_accumulation_steps': 2,
        },
        'rte': {
            'per_device_train_batch_size': 16,
        },
        'mnli': {
            'per_device_train_batch_size': 256,
            #'gradient_accumulation_steps': 2,
        }
    },
    'cnn_deep': {
        'qnli': {
            'per_device_train_batch_size': 16,
        },
        'qqp': {
            'per_device_train_batch_size': 128,
            'per_device_eval_batch_size': 64,
        },
        'rte': {
            'per_device_train_batch_size': 64,
        },
        'mnli': {
            'per_device_train_batch_size': 128,
        }
    },
    'diluted_cnn': {
        'qnli': {
            'per_device_train_batch_size': 64,
        },
        'qqp': {
            'per_device_train_batch_size': 128,
        },
        'rte': {
            'per_device_train_batch_size': 64,
        },
        'mnli': {
            'per_device_train_batch_size': 256,
        },
   },
    'lstm': {
        'qnli': {
            'per_device_train_batch_size': 8,
        },
        'qqp': {
            'per_device_train_batch_size': 128,
        },
        'rte': {
            'per_device_train_batch_size': 32,
        },
        'mnli': {
            'per_device_train_batch_size': 128,
        }
   },
    'wide_mlp': {
        'stsb': {
           'per_device_train_batch_size': 16,
        }
    }
}
