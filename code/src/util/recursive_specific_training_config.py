# for each student, provide task-specific TrainingArguments
# these should apply only to distillation
recursive_specific_training_args = {
    'mlp': {
        'qqp': {
            'per_device_train_batch_size': 64,
            # 'gradient_accumulation_steps': 2,
            'per_device_eval_batch_size': 32,
            # 'eval_accumulation_steps': 2,
        },
        'rte': {
            'per_device_train_batch_size': 32,
        },
        'mnli': {
            'per_device_train_batch_size': 32,
            # 'gradient_accumulation_steps': 2,
        },
        'qnli': {
            'per_device_train_batch_size': 32,
            # 'gradient_accumulation_steps':
            'per_device_eval_batch_size': 16,
        },

    },
    'cnn_deep': {
        'qnli': {
            'per_device_train_batch_size': 16,
        },
        'qqp': {
            'per_device_train_batch_size': 64,
            'gradient_accumulation_steps': 2,
            'per_device_eval_batch_size': 64,
            # 'eval_accumulation_steps': 2,
        },
        'rte': {
            'per_device_train_batch_size': 64,
        },
        'mnli': {
            'per_device_train_batch_size': 64,
            'gradient_accumulation_steps': 2,
        }
    },
    'diluted_cnn': {
        'qnli': {
            'per_device_train_batch_size': 24,
        },
        'qqp': {
            'per_device_train_batch_size': 128,
            'gradient_accumulation_steps': 2,
            'per_device_eval_batch_size': 64,
            # 'eval_accumulation_steps': 2,
        },
        'rte': {
            'per_device_train_batch_size': 64,
        },
        'mnli': {
            'per_device_train_batch_size': 128,
            'gradient_accumulation_steps': 2,
        }
    },

}
