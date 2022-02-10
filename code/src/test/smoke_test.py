from src.pipeline import Pipeline


def main():
    students_to_test = ['mlp', 'diluted_cnn', 'lstm', 'linear_probe']
    #embeddings_to_test = ['cmow', 'cbow', 'hybrid']
    embeddings_to_test = []
    pretrained_embeddings = ['none', 'paper', 'ANONYMIZED']
    siamese = [None, 'diffcat', 'hadamard']

    extra_embeddings_no_pretrained = ['conv']

    for student in students_to_test:
        for siam in siamese:
            for embedding in embeddings_to_test:
                print(f"### {student}, {siam}, {embedding}")
                config = {'task': 'rte',
                          'embeddings': embedding,
                          'student': student,
                          'teacher': 'bert',
                          'tokenizer': 'bert',
                          'job_name': 'smoke_test',
                          'use_pretrained_teacher': True,
                          'debug': False,
                          'out_dir': 'models/seq2mat/stest/',
                          'epochs': 1,
                          'alpha': 0.1,
                          'temperatur': 10.0,
                          'loss': 'ce+kldiv',
                          'use_pretrained_embeddings': True, # todo also test this
                          'tag': None,
                          'seed': 42,
                          'learning_rate': 5e-05,
                          'cache_teacher_predictions': True,
                          'siamese': siam,
                          'lr_warmup_steps': 0}

                p = Pipeline()
                p.run_task(config)

        for student in students_to_test:
            for siam in siamese:
                for embedding in extra_embeddings_no_pretrained:
                    print(f"### {student}, {siam}, {embedding}")
                    config = {'task': 'rte',
                              'embeddings': embedding,
                              'student': student,
                              'teacher': 'bert',
                              'tokenizer': 'bert',
                              'job_name': 'smoke_test',
                              'use_pretrained_teacher': True,
                              'debug': False,
                              'out_dir': 'models/seq2mat/stest/',
                              'epochs': 1,
                              'alpha': 0.1,
                              'temperatur': 10.0,
                              'loss': 'ce+kldiv',
                              'use_pretrained_embeddings': False,
                              'tag': None,
                              'seed': 42,
                              'learning_rate': 5e-05,
                              'cache_teacher_predictions': True,
                              'siamese': siam,
                              'lr_warmup_steps': 0}

                    p = Pipeline()
                    p.run_task(config)

if __name__ == '__main__':
    main()
