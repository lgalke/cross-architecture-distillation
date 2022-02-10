# Cross-Architecture Distillation Using Bidirectional CMOW Embeddings

This is the source code for the experiments of our submission entitled *Cross-Architecture Distillation Using Bidirectional CMOW Embeddings*.

Please find the code for the embedding models including our extensions and
the pretraining with general distillation in the folder `src/third_party/seq2mat/`.
In particular, `src/third_party/seq2mat/seq2mat/modeling_seq2mat.py` holds the code for the model.

The top-level source code in `src` implements the downstream classifiers and 
the task-specific distillation procedure.

The script to measure inference speed can be found at `src/third_party/seq2mat/measure_inference_speed.py`

The main entry points for our expeirments are:

- Pretraining with general distillation: `src/third_party/seq2mat/distillation/train.py` (adapted from hugginface/transformers repository)
- Fine-tuning (without task-specific distillation)`src/third_party/seq2mat/text-classification/run_glue.py` (adapted from hugginface/transformers repository)
- Fine-tuning (with task-specific distillation) `main.py` which delegates execution to `src/pipeline.py`.

The configuration for the Bidirectional CMOW/CBOW-Hybrid model that we have pretrained on the full unlabeled training data can be found at `src/third_party/seq2mat/config/seq2mat_hybrid_bidirectional_sbertlike.json`.

Thank you for taking the time to review our code.
