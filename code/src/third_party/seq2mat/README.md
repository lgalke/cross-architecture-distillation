# Seq2mat

This repository adds a `Seq2mat` model with the transformer interface.

## Dependencies

- pytorch -- please follow CPU / CUDA dependent installation guide
- transformers -- Should work via `pip install transformers`

## Usage

Below, you find a minimal example of using the new model in this repository.

```python3
import json
import torch
from seq2mat import Seq2matConfig, Seq2matForSequenceClassification
from transformers import BertTokenizer

# Init tokenizer (depends on teacher model, here: BERT)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Init model from config file
with open('config/seq2mat_XYZ.json') as fhandle:
    config = Seq2matConfig(**json.load(fhandle))
classifier = Seq2matForSequenceClassification(config)

# Mini-example of how to use the model
tokens = tokenizer.encode("The quick brown fox jumps over the lazy dog")
outputs = classifier(input_ids=tokens)
# output format depends on 'mode' specified in config.
# You can look it up in seq2mat/modeling_seq2mat.py
# at forward method of class Seq2matForSequenceClassification

# Accessing model parameters, e.g., for the optimizer
optimizer = torch.optim.Adam(classifier.parameters())

# ... <training loop> ...
```

## Relevant files

| File | Description |
| ---- | ----------- |
| `seq2mat/modeling_seq2mat.py` | Code for Seq2mat architectures |
| `seq2mat/configuration_seq2mat.py` | Code for Seq2matConfig |
| `seq2mat/matmul_pooling.py` | Utils for Seq2mat 1d-conv architecture |
| `seq2mat/recursive.py` | Utils for Seq2mat recursive architecture |
| `seq2mat/tokenization_seq2mat.py` | Unused! Use teacher's tokenizer instead |
| `distillation/` | Code for distillation |
| `text-classification/` | Code for text classification (GLUE benchmark)|
| `question-answering/` | Code for question answering (SQuAD 1.1 and 2.0)|

---

> Below is a copy of the [README file for adding a new transformer](https://github.com/huggingface/transformers/blob/master/templates/adding_a_new_model/README.md).

## How to add a new model in 🤗Transformers

This folder describes the process to add a new model in 🤗Transformers and provide templates for the required files.

The library is designed to incorporate a variety of models and code bases. As such the process for adding a new model usually mostly consists in copy-pasting to relevant original code in the various sections of the templates included in the present repository.

One important point though is that the library has the following goals impacting the way models are incorporated:

- one specific feature of the API is the capability to run the model and tokenizer inline. The tokenization code thus often have to be slightly adapted to allow for running in the python interpreter.
- the package is also designed to be as self-consistent and with a small and reliable set of packages dependencies. In consequence, additional dependencies are usually not allowed when adding a model but can be allowed for the inclusion of a new tokenizer (recent examples of dependencies added for tokenizer specificities include `sentencepiece` and `sacremoses`). Please make sure to check the existing dependencies when possible before adding a new one.

For a quick overview of the library organization, please check the [QuickStart section of the documentation](https://huggingface.co/transformers/quickstart.html).

## Typical workflow for including a model

Here an overview of the general workflow: 

- [ ] add model/configuration/tokenization classes
- [ ] add conversion scripts
- [ ] add tests
- [ ] finalize

Let's detail what should be done at each step

### Adding model/configuration/tokenization classes

Here is the workflow for adding model/configuration/tokenization classes:

- [ ] copy the python files from the present folder to the main folder and rename them, replacing `xxx` with your model name,
- [ ] edit the files to replace `XXX` (with various casing) with your model name
- [ ] copy-paste or create a simple configuration class for your model in the `configuration_...` file
- [ ] copy-paste or create the code for your model in the `modeling_...` files (PyTorch and TF 2.0)
- [ ] copy-paste or create a tokenizer class for your model in the `tokenization_...` file

## Adding conversion scripts

Here is the workflow for the conversion scripts:

- [ ] copy the conversion script (`convert_...`) from the present folder to the main folder.
- [ ] edit this script to convert your original checkpoint weights to the current pytorch ones.

## Adding tests:

Here is the workflow for the adding tests:

- [ ] copy the python files from the `tests` sub-folder of the present folder to the `tests` subfolder of the main folder and rename them, replacing `xxx` with your model name,
- [ ] edit the tests files to replace `XXX` (with various casing) with your model name
- [ ] edit the tests code as needed

## Final steps

You can then finish the addition step by adding imports for your classes in the common files:

- [ ] add import for all the relevant classes in `__init__.py`
- [ ] add your configuration in `configuration_auto.py`
- [ ] add your PyTorch and TF 2.0 model respectively in `modeling_auto.py` and `modeling_tf_auto.py`
- [ ] add your tokenizer in `tokenization_auto.py`
- [ ] add your models and tokenizer to `pipeline.py`
- [ ] add a link to your conversion script in the main conversion utility (in `commands/convert.py`)
- [ ] edit the PyTorch to TF 2.0 conversion script to add your model in the `convert_pytorch_checkpoint_to_tf2.py` file
- [ ] add a mention of your model in the doc: `README.md` and the documentation itself at `docs/source/pretrained_models.rst`.
- [ ] upload the pretrained weights, configurations and vocabulary files.
- [ ] create model card(s) for your models on huggingface.co. For those last two steps, check the [model sharing documentation](https://github.com/huggingface/transformers#quick-tour-of-model-sharing).
