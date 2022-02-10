"""
    This class parses user input into an config file.

"""

import argparse
import datetime
import os
import typing
import random
import sys


class CommandLineParser(argparse.ArgumentParser):
    def __init__(self) -> None:
        super().__init__(description="Distillation Framework for help please use --help command")
        self.config_dict = {
            "configs": []
        }
        self.__init_argparse__()

    def __init_argparse__(self):
        """initialise argparse

        Returns:
        argparse.ArgumentParser: a parser containining all arguments 
        """
        self.add_argument(
            "-t", "--task",
            dest="task",
            nargs="+",
            help="Identifier for the task")

        self.add_argument(
            "-e", "--epochs", dest="epochs", help="train epochs for student", default=20, type=int)

        self.add_argument(
            "-emb", "--embbeddings",
            dest="embeddings",
            default="hybrid",
            type=str,
            help="Lukas student embeddings for which we have a json",
            choices=["cmow", "hybrid", "conv", "recursive", "cbow"]
        )

        self.add_argument(
            "-tg", "--tag",
            dest="tag",
            help="Custom Tag for CometML")

        self.add_argument(
            "-c", "--classifier",
            dest="student",
            help="Identifier for the student classifier model", type=str, required=True)

        self.add_argument(
            "-d", "--debug", dest="debug", help="run in debug mode --> run with less epochs and data", action='store_true')

        self.add_argument("-teach", "--teacher", dest="teacher",
                          help="teacher identifier: \n\
            albert -> AlbertForSequenceClassification\n\
            bert -> BertForSequenceClassification\n\
            distillbert -> DistilBertForSequenceClassification", default="bert")

        self.add_argument("-toke", "--tokenizer", dest="tokenizer",
                          help="tokenizer identifier", default="bert")

        self.add_argument("-o", "--output", help="output directory", dest="out_dir",
                          default="models/seq2mat/")
        self.add_argument("-j", "--job", metavar="", nargs='?',
                          help="job name", dest="job")

        self.add_argument(
            "-pt",
            "--use-pretrained-teacher",
            help="If defined, tries to load a pre-trained teacher model from common paths. "
                 "If no suitable model is found, training will fail.",
            action='store_true'
        )

        self.add_argument(
            "-b",
            "--batch",
            metavar="KEY=VALUE",
            nargs="*",
            help="batch size"
        )

        self.add_argument(
            "-a",
            "--alpha",
            dest="alpha",
            metavar="KEY=VALUE",
            nargs="*",
            help="alpha is a weight for the hard loss. alpha=1 means only hard_loss,\n\
                as key please name the task and as value the alpha value.\n\n\
                the default value is 0.1"
        )

        self.add_argument(
            "-T",
            "--temperatur",
            metavar="KEY=VALUE",
            nargs="*",
            help="softmax temperature for distillatio\n\
                as key please name the task and as value the temperatur value.\n\n\
                the default value is 10."

        )

        self.add_argument(
            "-l",
            "--loss",
            #choices=["mse-raw", "mse-softmax", "ce+mse", "ce+kldiv", "bce"],
            metavar="KEY=VALUE",
            nargs="*",
            help="loss for distillatio\n\
                as key please name the task and as value the loss identifier.\n\n\
                the default value is ce+kldiv."
        )

        self.add_argument(
            '-pew',
            '--use-pretrained-embeddings',
            help="for CMOW, CBOW and Hybrid configurations, enables loading of pretrained embeddings",
            action='store_true'
        )

        self.add_argument(
            "-lr",
            "--learnrate",
            help="learningrate for all tasks, at default the pytorch default will be used",
            type=float,
            # adamW default (taken from https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments)
            default=5e-5

        )

        self.add_argument(
            "-sia",
            "--siamese",
            help="use siamese for all tasks",
            default=None,
            choices=[None, "hadamard", "concat",
                     "diffcat", "cosine", "mean"]
        )

        self.add_argument(
            '-sd',
            '--seed',
            help='if not defined, use random seed for experiments. to reproduce experiments, always specify this seed.'
                 'If not specified, we\'ll use a random seed',
            type=int,
            default=random.randint(0, 1000000)
        )

        self.add_argument(
            "-ctp", "--cache-predictions", dest="cache_teacher_predictions",
            help="cache teacher predictions for each task/teacher combination."
                 "this can speed up training significantly, because we don't need to forward-pass"
                 "the whole dataset for generating soft-labels anymore.", action='store_true')

        self.add_argument(
            "-wus", "--warmup-steps", dest="lr_warmup_steps",
            help="how many warmup steps for the learning rate scheduler", type=int, default=500)

        self.add_argument(
            "-dsp", "--custom-dataset-path",
            dest="custom_dataset_path",
            help="Pass a directory that contains customized datasets (for augmentation, f.ex.)",
            type=str, required=False)

        self.add_argument(
            "-ep", "--custom-embedding-path",
            dest="custom_embedding_path",
            help="Pass a directory that contains customized embeddings (in checkpoint format)",
            type=str, required=False)

        self.add_argument(
            "-bd", "--bidirectional",
            dest="bidirectional",
            help="using bidirectional embeding",
            action='store_true'
        )


    def __append_config_dict__(
        self,
        embeddings: str,
        task: str,
        student: str,
        teacher="bert",
        tokenizer="bert",
        debug=False,
        out_dir="models/seq2mat",
        job_name="",
        use_pretrained_teacher=False,
        epochs=20,
        alpha=0.1,
        temperatur=10.,
        loss="mse",
        use_pretrained_embeddings=False,
        tag=None,
        seed=random.randint(0, 1000000),
        learning_rate=None,
        cache_preds=False,
        siamese=None,
        lr_warmup_steps=500,
        custom_dataset_path=None,
        custom_embedding_path=None,
        bidirectional=False
    ):
        job_name = self.__create_job_name__(student, task, job_name)
        out_dir = self.__create_out_dir_path__(out_dir, job_name)
        dict = {
            "task": task,
            "embeddings": embeddings,
            "student": student,
            "teacher": teacher,
            "tokenizer": tokenizer,
            "job_name": job_name,
            "use_pretrained_teacher": use_pretrained_teacher,
            "debug": debug,
            "out_dir": out_dir,
            "epochs": epochs,
            "alpha": alpha,
            "temperatur": temperatur,
            "loss": loss,
            "use_pretrained_embeddings": use_pretrained_embeddings,
            "tag": tag,
            "seed": seed,
            "learning_rate": learning_rate,
            "cache_teacher_predictions": cache_preds,
            "siamese": siamese,
            "lr_warmup_steps": lr_warmup_steps,
            "custom_dataset_path": custom_dataset_path,
            "custom_embedding_path": custom_embedding_path,
            "bidirectional": bidirectional
        }
        self.config_dict["configs"].append(dict)

    def __create_job_name__(self, student, task, job=""):
        date = datetime.datetime.now().strftime('%d-%m-%y_%H-%M-%S')
        if not job:
            job_name = "__".join([student, task, date])
        else:
            job_name = "__".join([job, student, task, date])
        return job_name

    def __create_out_dir_path__(self, out_dir, job_name):
        dir_name = os.path.join(out_dir, job_name+"/")
        return dir_name

    def __parse_to_str_dict__(self, arg_list: typing.List[str]) -> typing.Dict[str, str]:
        """
        parses a list of string to an dictionary.
        using the '=' symbol to for seperating key and value.
        A list like ["key_a=a", "key_b=b"] becomes a dict like:
        {
            "key_a":"a",
            "key_b":"b"
        }

        Args:
            arg_list (typing.List[str]): is a list of strings like this: ["key_a=a", "key_b=b"]

        Returns:
            typing.Dict: a dict which is empty when there is nothing to parse
        """
        finetuned_dict = {}
        if arg_list:
            for val in arg_list:
                key, val = val.split("=")
                finetuned_dict[key] = val
        return finetuned_dict

    def __parse_to_float_dict__(self, alpha_args: typing.List[str]) -> typing.Dict[str, float]:
        """
        parses a list of string to an dictionary and casts the value to a float
        using the '=' symbol to for seperating key and value.
        A list like ["key_a=2", "key_b=11.2"] becomes a dict like:
        {
            "key_a":2.,
            "key_b":11.2
        }

        Args:
            arg_list (typing.List[str]): is a list of strings like this: ["key_a=1", "key_b=2-"]

        Returns:
            typing.Dict: a dict which is empty when there is nothing to parse
        """
        alpha_dict = {}
        if alpha_args:
            for val in alpha_args:
                key, val = val.split("=")
                val = float(val)
                alpha_dict[key] = val
        return alpha_dict

    def parse_args(self):
        args = super(CommandLineParser, self).parse_args()
        print(args.__dict__)
        tasks = args.task
        alpha_dict = self.__parse_to_float_dict__(args.alpha)
        temp_dict = self.__parse_to_float_dict__(args.temperatur)
        loss_dict = self.__parse_to_str_dict__(args.loss)

        for task in tasks:
            print(task)
            self.__append_config_dict__(
                task=task,
                embeddings=args.embeddings,
                student=args.student,
                teacher=args.teacher,
                tokenizer=args.tokenizer,
                debug=args.debug,
                out_dir=args.out_dir,
                epochs=args.epochs,
                use_pretrained_teacher=args.use_pretrained_teacher,
                alpha=alpha_dict.get(task, 0.1),
                temperatur=temp_dict.get(task, 10.),
                loss=loss_dict.get(task, "bce"),
                use_pretrained_embeddings=args.use_pretrained_embeddings,
                tag=args.tag,
                seed=args.seed,
                learning_rate=args.learnrate,
                cache_preds=args.cache_teacher_predictions,
                siamese=args.siamese,
                lr_warmup_steps=args.lr_warmup_steps,
                custom_dataset_path=args.custom_dataset_path,
                custom_embedding_path=args.custom_embedding_path,
                bidirectional=args.bidirectional,
            )
        return self.config_dict
