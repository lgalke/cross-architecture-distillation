from transformers import Trainer


class TeacherTrainer():
    """
        class for finetuning teacher and calculate predictions,
    """
# ============= CONSTRUCTOR =============

    def __init__(self,
                 job_name,
                 model,
                 train_dataset,
                 eval_dataset,
                 compute_metrics,
                 tokenizer,
                 training_args,
                 num_labels,
                 debug
                 ) -> None:
        super().__init__()
        self.trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
            data_collator=None,
            args=training_args
        )
        self.job_name = job_name
        self.num_labels = num_labels
        self.predictions = None
        self.debug = debug


# ============= HELPER FUNCTIONS =============

    def __update_train_dataset_with_trainer_preds(self, example, idx):
        """updates train dataset with prediction labels

        Args:
            example : a series
            idx (Number): the index for this prediction

        Returns:
            retval: a extended series containing the prediction label
        """
        retval = example
        if self.num_labels == 1:  # only stsb
            retval['label'] = (
                example['label'], self.predictions.predictions[idx][0])  # float
        elif self.num_labels == 2:
            label = [1, 0]
            if example['label'] == 1:
                label = [0, 1]
            retval['label'] = (label, self.predictions.predictions[idx])
        # num_labels == 3, mnli and mnli-mm (label column has values 0, 1 or 2)
        else:
            label = [1, 0, 0]
            if example['label'] == 1:
                label = [0, 1, 0]
            else:  # example['label'] == 2
                label = [0, 0, 1]
            retval['label'] = (label, self.predictions.predictions[idx])
        return retval

# ============= TRAIN AND PREDICT =============

    def train_and_predict(self):
        """fine tuning teacher and save predictions

        Returns:
            [type]: distil_train_dataset
        """
        # Training
        if self.debug:
            self.trainer.args.num_train_epochs = 1

        print("start fine_tuning teacher with {} epochs and {} labels".format(
            self.trainer.args.num_train_epochs, self.num_labels))

        self.trainer.train(model_path=None)

        # evaluate & save
        if not self.debug:
            # Saves the tokenizer too for easy upload
            self.trainer.save_model(
                output_dir=f"models/seq2mat/{self.job_name}/")
            eval_result = self.trainer.evaluate(eval_dataset=self.eval_dataset)
            print(eval_result)

        # preprocess the dataset AGAIN!
        print("predicting using tuned bert..")
        self.predictions = self.trainer.predict(self.train_dataset)

        distil_train_dataset = self.train_dataset.map(
            self.__update_train_dataset_with_trainer_preds, with_indices=True)

        distil_train_dataset.save_to_disk('./distil_train_dataset.bin')
        self.distil_train_dataset = distil_train_dataset
        return distil_train_dataset
