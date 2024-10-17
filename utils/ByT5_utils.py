# imports
from transformers import DataCollatorWithPadding, TrainerCallback
from transformers import Trainer, TrainingArguments
from sklearn.metrics import confusion_matrix, roc_curve
from datasets import Dataset
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch


# prepare data functions
def prepare_data(
        data,
        tokenizer,
        max_length=50
):
    """
    Preprocesses the training data for the URL model.

    Args:
        data (pandas.DataFrame):
            The input data containing the "url" and "label" columns.
        tokenizer (Tokenizer):
            The tokenizer object used to tokenize the URLs.
        max_length (int, optional):
            The maximum length of the tokenized sequences.
            Defaults to 50

    Returns:
        torch.utils.data.Dataset:
            The preprocessed training data in the form of a PyTorch Dataset.

    Requirements for data:
    - Must have a column named "url" containing the URLs.
    - Must have a column named "label" containing the true labels of the URLs.
    """
    data = Dataset.from_pandas(data)

    def tokenize_data(data, tokenizer=tokenizer, max_length=max_length):
        inputs = data['url']
        model_inputs = tokenizer(
            inputs,
            max_length=max_length,
            truncation=True,
            padding=True
        )
        model_inputs['labels'] = data['label']
        return model_inputs

    data = data.map(tokenize_data, batched=True)
    data.set_format(type='torch',
                    columns=[
                        'input_ids',
                        'attention_mask',
                        'labels'
                    ])
    return data


# train function
def train_ByT5(
    model,
    tokenizer,
    train_data,
    val_data,
    training_args=None,
    need_prepare_data=True,
    max_length=50,
    patience=5,
    callback=None,
):
    """
    Trains a ByT5 model using the provided data.

    Args:
        model (PreTrainedModel):
            The pre-trained model to be trained.
        tokenizer (PreTrainedTokenizer):
            The tokenizer used for tokenizing the input data.
        train_data (Dataset):
            The training dataset.
            You can either provide a preprocessed dataset or raw data.
        val_data (Dataset):
            The validation dataset.
            You can either provide a preprocessed dataset or raw data.
        training_args (TrainingArguments, optional):
            The training arguments for the Trainer.
            If not provided, default hyperparameters will be used.
        need_prepare_data (bool, optional):
            Whether to preprocess the data before training
            Defaults to True.
        max_length (int, optional):
            The maximum length of the tokenized sequences
            Defaults to 50.
        patience (int, optional):
            The number of epochs to wait for the loss to improve.
            Defaults to 5.
        callback (TrainerCallback, optional):
            The callback to use during training.
            If not provided, a default callback will be used.

    Returns:
        Trainer: The trained Trainer object.
    """

    # prepare data
    if need_prepare_data:
        print("Preparing data...")
        train_data = prepare_data(train_data, tokenizer, max_length)
        val_data = prepare_data(val_data, tokenizer, max_length)
        print("Data prepared.")

    # default hyperparameters
    # refer to https://huggingface.co/docs/transformers/en/main_classes/trainer
    if training_args is None:
        BATCH_SIZE = 256
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=100,
            weight_decay=0.01,
            metric_for_best_model='Validation Loss',
            load_best_model_at_end=True,
            save_total_limit=5,
        )

    # create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=DataCollatorWithPadding(tokenizer)
    )
    if callback is None:
        trainer.add_callback(HF_Trainer_Callback(patience=patience))

    trainer.train()

    return trainer


# eval functions
def evaluate_ByT5(trainer):
    """
    Evaluates the performance of the ByT5 model using the provided trainer.

    Args:
        trainer: The trainer object used for training the ByT5 model.

    Returns:
        metrics: A dictionary containing the evaluation metrics of the model.
    """
    metrics = trainer.evaluate()
    return metrics


def predict_single_url(
    url,
    model,
    tokenizer,
    max_length=50,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    Predicts the class and probability of a given URL using
    a pre-trained model.

    Args:
        url (str):
            The URL to be predicted.
        model (torch.nn.Module):
            The pre-trained model used for prediction.
        tokenizer (transformers.PreTrainedTokenizer):
            The tokenizer used to tokenize the input URL.
        max_length (int, optional):
            The maximum length of the tokenized sequences (default: 50).
        device (torch.device, optional):
            The device to run the prediction on
            (default: "cuda" if available, else "cpu").

    Returns:
        tuple:
            A tuple containing the predicted class (0 or 1)
            and the probability of the URL being classified as class 1.
    """
    # Tokenize the input
    inputs = tokenizer(
        url,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    ).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'])
        logits = outputs['logits']

    # Convert logits to probabilities
    probabilities = F.softmax(logits, dim=1)

    # Get the predicted class
    predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class, probabilities[0][1].item()


def predict_dataframe(
        data,
        model,
        tokenizer,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        steps=1000
):
    """
    Predicts the classes and probabilities for a dataframe
    of URLs using a given model and tokenizer.

    Args:
        data (pandas.DataFrame):
            The dataframe containing the URLs to predict.
        model (torch.nn.Module):
            The trained model to use for prediction.
        tokenizer (transformers.PreTrainedTokenizer):
            The tokenizer to use for tokenizing the URLs.
        device (torch.device, optional):
            The device to use for prediction
            Defaults to "cuda" if available, else "cpu".
        steps (int, optional):
            The number of samples to process before printing a progress update
            Defaults to 1000.

    Returns:
        y_true (list):
            The true labels of the URLs.
        predicted_classes (list):
            The predicted classes for the URLs.
        probabilities (list):
            The predicted probabilities for the URLs.

    Requirements for dataframe:
    - Must have a column named "url" containing the URLs to predict.
    - Must have a column named "label" containing the true labels of the URLs.
    """

    print(f"Processing {len(data)} samples")
    y_true = []
    predicted_classes = []
    probabilities = []
    for i in range(len(data)):
        y_true.append(data["label"][i])

        predicted_class, probability = predict_single_url(
            url=data["url"][i],
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

        predicted_classes.append(predicted_class)
        probabilities.append(probability)
        if i % steps == 0 and i != 0:
            print(f"Processed {i} samples")
    print(f"Done, processed {len(data)} samples")
    return y_true, predicted_classes, probabilities


# callback
class HF_Trainer_Callback(TrainerCallback):
    """
    Callback class for the Hugging Face Trainer.

    This callback tracks the evaluation loss during training and stops training
    if the loss does not improve for a certain number of epochs (patience).

    Args:
        patience (int, optional):
            The number of epochs to wait for the loss to improve.
            If the loss does not improve for `patience` epochs,
            training will stop.
            Defaults to 5.
    """

    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.inf

    def on_train_begin(self, args, state, control, **kwargs):
        print("Starting training")

    def on_evaluate(self, args, state, control, **kwargs):
        logs = kwargs.get("metrics", {})
        validation_loss = logs.get("eval_loss")
        if validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        print("Training ended")


# metrics
def fpr_comparison(
    y_true,
    y_proba,
    fprs=np.round(np.arange(0.01, -0.001, -0.001), decimals=3),
    pos_label=1
):
    df_vals = []
    fpr, _, thresholds = roc_curve(y_true, y_proba, pos_label=pos_label)
    for target_fpr in fprs:
        fpr_ind = np.argwhere(fpr <= target_fpr)
        if fpr_ind.shape[0] == 0:
            temp_dict = {"precision": None,
                         "fpr": None,
                         "specificity": None,
                         "accuracy": None,
                         "recall": None,
                         "f1_score": None,
                         "threshold": None}
        else:
            threshold = thresholds[fpr_ind[-1][0]]
            curr_y_pred = y_proba.copy()
            curr_y_pred[curr_y_pred >= threshold] = 1
            curr_y_pred[curr_y_pred < threshold] = 0

            tn, fp, fn, tp = confusion_matrix(y_true, curr_y_pred).ravel()
            accuracy = 100 * (tp + tn) / (tp + tn + fp + fn)
            recall = 100 * (tp) / (tp + fn)
            specificity = 100 * (tn) / (tn + fp)
            fpr_r = fpr[fpr_ind[-1][0]]
            precision = 100 * (tp) / (tp + fp)
            f1_score = 2/((1/recall) + (1/precision))

            temp_dict = {"precision": precision,
                         "fpr": fpr_r,
                         "specificity": specificity,
                         "accuracy": accuracy,
                         "recall": recall,
                         "f1_score": f1_score,
                         "threshold": threshold}
        df_vals.append(temp_dict)
    return_df = pd.DataFrame(df_vals)
    return_df.set_index(pd.Index(fprs), inplace=True)
    return return_df.transpose()


def calculate_accuracy_at_thresholds(
    y_true,
    y_proba,
    thresholds=[
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9
    ]
):
    accuracies = []
    for threshold in thresholds:
        predictions = [1 if prob >= threshold else 0 for prob in y_proba]
        correct_predictions = sum(
            [1 for true, pred in zip(y_true, predictions) if true == pred])
        accuracy = correct_predictions / len(y_true)
        accuracies.append((threshold, accuracy))
    return accuracies
