# kar min

## v1

Data:\
Model was trained on a 1% sample of malicious_phish.csv

Classifier head:
```
self.classifier = nn.Sequential(
    nn.Linear(self.model.config.d_model, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, num_labels),
)
```

Hyper Params:
```
MAX_LENGTH = 50
BATCH_SIZE = 128
training_args = TrainingArguments(
    # saving results/checkpoints
    output_dir=RESULTS_PATH,
    save_safetensors=False,

    # evaluation
    eval_strategy="epoch",
    eval_steps=1,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    
    # saving
    save_strategy="epoch",
    save_steps=1,
    save_total_limit=3,

    # hyperparameters
    learning_rate=0.005,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=100,
    weight_decay=0.01,
)
```

Results:
```
Threshold: 0.10, Accuracy: 0.80
Threshold: 0.20, Accuracy: 0.88
Threshold: 0.30, Accuracy: 0.91
Threshold: 0.40, Accuracy: 0.91
Threshold: 0.50, Accuracy: 0.92
Threshold: 0.60, Accuracy: 0.92
Threshold: 0.70, Accuracy: 0.90
Threshold: 0.80, Accuracy: 0.88
Threshold: 0.90, Accuracy: 0.86
```

This model performs well on the malicious_phish dataset, but it is also what it is trained on. This model is outdated because we have improved the classification head and the weights learned from this model are no longer applicable. This was more of a feasibility test than anything

## v2

Data:\
Model was trained on a 3% sample of malicious_phish.csv

Classifier head:
```
self.classifier = nn.Sequential(
    nn.Linear(self.model.config.d_model, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, num_labels),
)
```

Hyper Params:
```
MAX_LENGTH = 100
BATCH_SIZE = 128
training_args = TrainingArguments(
    # saving results/checkpoints
    output_dir=RESULTS_PATH,
    save_safetensors=False,

    # evaluation
    eval_strategy="epoch",
    eval_steps=1,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    
    # saving
    save_strategy="epoch",
    save_steps=1,
    save_total_limit=3,

    # hyperparameters
    learning_rate=0.005,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=100,
    weight_decay=0.01,
)
```

Results:
```
Threshold: 0.10, Accuracy: 0.81
Threshold: 0.20, Accuracy: 0.87
Threshold: 0.30, Accuracy: 0.89
Threshold: 0.40, Accuracy: 0.90
Threshold: 0.50, Accuracy: 0.91
Threshold: 0.60, Accuracy: 0.92
Threshold: 0.70, Accuracy: 0.92
Threshold: 0.80, Accuracy: 0.93
Threshold: 0.90, Accuracy: 0.93
```

This model performs well when tested on malicious_phish.csv, but is very terrible on other datasets. maybe the data is old, outdated, or too different? could it be synthetic data?

## v3

Data:\
Model was trained on a 10% sample of PhiUSIIL_Phishing_URL_Dataset.csv

Classifier head:
```
same as v2
```

Hyper Params:
```
same as v2
```

Results:
```
Threshold: 0.10, Accuracy: 0.91
Threshold: 0.20, Accuracy: 0.96
Threshold: 0.30, Accuracy: 0.97
Threshold: 0.40, Accuracy: 0.96
Threshold: 0.50, Accuracy: 0.96
Threshold: 0.60, Accuracy: 0.96
Threshold: 0.70, Accuracy: 0.95
Threshold: 0.80, Accuracy: 0.95
Threshold: 0.90, Accuracy: 0.93
```

This model works decently when tested on the URL dataset too, but accuracy is generally around 50%. This likely means that the data is also rather different, and we will require more EDA to gain more insight into the reasons as to why it performs so poorly on other datasets. this model performs especially badly on the malicious_phish dataset

## v4

Data:\
Model was trained on a 4% sample of URL dataset.csv

Classifier head:
```
same as v2
```

Hyper Params:
```
same as v2
```

Results:
```
Threshold: 0.10, Accuracy: 0.95
Threshold: 0.20, Accuracy: 0.96
Threshold: 0.30, Accuracy: 0.96
Threshold: 0.40, Accuracy: 0.96
Threshold: 0.50, Accuracy: 0.96
Threshold: 0.60, Accuracy: 0.96
Threshold: 0.70, Accuracy: 0.95
Threshold: 0.80, Accuracy: 0.95
Threshold: 0.90, Accuracy: 0.94
```

This model works decently when tested on the phiusiil dataset too, but accuracy is generally around 50%. This likely means that the data is also rather different, and we will require more EDA to gain more insight into the reasons as to why it performs so poorly on other datasets. this model performs especially badly on the malicious_phish dataset

## v5

Data:\
Model was trained on a 2% sample of all the datasets we have combined together

Classifier head:
```
same as v2
```

Hyper Params:
```
same as v2
```

Results:
```
Threshold: 0.10, Accuracy: 0.78
Threshold: 0.20, Accuracy: 0.85
Threshold: 0.30, Accuracy: 0.88
Threshold: 0.40, Accuracy: 0.90
Threshold: 0.50, Accuracy: 0.91
Threshold: 0.60, Accuracy: 0.91
Threshold: 0.70, Accuracy: 0.90
Threshold: 0.80, Accuracy: 0.89
Threshold: 0.90, Accuracy: 0.86
```

This model performs well on all the different datasets.
The malicious_phish dataset does not include "http://" or "https://" in the urls, which was probably what was messing up the results in the previous models. With the inclusion of these pieces of data along with data that includes the header, the model is now able to generalise very well across all datasets.

## v6

Data:\
Model was trained on a 2% sample of all the datasets we have combined together

Classifier head:
```
same as v2
last 2 layers of the decoder were unfrozen and trained
```

Hyper Params:
```
MAX_LENGTH = 100
BATCH_SIZE = 32
training_args = TrainingArguments(
    # saving results/checkpoints
    output_dir=RESULTS_PATH,
    save_safetensors=False,

    # evaluation
    eval_strategy="epoch",
    eval_steps=1,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    
    # saving
    save_strategy="epoch",
    save_steps=1,
    save_total_limit=3,

    # hyperparameters
    learning_rate=0.005,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=100,
    weight_decay=0.01,
)
```

Results:
```
Threshold: 0.10, Accuracy: 0.93
Threshold: 0.20, Accuracy: 0.94
Threshold: 0.30, Accuracy: 0.94
Threshold: 0.40, Accuracy: 0.94
Threshold: 0.50, Accuracy: 0.94
Threshold: 0.60, Accuracy: 0.95
Threshold: 0.70, Accuracy: 0.96
Threshold: 0.80, Accuracy: 0.96
Threshold: 0.90, Accuracy: 0.93
```

This model is by far the best performing model (0.99 accuracy on phiusiil and url_dataset, and about 0.90 on malicious_phish). This is most probably because the last 2 layers of the decoder were trained along with the classifier head, allowing the model to learn more about the URLs and their relations to whether they were malicious or not
