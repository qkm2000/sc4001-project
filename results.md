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
