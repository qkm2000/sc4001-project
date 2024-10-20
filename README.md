# SC4001 Project

## Project Overview
In this project, we aim to develop a robust model capable of classifying URLs as either malicious or non-malicious. With the rise of cyber threats, accurately identifying malicious URLs is crucial for enhancing online security and preventing phishing attacks.

## Approach
We leverage the ByT5 tokenizer and model, utilizing frozen weights to retain the knowledge from pre-training while focusing on our specific classification task. A classification head is added on top of the ByT5 architecture to predict the nature of the URLs. This approach allows us to harness the capabilities of deep learning and transfer learning effectively, improving our model's accuracy and performance while reducing computational requirements.

## Objectives
* To create a reliable model for identifying malicious URLs.
* To enhance cybersecurity measures through effective classification techniques.
* To contribute to the growing field of URL security by providing an open-source solution.

## Setup

### Python
We are using Python 3.12.7, but any Python 3.12 version should be fine

### Pip dependencies
Install all dependencies with `pip install -r requirements.txt`

### Torch
Install the [torch version](https://pytorch.org/get-started/locally/) that is compatible with your machine

### Environment variables
Create a .env file in the root of the project, and save the file in the following format:

```
RESULTS_PATH=".../results"
MODEL_PATH=".../model_files"
```

Replace the `...` with the path to the root of the project, so that all results and model files will be present in the project.

However, these will not be synced to git because they can be quite huge and annoying to deal with. Once we have the final model that we want, then we create a folder to save it in, and push that to git.