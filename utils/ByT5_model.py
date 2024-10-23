from transformers import T5ForConditionalGeneration
import torch.nn as nn
import torch


# Custom ByT5 Model
class ByT5ForClassification(nn.Module):
    def __init__(
        self,
        model_name="google/byt5-small",
        num_labels=2,
        freeze=True,
        encoder_unfrozen_layers=0,
        decoder_unfrozen_layers=0,
    ):
        """
        Initializes the ByT5ForClassification model.
        This model is a T5 model with a custom classification head
        for binary classification tasks.

        Args:
            model_name (str):
                The name or path of the pre-trained T5 model.
                A path can be provided instead, if a custom, pre-trained
                model is to be used instead of the base pre-trained model.
                Defaults to "google/byt5-small".
            num_labels (int, optional):
                The number of labels for classification. Defaults to 2.
            freeze (bool, optional):
                Whether to freeze the pre-trained model weights.
                If True, the pre-trained model weights will not be
                updated during training. Defaults to True.
            encoder_unfrozen_layers (int, optional):
                The number of encoder layers to leave unfrozen. Defaults to 0.
            decoder_unfrozen_layers (int, optional):
                The number of decoder layers to leave unfrozen. Defaults to 0.
        """
        self.num_labels = num_labels
        super(ByT5ForClassification, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        # Freeze all layers of the pre-trained T5 model
        if freeze:
            self._freeze_layers(
                encoder_unfrozen_layers,
                decoder_unfrozen_layers
            )

        # Define a simple fully connected classification
        # head with ReLU activations
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

        # Tie weights to handle shared memory issues with embeddings
        self.model.tie_weights()

        # Move to appropriate device (GPU if available, else CPU)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.to(self.device)

    def _freeze_layers(self, encoder_unfrozen_layers, decoder_unfrozen_layers):
        encoder_layers = self.model.encoder.block
        decoder_layers = self.model.decoder.block

        # Freeze encoder layers
        for layer in encoder_layers[:len(encoder_layers)-encoder_unfrozen_layers-1]:
            for param in layer.parameters():
                param.requires_grad = False

        # Freeze decoder layers
        for layer in decoder_layers[:len(decoder_layers)-decoder_unfrozen_layers-1]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Encode the inputs
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = encoder_outputs.last_hidden_state

        # Mean pooling
        pooled_output = mean_pooling(hidden_states, attention_mask)

        # Classifier - only the layers in the classifier will be trained
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1))

        return {'loss': loss, 'logits': logits}


def mean_pooling(hidden_states, attention_mask):
    # Expand attention mask to match the hidden states shape
    attention_mask_expanded = attention_mask.unsqueeze(
        -1).expand(hidden_states.size()).float()

    # Sum of hidden states multiplied by the attention mask
    sum_hidden_states = torch.sum(
        hidden_states * attention_mask_expanded, dim=1)

    # Divide by the sum of attention mask to get the mean, excluding padding
    sum_attention_mask = torch.clamp(
        attention_mask_expanded.sum(dim=1), min=1e-9)
    mean_hidden_states = sum_hidden_states / sum_attention_mask

    return mean_hidden_states
