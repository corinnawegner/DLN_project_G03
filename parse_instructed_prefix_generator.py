from transformers import BartPretrainedModel, BartModel
import argparse
import random
import numpy as np
import pandas as pd
import torch
from sacrebleu.metrics import BLEU
#from nltk.translate.meteor_score import single_meteor_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig
from optimizer import AdamW
from torch.cuda.amp import autocast, GradScaler
from penalty_function import ngram_penalty, diversity_penalty#, length_penalty
import time
import warnings
import socket
import os
import time
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn.functional as F

try:
    local_hostname = socket.gethostname()
except:
    local_hostname = None

DEV_MODE = False
if local_hostname == 'Corinna-PC' or local_hostname == "TABLET-TTS0K9R0": #Todo: Add also laptop
    DEV_MODE = True

TQDM_DISABLE = not DEV_MODE

warnings.filterwarnings("ignore", category=FutureWarning)

class PIPBartForConditionalGeneration(BartPretrainedModel):
    def __init__(self, config, pip_direct=False, pip_indirect=False):
        super().__init__(config)
        self.model = BartModel(config)
        self.lm_head = torch.nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.pip_direct = pip_direct
        self.pip_indirect = pip_indirect
        self.prefix_length = config.prefix_length if hasattr(config, 'prefix_length') else 10
        self.prefix_embedding = torch.nn.Embedding(self.prefix_length, config.d_model)
        self.init_weights()

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        if self.pip_direct:
            prefix_tokens = torch.arange(self.prefix_length, dtype=torch.long, device=input_ids.device)
            prefix_embeddings = self.prefix_embedding(prefix_tokens).unsqueeze(0).expand(input_ids.size(0), -1, -1)
            inputs_embeds = torch.cat([prefix_embeddings, self.model.embed_tokens(input_ids)], dim=1)
        else:
            inputs_embeds = self.model.embed_tokens(input_ids)

        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)

        lm_logits = self.lm_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            if self.pip_indirect:
                loss += self.compute_parse_encoding_loss(prefix_embeddings)
        return loss, lm_logits



def train_model_with_pip(model, train_data, val_data, device, tokenizer, learning_rate, batch_size, patience,
                         pip_direct=False, pip_indirect=False, print_messages=True, model_save_path='best_model.pth', alpha_ngram=0, alpha_diversity=0):

    # Set optimizer and gradient scaler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    # Set number of epochs
    num_epochs = 100 if not DEV_MODE else 10
    best_bleu_score = -10
    best_epoch = 0
    epochs_without_improvement = 0

    # Adjust batch size for gradient accumulation
    accumulation_steps = max(1, int(batch_size / 32))

    # Empty GPU cache if not in dev mode
    if not DEV_MODE:
        torch.cuda.empty_cache()

    # Set up progress bar
    num_training_steps = num_epochs * len(train_data) // accumulation_steps
    progress_bar = tqdm(range(num_training_steps), disable=TQDM_DISABLE)

    # Start timing
    total_start_time = time.time()

    model.train()
    for epoch in range(num_epochs):
        model.zero_grad()
        epoch_start_time = time.time()

        for i, batch in enumerate(train_data):
            input_ids, attention_mask, labels = [tensor.to(device, non_blocking=True) for tensor in batch]

            # Mixed precision training
            with autocast():
                loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                pip_direct=pip_direct, pip_indirect=pip_indirect)
                loss = loss / accumulation_steps  # Scale loss for accumulation steps

                # Compute the n-gram and diversity penalties
                with torch.no_grad():
                    predictions = model.generate(input_ids, attention_mask=attention_mask, max_length=50, num_beams=5, early_stopping=True)
                    if alpha_ngram != 0 or alpha_diversity != 0:
                        penalty = alpha_ngram * ngram_penalty(predictions) + alpha_diversity * diversity_penalty(predictions)
                        loss += penalty  # Add penalty to loss

                # Backpropagation with scaled loss
                scaler.scale(loss).backward()

            # Perform optimizer step after accumulating gradients
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                progress_bar.update(1)

        # End time for this epoch
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        if print_messages:
            print(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_duration:.2f} seconds.")

        # Validation and early stopping
        if val_data is not None:
            scores = evaluate_model(model, val_data, device, tokenizer, print_messages=print_messages)
            bleu_score = scores['bleu_score']

            if bleu_score > best_bleu_score:
                best_bleu_score = bleu_score
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                # Save the best model
                torch.save(model.state_dict(), model_save_path)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                if print_messages:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    print(f'Best BLEU score: {best_bleu_score} at epoch {best_epoch}.')
                    print(f"History: {bleu_scores}")
                break

    # End total timing
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    if print_messages:
        print(f"Total training time: {total_training_time:.2f} seconds.")

    # Load the best model before returning
    torch.cuda.empty_cache()
    model.load_state_dict(torch.load(model_save_path))
    model = model.to(device)

    return model


config = BartConfig.from_pretrained("facebook/bart-large")
config.prefix_length = 10  # Example value for prefix length
model = PIPBartForConditionalGeneration.from_pretrained("facebook/bart-large", config=config, pip_direct=True, pip_indirect=True)

