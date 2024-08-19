from adjacency_matrix_from_dependence_tree import adjacency_matrix
import torch
import argparse
import random
import numpy as np
import pandas as pd
import spacy
import torch
from sacrebleu.metrics import BLEU
#from nltk.translate.meteor_score import single_meteor_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration
from optimizer import AdamW
from torch.cuda.amp import autocast, GradScaler
#from penalty_function import ngram_penalty, diversity_penalty
import time
import warnings
import socket
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR, MultiStepLR
from gcn import GraphConvolution


try:
    local_hostname = socket.gethostname()
except:
    local_hostname = None

DEV_MODE = False
if local_hostname in ['Corinna-PC', "TABLET-TTS0K9R0", "DESKTOP-3D9LKBO"]:
    DEV_MODE = True

TQDM_DISABLE = not DEV_MODE

hyperparams = {
    'optimizer': AdamW,
    'learning_rate': 1e-5,
    'batch_size': 64,
    'dropout_rate': 0.1,
    'patience': 3,
    'num_epochs': 100 if not DEV_MODE else 2,
    'alpha': 0.0,
    'scheduler': None,
    'POS_NER_tagging': False
}  # Todo: make every function take values from here

r = random.randint(10000, 99999)

from transformers.modeling_outputs import BaseModelOutput

class CustomEncoderOutput(BaseModelOutput):
    def __init__(self, last_hidden_state, **kwargs):
        super().__init__(last_hidden_state=last_hidden_state, **kwargs)

def generate(self, input_ids, attention_mask, adj_matrices, **kwargs):
    # Pass inputs through BART's encoder
    encoder_outputs = self.bart_model.model.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    # Apply the GCN layer to the encoder's output
    gcn_output = self.gcn_layer(encoder_outputs.last_hidden_state, adj_matrices)

    # Wrap the GCN output in a suitable format
    # Note: Ensure gcn_output is correctly formatted
    encoder_outputs = CustomEncoderOutput(last_hidden_state=gcn_output)

    # Generate using BART's decoder with the GCN-enhanced encoder output
    generated_ids = self.bart_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        encoder_outputs=encoder_outputs,  # Pass the wrapped encoder outputs
        **kwargs
    )

    return generated_ids


class BartWithGCN(torch.nn.Module):
    def __init__(self, bart_model, gcn_layer):
        super(BartWithGCN, self).__init__()
        self.bart_model = bart_model
        self.gcn_layer = gcn_layer

    def forward(self, input_ids, attention_mask, labels=None, adj_matrices=None):
        # Pass inputs through BART's encoder
        encoder_outputs = self.bart_model.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Apply the GCN layer to the encoder's output
        gcn_output = self.gcn_layer(encoder_outputs.last_hidden_state, adj_matrices)

        # Pass the GCN-enhanced encoder output to the BART decoder
        decoder_outputs = self.bart_model(
            input_ids=None,
            attention_mask=attention_mask,
            encoder_outputs=(gcn_output,),
            labels=labels
        )

        return decoder_outputs

    def generate(self, input_ids, attention_mask, adj_matrices, **kwargs):
        # Pass inputs through BART's encoder
        encoder_outputs = self.bart_model.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Apply the GCN layer to the encoder's output
        gcn_output = self.gcn_layer(encoder_outputs.last_hidden_state, adj_matrices)

        # Wrap the GCN output in a suitable format
        encoder_outputs = CustomEncoderOutput(last_hidden_state=gcn_output)

        # Generate using BART's decoder with the GCN-enhanced encoder output
        generated_ids = self.bart_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,  # Pass the wrapped encoder outputs
            **kwargs
        )

        return generated_ids

def train_gcn_bart_model(model, train_data, val_data, device, tokenizer, learning_rate=hyperparams['learning_rate'],
                batch_size=hyperparams['batch_size'], patience=hyperparams['patience'],
                print_messages=DEV_MODE, alpha_ngram=0.0, alpha_diversity=0.0,
                use_scheduler=None, model_save_path = f"models/bart_generation_gcn_{r}.pt"):

    accumulation_steps = int(batch_size / 32)
    if not DEV_MODE:
        torch.cuda.empty_cache()
    if print_messages:
        print("Starting training")
    num_epochs = hyperparams['num_epochs']
    num_training_steps = num_epochs * len(train_data) // accumulation_steps
    progress_bar = tqdm(range(num_training_steps), disable=TQDM_DISABLE)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    if use_scheduler is None:
        scheduler = None
    elif use_scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5)
    elif use_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif use_scheduler == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
                                                        steps_per_epoch=len(train_data), epochs=num_epochs)
    elif use_scheduler == 'MultiStepLR':
        milestones = [int(0.5 * num_epochs), int(0.75 * num_epochs)]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    else:
        try:
            scheduler = use_scheduler
        except:
            print("unknown scheduler, provide full information")

    scaler = GradScaler()

    bleu_scores = []
    best_bleu_score = -10
    best_epoch = 0
    epochs_without_improvement = 0

    # Start timing
    total_start_time = time.time()

    model.train()
    for epoch in tqdm(range(num_epochs), disable = TQDM_DISABLE):
        if print_messages:
            print(f"Epoch {epoch}")
        epoch_start_time = time.time()  # Start time for this epoch
        optimizer.zero_grad()
        for i, batch in enumerate(train_data):
            input_ids, attention_mask, labels, adj_matrices = [tensor.to(device, non_blocking=True) for tensor in batch]
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                adj_matrices=adj_matrices)
                loss = outputs.loss / accumulation_steps

                if alpha_ngram != 0 or alpha_diversity != 0:
                    predictions = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        adj_matrices=adj_matrices,
                        max_length=50,
                        num_beams=5,
                        early_stopping=True,
                    )

                    pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
                    input_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

                    #penalty = alpha_ngram * ngram_penalty(pred_texts,input_texts) + alpha_diversity * diversity_penalty(pred_texts, input_texts)

                    #loss = loss + penalty
            if print_messages:
                print("Loss computed")
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if use_scheduler is not None:
                    scheduler.step()
                progress_bar.update(1)

        # Evaluate the model on validation data and perform early stopping
        # (Same as in your original code)

    # Load the best model before returning
    #model.load_state_dict(torch.load(model_save_path)) #Todo: reimplement earlystopping
    #model = model.to(device)
    return model


def pad_adjacency_matrix(matrix, max_dim):
    # Create a new 3D matrix with the shape (num_labels, max_dim, max_dim) filled with zeros
    padded_matrix = np.zeros((matrix.shape[0], max_dim, max_dim), dtype=np.float32)

    # Copy the original matrix into the padded matrix
    for i in range(matrix.shape[0]):
        padded_matrix[i, :matrix.shape[1], :matrix.shape[2]] = matrix[i]

    return padded_matrix

def transform_data_gcn(dataset, max_length=256):
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    input_ids = []
    attention_masks = []
    labels = []
    adj_matrices = []

    # Determine the maximum dimension for adjacency matrices
    max_dim = max_length
    adj_matrices_list = []

    for _, row in tqdm(dataset.iterrows(), total=len(dataset), disable=TQDM_DISABLE):
        sentence_1 = row['sentence1']
        adj_matrix = adjacency_matrix(sentence_1)
        max_dim = max(max_dim, adj_matrix.shape[1])
        adj_matrices_list.append(adj_matrix)

    # Pad all adjacency matrices to the maximum dimension
    for adj_matrix in adj_matrices_list:
        padded_matrix = pad_adjacency_matrix(adj_matrix, max_dim)
        adj_matrices.append(torch.tensor(padded_matrix, dtype=torch.float))

    for idx, row in tqdm(dataset.iterrows(), total=len(dataset), disable=TQDM_DISABLE):
        sentence_1 = row['sentence1']
        combined_input = f"{sentence_1} [SEP] {row['sentence1_segment_location']} [SEP] {row['paraphrase_types']}"

        encoding = tokenizer(
            combined_input,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        if 'sentence2' in row:
            sentence_2 = row['sentence2']
            label_encoding = tokenizer(
                sentence_2,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            labels.append(label_encoding['input_ids'].squeeze())
        else:
            labels.append(torch.zeros(max_length, dtype=torch.long))

        input_ids.append(encoding['input_ids'].squeeze())
        attention_masks.append(encoding['attention_mask'].squeeze())

    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    labels = torch.stack(labels)
    adj_matrices = torch.stack(adj_matrices)  # Shape: [batch_size, num_labels, max_dim, max_dim]

    dataset = TensorDataset(input_ids, attention_masks, labels, adj_matrices)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    for batch in dataloader:
        input_ids, attention_masks, labels, adj_matrices = batch
        print(f"input_ids shape: {input_ids.shape}")
        print(f"adj_matrices shape: {adj_matrices.shape}")
        #break  # Check the shape for one batch to verify

    return dataloader



def finetune_paraphrase_generation_gcn(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)

    gcn_layer = GraphConvolution(
        in_features=1024,  # BART-large hidden size
        out_features=1024, # todo: I changed it from 1024 to 256 because of the transform_data dimension
        num_labels=44  # Number of labels in your adjacency matrices
    ).to(device)

    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", local_files_only=True).to(device)
    model = BartWithGCN(bart_model, gcn_layer)

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t")
    train_dataset = train_dataset if not DEV_MODE else train_dataset[:10]
    train_dataset_shuffled = train_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    val_dataset = train_dataset_shuffled.sample(frac=0.2, random_state=42)
    train_dataset = train_dataset_shuffled.drop(val_dataset.index)

    train_data = transform_data_gcn(train_dataset)
    val_data = transform_data_gcn(val_dataset)

    model = train_gcn_bart_model(model, train_data, val_data, device, tokenizer,
                        learning_rate=hyperparams['learning_rate'], batch_size=hyperparams['batch_size'],
                        patience=hyperparams['patience'], print_messages=True,
                        alpha_ngram=hyperparams['alpha'], alpha_diversity=hyperparams['alpha'])  # Use the best alpha values found

    evaluate_model_gcn(model, val_data, device, tokenizer)

from sacrebleu.metrics import BLEU

def generate_paraphrases_gcn(model, dataloader, device, tokenizer):
    model.eval()
    predictions = []
    references = []
    inputs = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels, _ = [tensor.to(device) for tensor in batch]

            # Generate paraphrases
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=5,
                early_stopping=True,
            )

            pred_text = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in outputs
            ]
            references.extend([
                tokenizer.decode(label, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for label in labels
            ])
            inputs.extend([
                tokenizer.decode(input_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for input_id in input_ids
            ])
            predictions.extend(pred_text)
            paraphrases = pd.DataFrame({
                'references': references,
                'predictions': predictions
            })
            return paraphrases
def evaluate_model_gcn(model, dataloader, device, tokenizer, print_messages=True):
    """
    Evaluates the model performance using BLEU score.

    Args:
        model: The model to evaluate.
        dataloader: DataLoader containing input data, target sentences, and adjacency matrices.
        device: Device to run the model on (CPU or GPU).
        tokenizer: Tokenizer used to decode the token IDs.
        print_messages: Whether to print evaluation messages.

    Returns:
        A dictionary with BLEU score and optionally METEOR score.
    """
    model.eval()
    bleu = BLEU()
    predictions = []
    references = []
    inputs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=not print_messages):
            input_ids, attention_mask, labels, adj_matrices = [tensor.to(device) for tensor in batch]

            print(f"input_ids type: {input_ids.dtype}")
            print(f"attention_mask type: {attention_mask.dtype}")
            print(f"adj_matrices type: {adj_matrices.dtype}")

            # Generate paraphrases
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                adj_matrices=adj_matrices,  # Include adj_matrices here
                max_length=50,
                num_beams=5,
                early_stopping=True,
            )

            # Decode predictions and references
            pred_texts = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in outputs
            ]
            ref_texts = [
                tokenizer.decode(label, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for label in labels
            ]
            input_texts = [
                tokenizer.decode(input_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for input_id in input_ids
            ]

            predictions.extend(pred_texts)
            references.extend(ref_texts)
            inputs.extend(input_texts)

    # Compute BLEU scores
    # BLEU score for references
    bleu_score_reference = bleu.corpus_score(predictions, references).score

    # BLEU score for inputs
    bleu_score_inputs = 100 - bleu.corpus_score(predictions, [[inp] for inp in inputs]).score

    # Penalized BLEU score
    penalized_bleu = bleu_score_reference * bleu_score_inputs / 52

    if print_messages:
        print(f"BLEU Score: {bleu_score_reference:.2f}")
        print(f"Negative BLEU Score with input: {bleu_score_inputs:.2f}")
        print(f"Penalized BLEU Score: {penalized_bleu:.2f}")

    # Placeholder for METEOR score if needed
    meteor_score = "METEOR score not computed"

    return {"bleu_score": penalized_bleu, "meteor_score": meteor_score}
