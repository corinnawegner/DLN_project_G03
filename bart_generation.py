import argparse
import random
import numpy as np
import pandas as pd
import torch
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig, AdamW
from torch.cuda.amp import autocast, GradScaler
from penalty_function import ngram_penalty, diversity_penalty
import time
import warnings
import socket
import os

try:
    local_hostname = socket.gethostname()
except:
    local_hostname = None

DEV_MODE = False
if local_hostname in ['Corinna-PC', "TABLET-TTS0K9R0"]:
    DEV_MODE = True

TQDM_DISABLE = not DEV_MODE

warnings.filterwarnings("ignore", category=FutureWarning)

r = random.randint(10000, 99999)
model_save_path = f"models/bart_generation_prefix_{r}.pt"

hyperparams = {
    'optimizer': AdamW,
    'learning_rate': 5e-5,
    'batch_size': 32,
    'dropout_rate': 0.0,
    'patience': 3,
    'num_epochs': 100 if not DEV_MODE else 10,
    'alpha': 1e-2,
}


class PrefixTuningBart(BartForConditionalGeneration):
    def __init__(self, config: BartConfig, prefix_length: int = 5):
        super().__init__(config)
        self.prefix_length = prefix_length
        self.prefix_embedding = torch.nn.Embedding(prefix_length, config.d_model)

    def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None, **kwargs):
        print(f'input_ids in the class PrefixTuningBart: {input_ids}')
        print(f'input_embeds in the class PrefixTuningBart: {inputs_embeds}')

        if input_ids is not None:
            batch_size = input_ids.shape[0]
            device = input_ids.device
        elif inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
            device = inputs_embeds.device
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        print("first forward done")

        # Create prefix tokens and embeddings
        prefix_tokens = torch.arange(self.prefix_length, dtype=torch.long, device=device)
        prefix_tokens = prefix_tokens.unsqueeze(0).expand(batch_size, -1)
        prefix_embeddings = self.prefix_embedding(prefix_tokens)

        if input_ids is not None:
            inputs_embeds = self.model.shared(input_ids)
        else:
            inputs_embeds = inputs_embeds

        inputs_embeds = torch.cat((prefix_embeddings, inputs_embeds), dim=1)

        if attention_mask is not None:
            prefix_attention_mask = torch.ones(batch_size, self.prefix_length, device=device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        # Call the forward method of the parent class
        outputs = super().forward(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        return outputs


def transform_data(dataset, max_length=256):
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

    input_ids = []
    attention_masks = []
    labels = []

    for _, row in tqdm(dataset.iterrows(), total=len(dataset), disable=TQDM_DISABLE):
        sentence_1 = row['sentence1']
        segment_location_1 = row['sentence1_segment_location']
        paraphrase_type = row['paraphrase_types']
        combined_input = f"{sentence_1} [SEP] {segment_location_1} [SEP] {paraphrase_type}"

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

    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    return dataloader

def train_model(model, train_data, val_data, device, tokenizer, learning_rate=hyperparams['learning_rate'],
                batch_size=hyperparams['batch_size'], patience=hyperparams['patience'],
                print_messages=DEV_MODE, alpha_ngram=1e-2, alpha_diversity=1e-2, prefix_length=5):
    accumulation_steps = int(batch_size / 32)
    if not DEV_MODE:
        torch.cuda.empty_cache()

    num_epochs = 100 if not DEV_MODE else 10
    num_training_steps = num_epochs * len(train_data) // accumulation_steps
    progress_bar = tqdm(range(num_training_steps), disable=TQDM_DISABLE)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    bleu_scores = []
    best_bleu_score = -10
    best_epoch = 0
    epochs_without_improvement = 0

    total_start_time = time.time()

    model.train()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        optimizer.zero_grad()
        for i, batch in enumerate(train_data):
            input_ids, attention_mask, labels = [tensor.to(device, non_blocking=True) for tensor in batch]
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / accumulation_steps
                with torch.no_grad():
                    print(f"input_ids just before predictions: {input_ids}")
                    predictions = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=50,
                        num_beams=5,
                        early_stopping=True,
                    )
                    #if alpha_ngram != 0 or alpha_diversity != 0:
                     #   penalty = alpha_ngram * ngram_penalty(predictions) + alpha_diversity * diversity_penalty(predictions)
                       # loss = loss + penalty

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                progress_bar.update(1)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        if print_messages:
            print(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_duration:.2f} seconds.")

        if val_data is not None:
            scores = evaluate_model(model, val_data, device, tokenizer, print_messages=print_messages)
            b = scores['bleu_score']
            bleu_scores.append(b)

            if b > best_bleu_score:
                best_bleu_score = scores['bleu_score']
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                if print_messages:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    print(f'Best BLEU score: {best_bleu_score} at epoch {best_epoch}.')
                    print(f"History: {bleu_scores}")
                break

    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    if print_messages:
        print(f"Total training time: {total_training_time:.2f} seconds.")

    del model
    torch.cuda.empty_cache()
    model = PrefixTuningBart.from_pretrained("facebook/bart-large", local_files_only=True, prefix_length=prefix_length)
    model.load_state_dict(torch.load(model_save_path))
    model = model.to(device)
    return model


def test_model(test_data, test_ids, device, model, tokenizer):
    model.eval()
    generated_sentences = []

    with torch.no_grad():
        for batch in test_data:
            input_ids, attention_mask, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

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

            generated_sentences.extend(pred_text)

    results = pd.DataFrame({
        'id': test_ids,
        'Generated_sentence2': generated_sentences
    })

    return results

def generate_paraphrases(model, dataloader, device, tokenizer):
    model.eval()
    predictions = []
    references = []
    inputs = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [tensor.to(device) for tensor in batch]

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

def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--prefix_length", type=int, default=5)
    parser.add_argument("--train_data_path", type=str, default="data/etpc-paraphrase-train.csv")
    parser.add_argument("--val_data_path", type=str, default="val.csv")
    parser.add_argument("--test_data_path", type=str, default="data/etpc-paraphrase-detection-test-student.csv")
    parser.add_argument("--output_path", type=str, default="results.csv")
    parser.add_argument("--ngram_alpha", type=float, default=1e-2)
    parser.add_argument("--diversity_alpha", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--dropout_rate", type=float, default=0.1)

    args = parser.parse_args()
    return args


def evaluate_model(model, dataloader, device, tokenizer, print_messages=False):
    paraphrases = generate_paraphrases(model, dataloader, device, tokenizer)
    references = paraphrases['references'].tolist()
    predictions = paraphrases['predictions'].tolist()

    metric = BLEU()
    bleu_score = metric.corpus_score(predictions, [references]).score

    if print_messages:
        print(f'BLEU Score: {bleu_score}')

    return {'bleu_score': bleu_score}


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t")
    train_dataset = train_dataset if not DEV_MODE else train_dataset[:10]
    train_dataset_shuffled = train_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    val_dataset = train_dataset_shuffled.sample(frac=0.2, random_state=42)
    train_dataset = train_dataset_shuffled.drop(val_dataset.index)

    train_dataloader = transform_data(train_dataset)
    val_dataloader = transform_data(val_dataset)
    #test_dataloader = transform_data(test_dar)

    config = BartConfig.from_pretrained("facebook/bart-large")
    config.dropout = hyperparams['dropout_rate']

    model = PrefixTuningBart.from_pretrained("facebook/bart-large", config=config, prefix_length=args.prefix_length)
    model = model.to(device)

    # Train and validate the model
    trained_model = train_model(model, train_dataloader, val_dataloader, device, tokenizer,
                                learning_rate=args.learning_rate, batch_size=args.batch_size,
                                patience=args.patience, alpha_ngram=args.ngram_alpha,
                                alpha_diversity=args.diversity_alpha, prefix_length=args.prefix_length)

    # Test the model
    #test_ids = test_df['id'].tolist()
    #results = test_model(test_dataloader, test_ids, device, trained_model, tokenizer)

    # Save the results
    #results.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    main()
