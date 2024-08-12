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
from optimizer import AdamW, EAdam
from torch.cuda.amp import autocast, GradScaler
from penalty_function import ngram_penalty, diversity_penalty#, length_penalty
import time
import warnings
import socket
import os
#import nltk
#nltk.download('wordnet')

try:
    local_hostname = socket.gethostname()
except:
    local_hostname = None

DEV_MODE = False
if local_hostname == 'Corinna-PC' or local_hostname == "TABLET-TTS0K9R0": #Todo: Add also laptop
    DEV_MODE = True

TQDM_DISABLE = not DEV_MODE


warnings.filterwarnings("ignore", category=FutureWarning)

hyperparams = {
    'optimizer': [AdamW, EAdam],
    'learning_rate': [1e-5, 5e-5, 8e-5, 1e-4, 1e-6],
    'batch_size': [128, 64, 32],
    'dropout_rate': [0.3, 0.0],
    'patience': 3,
    'num_epochs': 100 if not DEV_MODE else 10,
    'accumulation_steps_base': 32,
    'max_length': 256,
    'alpha': 1e-2,
}  # Todo: make every function take values from here

def transform_data(dataset, max_length=256):
    """
    Turn the data to the format you want to use.
    Use AutoTokenizer to obtain encoding (input_ids and attention_mask).
    Tokenize the sentence pair in the following format:
    sentence_1 + SEP + sentence_1 segment location + SEP + paraphrase types.
    Return Data Loader.
    """
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
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False) #Todo: Shuffle train dataset? I think I do this when loading the dataset

    return dataloader



def train_model(model, train_data, val_data, device, tokenizer, learning_rate=hyperparams['learning_rate'], batch_size=hyperparams['batch_size'],
                patience=hyperparams['patience'], print_messages=DEV_MODE, alpha_ngram=1e-2, alpha_diversity=1e-2):
    accumulation_steps = int(batch_size / 32)
    if not DEV_MODE:
        torch.cuda.empty_cache()

    num_epochs = 100 if not DEV_MODE else 10
    num_training_steps = num_epochs * len(train_data) // accumulation_steps
    progress_bar = tqdm(range(num_training_steps), disable=TQDM_DISABLE)

    optimizer = EAdam(model.parameters(), lr=learning_rate)
    #optimizer = hyperparams['optimizer'] #Todo: After hyperparameter search, use this
    scaler = GradScaler()

    bleu_scores = []
    best_bleu_score = -10
    best_epoch = 0
    epochs_without_improvement = 0
    model_save_path = f"models/bart_generation_earlystopping.pt"

    # Start timing
    total_start_time = time.time()

    model.train()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # Start time for this epoch

        optimizer.zero_grad()
        for i, batch in enumerate(train_data):
            input_ids, attention_mask, labels = [tensor.to(device, non_blocking=True) for tensor in batch]
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / accumulation_steps

                # Compute the n-gram penalty
                with torch.no_grad():
                    predictions = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_length=50,
                        num_beams=5,
                        early_stopping=True,
                    )
                    penalty = alpha_ngram * ngram_penalty(predictions) + alpha_diversity * diversity_penalty(
                        predictions)

                # Add penalty to loss
                loss = loss + penalty

            scaler.scale(loss).backward()

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

        if val_data is not None:
            scores = evaluate_model(model, val_data, device, tokenizer, print_messages=print_messages)
            b = scores['bleu_score']
            bleu_scores.append(b)

            if b > best_bleu_score:
                best_bleu_score = scores['bleu_score']
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                # Save the best model
                torch.save(model.state_dict(), model_save_path)
                if print_messages:
                    print(f"Best model saved at epoch {epoch + 1} with BLEU score: {best_bleu_score:.3f}")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                if print_messages:
                    print(f"Early stopping triggered after {epoch + 1} epochs. \n")
                    print(f'Best BLEU score: {best_bleu_score} at epoch {best_epoch}. \n')
                    print(f"History: {bleu_scores}")
                break

    # End total timing
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    if print_messages:
        print(f"Total training time: {total_training_time:.2f} seconds.")

    # Load the best model before returning
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        if print_messages:
            print(f"Best model reloaded from {model_save_path}.")

    return model


def test_model(test_data, test_ids, device, model, tokenizer):
    model.eval()
    generated_sentences = []

    with torch.no_grad():
        for batch in test_data:
            input_ids, attention_mask, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

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

def evaluate_model(model, dataloader, device, tokenizer, print_messages=True):
    """
    You can use your train/validation set to evaluate models performance with the BLEU score.
    test_data is a DataLoader, where the column "sentence1" contains all input sentence and
    the column "sentence2" contains all target sentences
    """
    model.eval()
    bleu = BLEU()
    predictions = []
    references = []
    inputs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [tensor.to(device) for tensor in batch]

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

    # Calculate BLEU score
    bleu_score_reference = bleu.corpus_score(references, [predictions]).score
    # Penalize BLEU score if its to close to the input
    bleu_score_inputs = 100 - bleu.corpus_score(inputs, [predictions]).score
    penalized_bleu = bleu_score_reference * bleu_score_inputs / 52
    if print_messages:
        print(f"BLEU Score: {bleu_score_reference}", f"Negative BLEU Score with input: {bleu_score_inputs}")

        # Penalize BLEU and rescale it to 0-100
        # todo: If you perfectly predict all the targets, you should get an penalized BLEU score of around 52
        print(f"Penalized BLEU Score: {penalized_bleu}")

    #meteor_score = single_meteor_score(references, predictions)

    return {"bleu_score": penalized_bleu, "meteor_score": "meteor_score not computed"} #todo: put back meteor if want to use

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
    args = parser.parse_args()
    return args


def finetune_paraphrase_generation(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t")
    train_dataset = train_dataset if not DEV_MODE else train_dataset[:10]
    train_dataset_shuffled = train_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    val_dataset = train_dataset_shuffled.sample(frac=0.2, random_state=42)
    train_dataset = train_dataset_shuffled.drop(val_dataset.index)

    train_data = transform_data(train_dataset)
    val_data = transform_data(val_dataset)

    #test_dataset = pd.read_csv("data/etpc-paraphrase-generation-test-student.csv", sep="\t")#[:10]
    #test_dataset = test_dataset if not DEV_MODE else test_dataset[:10] #todo: put back at the end

    print(f"Loaded {len(train_dataset)} training samples.")

    best_bleu = 0
    best_lr = 0
    best_dropout = 42
    best_batchsize = 0

    hyperparameter_grid = {
        'learning_rate': [1e-5, 5e-5, 8e-5, 1e-4, 1e-6],
        'batch_size': [128, 64, 32],  #, 128], This gives memory issues
        'dropout_rate': [0.3, 0.0],
        'alpha': [],#todo: add alpha from penalty function
    }

    hyperparameter_grid = {
        'learning_rate': [1e-5],# 5e-5, 8e-5, 1e-4, 1e-6],
        'batch_size': [128],
        'dropout_rate': [0.0]#, 0.1, 0.0]
    }

    for b in hyperparameter_grid['batch_size']:
        for dropout in hyperparameter_grid['dropout_rate']:
            for lr in hyperparameter_grid['learning_rate']:
                #if dropout > 0 and b > 32:
                 #   continue
                if dropout > 0:
                    config = BartConfig.from_pretrained("facebook/bart-large")
                    config.attention_dropout = dropout
                    config.activation_dropout = dropout
                    config.dropout = dropout
                    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", config=config,
                                                                         local_files_only=True)
                else:
                    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large",
                                                                         local_files_only=True)
                model.to(device)
                if DEV_MODE:
                    scores_before_training = evaluate_model(model, val_data, device, tokenizer)
                    bleu_score_before_training, _ = scores_before_training.values()
                model = train_model(model, train_data, val_data, device, tokenizer, learning_rate=lr, batch_size=b, patience=hyperparams['patience'], print_messages=True, alpha_ngram=0.0, alpha_diversity=0.0) #todo: set print_messages to DEV_MODE again
                scores = evaluate_model(model, val_data, device, tokenizer)
                bleu_score, _ = scores.values()
                print(f"Results for learning rate {lr}, batch_size: {b}, dropout rate: {dropout}:")
                print(f"The penalized BLEU-score of the model is: {bleu_score:.3f}")
                print("\n")
                #print(f"The METEOR-score of the model is: {meteor_score:.3f}")
                #print(f"Without training: \n BLEU: {bleu_score_before_training:.3f}")# \n METEOR: {meteor_score_before_training}")
                if bleu_score > best_bleu:
                    best_bleu = bleu_score
                    best_lr = lr
                    best_batchsize = b
                    best_dropout = dropout
                # Clear GPU memory
                #del model #todo: put back
                #torch.cuda.empty_cache() #Todo: will it also delete the best parameters? and put it back
    print(f"Best params: \n LR: {best_lr} \n batch size: {best_batchsize} \n dropout: {best_dropout}. Best BLEU: {best_bleu}.")

    #paraphrases = generate_paraphrases(model, val_data, device, tokenizer)
    #paraphrases.to_csv("predictions/bart/generation_predict.csv", index=False, sep="\t")


    #test_ids = test_dataset["id"]
    #test_results = test_model(test_data, test_ids, device, model, tokenizer)
    #if not DEV_MODE:
    #    test_results.to_csv(
    #        "predictions/bart/etpc-paraphrase-generation-test-output.csv", index=False, sep="\t"
    #    )

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_generation(args)
