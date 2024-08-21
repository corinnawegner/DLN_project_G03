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
from penalty_function import ngram_penalty, diversity_penalty
import time
import warnings
import socket
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
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

r = random.randint(10000, 99999)
model_save_path = f"models/bart_generation_earlystopping_{r}.pt"

hyperparams = {
    'optimizer': AdamW,
    'learning_rate': 1e-5,
    'batch_size': 64,
    'dropout_rate': 0.1,
    'patience': 3,
    'num_epochs': 100 if not DEV_MODE else 10,
    'alpha': 0.0,
    'scheduler': "CosineAnnealingLR",
    'POS_NER_tagging': True
}  # Todo: make every function take values from here

if hyperparams['POS_NER_tagging'] == True:
    nlp = spacy.load("en_core_web_sm")

def perform_pos_ner(text):
    """
    Perform POS tagging and NER on the given text.
    """
    doc = nlp(text)
    pos_tags = [(token.text, token.pos_) for token in doc]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return pos_tags, entities


def transform_data(dataset, max_length=256, use_tagging=hyperparams['POS_NER_tagging']):
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

    SEP = tokenizer.sep_token

    for _, row in tqdm(dataset.iterrows(), total=len(dataset), disable=TQDM_DISABLE):
        sentence_1 = row['sentence1']
        segment_location_1 = row['sentence1_segment_location']
        paraphrase_type = row['paraphrase_types']

        if use_tagging:
            pos_tags, entities = perform_pos_ner(sentence_1)
            pos_tags_str = ' '.join([f"{token}/{tag}" for token, tag in pos_tags])
            entities_str = ' '.join([f"{ent}/{label}" for ent, label in entities])
            combined_input = f"{sentence_1} {SEP} {segment_location_1} {SEP} {paraphrase_type} {SEP} POS: {pos_tags_str} {SEP} NER: {entities_str}"
        else:
            combined_input = f"{sentence_1} {SEP} {segment_location_1} {SEP} {paraphrase_type}"

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
                batch_size=hyperparams['batch_size'],
                patience=hyperparams['patience'], print_messages=DEV_MODE, alpha_ngram=0.0, alpha_diversity=0.0, val_dataset = None):

    accumulation_steps = int(batch_size / 32)
    if not DEV_MODE:
        torch.cuda.empty_cache()

    num_epochs = 100 if not DEV_MODE else 5
    num_training_steps = num_epochs * len(train_data) // accumulation_steps
    progress_bar = tqdm(range(num_training_steps), disable=TQDM_DISABLE)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    scaler = GradScaler()

    bleu_scores = []
    best_bleu_score = -10
    best_epoch = 0
    epochs_without_improvement = 0

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

                if alpha_ngram != 0 or alpha_diversity !=0:
                    # Generate paraphrases
                    predictions = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_length=50,
                        num_beams=5,
                        early_stopping=True,
                    )

                    # Convert predictions to tensors if they are not already
                    if isinstance(predictions, list):
                        predictions = torch.tensor(predictions, device=device)

                    # Compute the n-gram penalty
                    # Decode predictions and inputs for penalty computation
                    pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
                    input_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

                    # Compute penalties
                    penalty = alpha_ngram * ngram_penalty(pred_texts, input_texts) + alpha_diversity * diversity_penalty(
                        pred_texts, input_texts)

                    loss = loss + penalty

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step(loss)
                progress_bar.update(1)

        # End time for this epoch
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        if print_messages:
            print(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_duration:.2f} seconds.")

        if val_data is not None:
            scores = evaluate_model(model, val_data, device, tokenizer, print_messages=print_messages, dataset=val_dataset)
            b = scores['bleu_score']
            bleu_scores.append(b)

            if b > best_bleu_score:
                best_bleu_score = scores['bleu_score']
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                # Save the best model
                torch.save(model.state_dict(), model_save_path)
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
    del model
    torch.cuda.empty_cache()
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large",
                                                         local_files_only=True)
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

def evaluate_model(model, dataloader, device, tokenizer, print_messages=True, dataset = None):
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
            predictions.extend(pred_text)
            if dataset is None:
                references.extend([
                    tokenizer.decode(label, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    for label in labels
                ])
                inputs.extend([
                    tokenizer.decode(input_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    for input_id in input_ids
                ])
            else:
                inputs = dataset["sentence1"].tolist()
                references = dataset["sentence2"].tolist()

            model.train()


    print(references)
    # Calculate BLEU score
    bleu_score_reference = bleu.corpus_score(references, [predictions]).score
    # Penalize BLEU score if its to close to the input
    bleu_score_inputs = 100 - bleu.corpus_score(inputs, [predictions]).score
    penalized_bleu = bleu_score_reference * bleu_score_inputs / 52
    if print_messages:
        print(f"BLEU Score: {bleu_score_reference}", f"Negative BLEU Score with input: {bleu_score_inputs}")
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

    test_dataset = pd.read_csv("data/etpc-paraphrase-generation-test-student.csv", sep="\t")#[:10]
    test_dataset = test_dataset if not DEV_MODE else test_dataset[:10] #todo: put back at the end
    test_data = transform_data(test_dataset)

    print(f"Loaded {len(train_dataset)} training samples.")

    #if DEV_MODE:
     #   scores_before_training = evaluate_model(model, val_data, device, tokenizer)
      #  bleu_score_before_training, _ = scores_before_training.values()


    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large",
                                                         local_files_only=True)
    model.to(device)
    model = train_model(model, train_data, val_data, device, tokenizer,
                        learning_rate=hyperparams['learning_rate'], batch_size=hyperparams['batch_size'],
                        patience=hyperparams['patience'], print_messages=True, alpha_ngram=hyperparams["alpha"],
                        alpha_diversity=hyperparams["alpha"], val_dataset = val_dataset)  # todo: set print_messages to DEV_MODE again
    scores = evaluate_model(model, val_data, device, tokenizer, val_dataset = val_dataset)
    bleu_score, _ = scores.values()
    print(f"The penalized BLEU-score of the model is: {bleu_score:.3f}")



#print(f"The METEOR-score of the model is: {meteor_score:.3f}")
    #print(f"Without training: \n BLEU: {bleu_score_before_training:.3f}")# \n METEOR: {meteor_score_before_training}")
    # Clear GPU memory
    #del model
    #torch.cuda.empty_cache()

    #paraphrases = generate_paraphrases(model, val_data, device, tokenizer)
    #paraphrases.to_csv("predictions/bart/generation_predict.csv", index=False, sep="\t")


    test_ids = test_dataset["id"]
    #bleu_test = evaluate_model_on_testset(model, test_dataset, device, tokenizer)
    #print(f"Bleu test: {bleu_test}")


    #if not DEV_MODE:
        #test_results = test_model(test_data, test_ids, device, model, tokenizer)
    #    test_results.to_csv(
    #        "predictions/bart/etpc-paraphrase-generation-test-output.csv", index=False, sep="\t"
    #    )

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_generation(args)
    # Delete the saved model file
    if os.path.exists(model_save_path):
        os.remove(model_save_path)
        print(f"Deleted model file at {model_save_path}")
    else:
        print(f"Model file at {model_save_path} not found.")
