import argparse
import random
import numpy as np
import pandas as pd
import torch
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration
from optimizer import AdamW
#from Sophia import SophiaG
import warnings
import socket

try:
    local_hostname = socket.gethostname()
except:
    local_hostname = None

DEV_MODE = False
if local_hostname == 'Corinna-PC' or local_hostname == "TABLET-TTS0K9R0": #Todo: Add also laptop
    DEV_MODE = True

TQDM_DISABLE = not DEV_MODE
TRAINING = not DEV_MODE

batch_size = 64 if not DEV_MODE else 1

warnings.filterwarnings("ignore", category=FutureWarning)

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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) #Todo: Shuffle train dataset?

    return dataloader

def train_model(model, train_data, val_data, device, tokenizer, patience=3):
    """
    Train the model. Return and save the best model.
    https://huggingface.co/docs/transformers/en/training#train-in-native-pytorch #Todo: Put in references
    """
    if not DEV_MODE:
        torch.cuda.empty_cache()

    num_epochs = 50 if not DEV_MODE else 10
    num_training_steps = num_epochs * len(train_data)
    progress_bar = tqdm(range(num_training_steps))

    optimizer = AdamW(model.parameters(), lr=5e-5)
    #optimizer = SophiaG(model.parameters(), lr=2e-4, betas=(0.965, 0.99), rho = 0.01, weight_decay=1e-1)


    best_bleu_score = -10
    epochs_without_improvement = 0

    model.train()
    for epoch in range(num_epochs):
        for batch in train_data:
            input_ids, attention_mask, labels = [tensor.to(device) for tensor in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        if val_data is not None:
            bleu_score = evaluate_model(model, val_data, device, tokenizer)
            print(f"Validation BLEU score after epoch {epoch + 1}: {bleu_score:.3f}")

            # Save the best model
            if bleu_score > best_bleu_score:
                print(f'new best bleu score: {bleu_score}')
                best_bleu_score = bleu_score
                epochs_without_improvement = 0
            else:
                print('no improvement')
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

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


def evaluate_model(model, dataloader, device, tokenizer):
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

    print(f"BLEU Score: {bleu_score_reference}", f"Negative BLEU Score with input: {bleu_score_inputs}")

    # Penalize BLEU and rescale it to 0-100
    # todo: If you perfectly predict all the targets, you should get an penalized BLEU score of around 52
    penalized_bleu = bleu_score_reference * bleu_score_inputs / 52
    print(f"Penalized BLEU Score: {penalized_bleu}")

    return penalized_bleu


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
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", local_files_only=True)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t")
    train_dataset = train_dataset if not DEV_MODE else train_dataset[:10]
    train_dataset_shuffled = train_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    test_dataset = pd.read_csv("data/etpc-paraphrase-generation-test-student.csv", sep="\t")#[:10]
    test_dataset = test_dataset if not DEV_MODE else test_dataset[:10]

    # You might do a split of the train data into train/validation set here
    val_ratio = 0.2
    split_index = int(len(train_dataset_shuffled) * val_ratio)

    train_dataset = train_dataset_shuffled.iloc[split_index:]
    val_dataset = train_dataset_shuffled.iloc[:split_index]
    if DEV_MODE: #Trying to check if early stopping works
        val_dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t")[:15]

    train_data = transform_data(train_dataset)
    val_data = transform_data(val_dataset)
    test_data = transform_data(test_dataset)

    print(f"Loaded {len(train_dataset)} training samples.")

    bleu_score_before_training = evaluate_model(model, val_data, device, tokenizer)

    if TRAINING:
        model = train_model(model, train_data, val_data, device, tokenizer, patience=5)

    print("Training finished.")

    bleu_score = evaluate_model(model, val_data, device, tokenizer)
    print(f"The penalized BLEU-score of the model is: {bleu_score:.3f}")
    print(f"Without training: {bleu_score_before_training:.3f}")

    #test_ids = test_dataset["id"]
    #test_results = test_model(test_data, test_ids, device, model, tokenizer)
    #if not DEV_MODE:
     #   test_results.to_csv(
      #      "predictions/bart/etpc-paraphrase-generation-test-output.csv", index=False, sep="\t"
       # )

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_generation(args)