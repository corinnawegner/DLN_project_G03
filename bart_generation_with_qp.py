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
import qp_model
import warnings
import socket
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
#from nltk.translate.meteor_score import single_meteor_score

from torch.cuda.amp import autocast, GradScaler
#from penalty_function import ngram_penalty, diversity_penalty#, length_penalty
import time

try:
    local_hostname = socket.gethostname()
except:
    local_hostname = None

DEV_MODE = False
if local_hostname == 'Corinna-PC' or local_hostname == "TABLET-TTS0K9R0":  # Todo: Add also laptop
    DEV_MODE = True

TQDM_DISABLE = not DEV_MODE

batch_size = 64 if not DEV_MODE else 1

warnings.filterwarnings("ignore", category=FutureWarning)

r = random.randint(10000, 99999)
model_save_path = f"models/bart_generation_quality_control_{r}.pt"

hyperparams = {
    'optimizer': AdamW,
    'learning_rate': 1e-5,
    'batch_size': 64,
    'dropout_rate': 0.0,
    'patience': 3,
    'num_epochs': 100 if not DEV_MODE else 10,
    'alpha': 1e-2,
}  # Todo: make every function take values from here

def transform_data_with_qualitypredictor(dataset, qpmodel, predict_with_qp, q_sem=1, q_syn=1,
                                         q_lex=1, max_length=256):  # todo: tune q values
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
        if 'sentence2' in row and predict_with_qp == True:
            sentence_2 = row['sentence2']
            predictions = qp_model.predict_quality(qpmodel, [(sentence_1, sentence_2)])
        else:
            predictions = torch.tensor([[q_sem, q_syn, q_lex]])
        combined_input = f"{sentence_1} [SEP] {segment_location_1} [SEP] {paraphrase_type} [SEP] {predictions}"

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
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)  # Todo: Shuffle train dataset?

    return dataloader


def train_model(model, train_data, val_data, device, tokenizer, learning_rate=hyperparams['learning_rate'], batch_size=hyperparams['batch_size'],
                patience=hyperparams['patience'], print_messages=DEV_MODE, alpha_ngram=1e-2, alpha_diversity=1e-2):
    accumulation_steps = int(batch_size / 32)
    if not DEV_MODE:
        torch.cuda.empty_cache()

    num_epochs = 100 if not DEV_MODE else 10
    num_training_steps = num_epochs * len(train_data) // accumulation_steps
    progress_bar = tqdm(range(num_training_steps), disable=TQDM_DISABLE)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    #optimizer = hyperparams['optimizer'] #Todo: After hyperparameter search, use this
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

                # Compute the n-gram penalty
                with torch.no_grad():
                    predictions = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_length=50,
                        num_beams=5,
                        early_stopping=True,
                    )
                    #if alpha_ngram != 0 or alpha_diversity != 0:
                     #   penalty = alpha_ngram * ngram_penalty(predictions) + alpha_diversity * diversity_penalty(
                      #  predictions)
                       # loss = loss + penalty

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
            scores = evaluate_model(model, val_data, device, tokenizer)
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

    # meteor_score = single_meteor_score(references, predictions)

    return {"bleu_score": penalized_bleu, "meteor_score": 'not computed'}


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
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", local_files_only=True)
    model.to(device)

    # config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20-D12')
    bleurt_model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20-D12')  # , config=config)
    bleurt_tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20-D12')

    df_train_data_qp = qp_model.load_etpc_paraphrase(bleurt_model, bleurt_tokenizer)

    qpmodel = qp_model.QualityPredictor()

    print('Training quality predictor')
    qpmodel = qp_model.train_quality_predictor(qpmodel, df_train_data_qp, num_epochs= 6 if not DEV_MODE else 1)  # Todo: find out best num epochs

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t")
    train_dataset = train_dataset if not DEV_MODE else train_dataset[:10]
    train_dataset_shuffled = train_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    test_dataset = pd.read_csv("data/etpc-paraphrase-generation-test-student.csv", sep="\t")  # [:10]
    test_dataset = test_dataset if not DEV_MODE else test_dataset[:10]

    # You might do a split of the train data into train/validation set here
    val_ratio = 0.2
    split_index = int(len(train_dataset_shuffled) * val_ratio)

    train_dataset = train_dataset_shuffled.iloc[split_index:]
    val_dataset = train_dataset_shuffled.iloc[:split_index]
    if DEV_MODE:  # Trying to check if early stopping works
        val_dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t")[:15]

    print("Transforming data.")
    train_data = transform_data_with_qualitypredictor(train_dataset, qpmodel, predict_with_qp = True)
    val_data = transform_data_with_qualitypredictor(val_dataset, qpmodel, predict_with_qp = False)
    # test_data = transform_data_with_qualitypredictor(test_dataset, predict_with_qp = False)

    print(f"Loaded {len(train_dataset)} training samples.")

    scores_before_training = evaluate_model(model, val_data, device, tokenizer)
    bleu_score_before_training, meteor_score_before_training = scores_before_training.values()

    #Todo: put back
    model = train_model(model, train_data, val_data, device, tokenizer)

    print("Training finished.")

    scores = evaluate_model(model, val_data, device, tokenizer)
    bleu_score, meteor_score = scores.values()
    print(f"The penalized BLEU-score of the model is: {bleu_score:.3f}")
    # print(f"The METEOR-score of the model is: {meteor_score:.3f}")
    print(f"Without training: \n BLEU: {bleu_score_before_training:.3f}")  # \n METEOR: {meteor_score_before_training}")

    # Todo: Put back test if needed
    # test_ids = test_dataset["id"]
    # test_results = test_model(test_data, test_ids, device, model, tokenizer)
    # if not DEV_MODE:
    #    test_results.to_csv(
    #        "predictions/bart/etpc-paraphrase-generation-test-output.csv", index=False, sep="\t"
    #    )


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_generation(args)
