import argparse

import paraphrase_generation_RL.paraphrase_detector_train
from optimizer import AdamW
from peft import PeftModel, PeftConfig
import qp_model
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
#from nltk.translate.meteor_score import single_meteor_score
from torch.cuda.amp import autocast, GradScaler
#from penalty_function import ngram_penalty, diversity_penalty#, length_penalty
import random
import numpy as np
import pandas as pd
import spacy
import torch
from sacrebleu.metrics import BLEU
# from nltk.translate.meteor_score import single_meteor_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration
from optimizer import EAdam
from torch.cuda.amp import autocast, GradScaler
from penalty_function import ngram_penalty, diversity_penalty
import time
import warnings
import socket
import os
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, MultiStepLR, OneCycleLR
from torch.optim import Adam
AdamW = Adam


# import nltk
# nltk.download('wordnet')

try:
    local_hostname = socket.gethostname()
except:
    local_hostname = None

DEV_MODE = False
if local_hostname == 'Corinna-PC' or local_hostname == "TABLET-TTS0K9R0":  # Todo: Add also laptop
    DEV_MODE = True

TQDM_DISABLE = not DEV_MODE

warnings.filterwarnings("ignore", category=FutureWarning)

hyperparams = {
    'optimizer': AdamW,
    'learning_rate': 8e-5,
    'batch_size': 64,
    'dropout_rate': 0.1,
    'patience': 5,
    'num_epochs': 100 if not DEV_MODE else 2,
    'alpha': 0.001,
    'scheduler': "ReduceLROnPlateau",
    'POS_NER_tagging': True,
    'l2_regularization': 0.01,
    'use_QP': False,
    'use_lora': False,
    'use_RL': False
}  # Todo: make every function take values from here

if hyperparams['POS_NER_tagging'] == True:
    nlp = spacy.load("en_core_web_sm")

if hyperparams["use_QP"] == True:
    from Quality_predictor import bart_generation_with_qp, qp_model, quality_measure

if hyperparams['use_lora'] == True:
    from peft import get_peft_model, LoraConfig, TaskType

if hyperparams['use_QP'] == True:
    from paraphrase_generation_RL import paraphrase_detector_train
    evaluator_model_path = "models/finetune-10-1e-05-qqp.pt"
    if DEV_MODE:
        evaluator_model_path = r"C:\Users\corin\OneDrive\Physik Master\SoSe 24\Deep Learning for Natural Language Processing\Project\models\qqp-finetune-10-1e-05.pt"  # models/finetune-10-1e-05-qqp.pt"

# Define a model save path
r = random.randint(10000, 99999)
model_save_path = f"models/bart_generation_{r}.pt"

def perform_pos_ner(text):
    """
    Perform POS tagging and NER on the given text.

    Parameters:
    text (str): The input text.
    nlp (spacy.Language): The spaCy language model.

    Returns:
    tuple: POS tags and NER entities.
    """
    doc = nlp(text)
    pos_tags = [(token.text, token.pos_) for token in doc]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return pos_tags, entities


def transform_data(dataset, max_length=256, use_tagging=hyperparams['POS_NER_tagging'],
                   use_QP= False, predict_with_qp = False, qpmodel = None, q_sem=0.7, q_syn=0.7,q_lex=1): #Row for QP use. We want a high lexical diversity, so we put higher q value on lexical
    """
     Transform the dataset for training or evaluation.

     Parameters:
     dataset (pd.DataFrame): Input dataset.
     tokenizer (transformers.PreTrainedTokenizer): Tokenizer for encoding.
     max_length (int): Maximum sequence length.
     use_tagging (bool): Whether to use POS tagging and NER.

     Returns:
     DataLoader: DataLoader for the processed dataset.
     """

    if use_QP == True:
        dataloader = bart_generation_with_qp.transform_data_with_qualitypredictor(dataset, qpmodel, predict_with_qp, q_sem=q_sem, q_syn=q_syn, q_lex=q_lex, max_length=max_length, use_tagging=use_tagging)
        return dataloader

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    input_ids = []
    attention_masks = []
    labels = []
    indices = []

    SEP = tokenizer.sep_token

    for idx, row in tqdm(dataset.iterrows(), total=len(dataset), disable=TQDM_DISABLE):
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
        indices.append(idx)

    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    labels = torch.stack(labels)
    indices = torch.tensor(indices)

    dataset = TensorDataset(input_ids, attention_masks, labels, indices)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    return dataloader

def train_model(model, train_data, val_data, device, tokenizer,
                learning_rate=hyperparams['learning_rate'], batch_size=hyperparams['batch_size'], optimizer="Adam", l2_lambda=hyperparams["l2_regularization"],
                patience=5, use_scheduler="ReduceLROnPlateau", print_messages=True,
                alpha_ngram=0.0, alpha_diversity=0.0, train_dataset = None, # This line is for loss function engineering
                qpmodel = None): # This line is for QP
    """
    Train a model.

    Parameters:
    - model (torch.nn.Module): The model to be trained.
    - train_data (DataLoader): DataLoader for training data, containing input tensors and corresponding labels.
    - val_data (DataFrame): DataFrame containing validation data for evaluating the model's performance.
    - device (torch.device): Device to run the training on (e.g., 'cuda' or 'cpu').
    - tokenizer (PreTrainedTokenizer): Tokenizer for encoding and decoding text data.
    - learning_rate (float, default=hyperparams['learning_rate']): Learning rate for the optimizer.
    - batch_size (int, default=hyperparams['batch_size']): Batch size for training.
    - optimizer (str, default="Adam"): Type of optimizer to use ('Adam' or 'EAdam').
    - l2_lambda (float, default=hyperparams["l2_regularization"]): Coefficient for L2 regularization.
    - patience (int, default=5): Number of epochs to wait for improvement before applying early stopping.
    - print_messages (bool, default=True): Flag to enable or disable printing training progress and statistics.
    - alpha_ngram (float, default=0.0): Weight for n-gram penalty applied during loss calculation.
    - alpha_diversity (float, default=0.0): Weight for diversity penalty applied during loss calculation.
    - train_dataset (pd.DataFrame, optional): DataFrame required for loss function engineering (only needed if alpha_diversity or alpha_ngram are nonzero).
    - qpmodel (object, optional): Quality Predictor model if Quality Prediction is applied.

    Returns:
    - torch.nn.Module: The best-performing model after training and validation.
    """

    torch.save(model.state_dict(), model_save_path)

    if hyperparams["use_lora"] == True:
        lora_save_path = "models/lora_model"
        model.save_pretrained(lora_save_path)

    accumulate_steps = int(batch_size / 32)
    num_epochs = hyperparams['num_epochs']

    val_dataloader = transform_data(val_data)
    if hyperparams['use_QP'] == True:
        val_dataloader = bart_generation_with_qp.transform_data_with_qualitypredictor(val_data, qpmodel, predict_with_qp=False)

    progress_bar = tqdm(range(num_epochs * len(train_data) // accumulate_steps), disable=TQDM_DISABLE)


    input_texts = train_dataset["sentence1"].tolist()

    # Initialize optimizer
    if optimizer == "Adam":
        optimizer = AdamW(model.parameters(), lr=learning_rate)
    elif optimizer == "EAdam":
        optimizer = EAdam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Invalid optimizer choice.")

    # Initialize scheduler
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5)


    scaler = GradScaler()

    bleu_scores = []
    best_bleu_score = -10
    best_epoch = 0
    epochs_without_improvement = 0

    total_start_time = time.time()

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        model.train()
        epoch_start_time = time.time()
        for i, batch in enumerate(train_data):
            input_ids, attention_mask, labels, indices = [tensor.to(device) for tensor in batch]

            optimizer.zero_grad()

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / accumulate_steps

                # Add L2 regularization
                l2_reg = sum(param.norm() ** 2 for param in model.parameters())
                loss = loss + l2_lambda * l2_reg

                if alpha_ngram != 0 or alpha_diversity != 0:
                    predictions = model.generate(input_ids = input_ids, attention_mask=attention_mask, max_length=50,
                                                 num_beams=5, early_stopping=True)
                    pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)

                    batch_indices = indices.tolist()
                    input_texts_batch = [input_texts[idx] for idx in batch_indices]
                    penalty = alpha_ngram * ngram_penalty(pred_texts, input_texts_batch) + alpha_diversity * diversity_penalty(pred_texts, input_texts_batch)
                    loss = loss + penalty

            scaler.scale(loss).backward()

            if (i + 1) % accumulate_steps == 0:
                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()

                # For OneCycleLR, the scheduler needs to step after every batch
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()

                progress_bar.update(1)

        # Validation phase
        model.eval()
        val_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in val_dataloader:
                # Move each batch of tensors to the device
                val_input_ids, val_attention_mask, val_labels, _ = [tensor.to(device) for tensor in batch]

                # Forward pass
                val_outputs = model(input_ids=val_input_ids, attention_mask=val_attention_mask, labels=val_labels)
                val_loss += val_outputs.loss.item()
                num_batches += 1

            # Average validation loss over all batches
            val_loss /= num_batches

            # Step the scheduler based on validation loss if using ReduceLROnPlateau
            if scheduler and isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)

        # For other schedulers that require epoch-level stepping
        if scheduler and not isinstance(scheduler, (ReduceLROnPlateau, torch.optim.lr_scheduler.OneCycleLR)):
            scheduler.step()

        #current_lr = scheduler.get_last_lr()[0] if scheduler else learning_rate

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        if print_messages:
            print(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_duration:.2f} seconds.")

        torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():
            scores = evaluate_model(model, val_data, device, tokenizer, print_messages=print_messages)

        bleu_scores.append(scores['bleu_score'])

        if scores['bleu_score'] > best_bleu_score and epoch > 3:
            best_bleu_score = scores['bleu_score']
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            if hyperparams["use_lora"] == True:
                model.save_pretrained(lora_save_path)
            else:
                torch.save(model.state_dict(), model_save_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience and epoch > 15:
            if print_messages:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                print(f'Best BLEU score: {best_bleu_score} at epoch {best_epoch}.')
                print(f"History: {bleu_scores}")
            break

    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    if print_messages:
        print(f"Total training time: {total_training_time:.2f} seconds.")

    torch.cuda.empty_cache()

    # Reload the best model
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", local_files_only=True)

    if hyperparams["use_lora"] == True:
        base_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", local_files_only=True)

        # Load the LoRA configuration
        model = PeftModel.from_pretrained(base_model, lora_save_path)
        model.to(device)
        print("Loaded best Lora model")
    else:
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


def evaluate_model(model, dataset, device, tokenizer, print_messages=True):
    """
        You can use your train/validation set to evaluate models performance with the BLEU score.
        test_data is a Pandas Dataframe, the column "sentence1" contains all input sentence and
        the column "sentence2" contains all target sentences
        """
    model.eval()
    bleu = BLEU()
    predictions = []

    dataloader = transform_data(dataset)
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels, indices = [tensor.to(device) for tensor in batch]

            # Generate paraphrases
            outputs = model.generate(
                input_ids=input_ids,
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

    inputs = dataset["sentence1"].tolist()
    references = dataset["sentence2"].tolist()

   # if print_messages:
    #    for i in range(3 if not DEV_MODE else 1):
     #       print(i, "inputs:", inputs[i], "predictions:", predictions[i], "references:", references[i])

    # Calculate BLEU score
    bleu_score_reference = bleu.corpus_score(references, [predictions]).score
    # Penalize BLEU score if its to close to the input
    bleu_score_inputs = 100 - bleu.corpus_score(inputs, [predictions]).score

    print(f"BLEU Score: {bleu_score_reference}", f"Negative BLEU Score with input: {bleu_score_inputs}")

    # Penalize BLEU and rescale it to 0-100
    # If you perfectly predict all the targets, you should get an penalized BLEU score of around 52
    penalized_bleu = bleu_score_reference * bleu_score_inputs / 52
    print(f"Penalized BLEU Score: {penalized_bleu}")

    # meteor_score = single_meteor_score(references, predictions)

    model.train()

    return {"bleu_score": penalized_bleu, "meteor_score": "meteor_score not computed"}  # todo: put back meteor if want to use

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

    hyperparamer_tuning_mode = False
    normal_mode = True

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t")
    train_dataset = train_dataset if not DEV_MODE else train_dataset[:10]
    train_dataset_shuffled = train_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    val_dataset = train_dataset_shuffled.sample(frac=0.2, random_state=42)

    train_dataset = train_dataset_shuffled.drop(val_dataset.index)
    train_dataset = train_dataset.reset_index()

    print(f"Loaded {len(train_dataset)} training samples.")

    val_dataset = val_dataset.reset_index()
    val_data = val_dataset

    train_data = transform_data(train_dataset)
    # val_data = transform_data(val_dataset)

    test_dataset = pd.read_csv("data/etpc-paraphrase-generation-test-student.csv", sep="\t")
    test_dataset = test_dataset if not DEV_MODE else test_dataset[:10]  # todo: put back at the end
    #test_data = transform_data(test_dataset)

    if hyperparams["use_RL"]:
        from paraphrase_generation_RL import paraphrase_detector_train
        evaluator_model_path = "models/finetune-10-1e-05-qqp.pt"
        if DEV_MODE:
            evaluator_model_path = r"C:\Users\corin\OneDrive\Physik Master\SoSe 24\Deep Learning for Natural Language Processing\Project\models\qqp-finetune-10-1e-05.pt"  # models/finetune-10-1e-05-qqp.pt"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large").to(device)
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

        evaluator, evaluator_tokenizer = paraphrase_detector_train.load_evaluator(evaluator_model_path, device)

        print('Training generator.\n')
        model = train_model(model, train_data, val_data, device, tokenizer,
                            learning_rate=8e-5, batch_size=hyperparams['batch_size'],
                            patience=hyperparams['patience'], print_messages=True, alpha_ngram=hyperparams["alpha"],
                            alpha_diversity=hyperparams["alpha"], optimizer="Adam",
                            use_scheduler='ReduceLROnPlateau', train_dataset = train_dataset)
        print('Finished training generator.')

        score_before_finetune = evaluate_model(model, val_data, device, tokenizer)
        print(f'Score before fine-tuning with RL: {score_before_finetune}\n')

        print('Training generator with feedback from evaluator.\n')
        model = paraphrase_detector_train.fine_tune_generator(model, evaluator, evaluator_tokenizer, train_data, device, tokenizer, num_epochs=5)

        score_after_finetune = evaluate_model(model, val_data, device, tokenizer)
        print(f'Score after fine-tuning with evaluator: {score_after_finetune}\n')

    if hyperparams["use_QP"]:
        qpmodel = bart_generation_with_qp.load_and_train_qp_model()
        train_data_qp = bart_generation_with_qp.transform_data_with_qualitypredictor(train_dataset, qpmodel, predict_with_qp=True)

        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", local_files_only=True)
        model.to(device)
        model = train_model(model, train_data_qp, val_data, device, tokenizer,
                            learning_rate=hyperparams['learning_rate'], batch_size=hyperparams['batch_size'],
                            patience=hyperparams['patience'], print_messages=True,
                            use_scheduler='ReduceLROnPlateau', qpmodel = qpmodel, train_dataset= train_dataset)

        print("Training finished.")

        scores = evaluate_model(model, val_data, device, tokenizer)
        bleu_score, _ = scores.values()
        print(f"The penalized BLEU-score of the model is: {bleu_score:.3f}")
        return

    if hyperparams['use_lora'] == True:

        base_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", local_files_only=True)

        # Configure LoRA
        lora_config = LoraConfig(
            r=8,  # Low-rank factor
            lora_alpha=16,  # Scaling factor
            lora_dropout=0.1,  # Dropout rate for LoRA layers
            task_type=TaskType.SEQ_2_SEQ_LM  # Task type for sequence-to-sequence models
        )

        # Wrap the model with LoRA
        model = get_peft_model(base_model, lora_config)

        model.to(device)

        model = train_model(model, train_data, val_dataset, device, tokenizer,
                            learning_rate=8e-5, batch_size=hyperparams['batch_size'],
                            patience=hyperparams['patience'], print_messages=True, alpha_ngram=0.001,
                            alpha_diversity=0.001, optimizer="Adam",
                            use_scheduler='ReduceLROnPlateau', train_dataset=train_dataset)
        scores = evaluate_model(model, val_data, device, tokenizer)
        bleu_score, _ = scores.values()
        print(f"The penalized BLEU-score of the model is: {bleu_score:.3f}")

    if hyperparamer_tuning_mode == True:

        list_alpha = [0.001, 0.01, 0.1, 1]
        list_bleu = []
        for a in list_alpha:
            print(f"alpha: {a}")
            model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", local_files_only=True)
            model.to(device)
            model = train_model(model, train_data, val_data, device, tokenizer,
                            learning_rate=8e-5, batch_size=hyperparams['batch_size'],
                            patience=hyperparams['patience'], print_messages=True, alpha_ngram=a,
                            alpha_diversity=a, optimizer="Adam",
                            use_scheduler='ReduceLROnPlateau', train_dataset = train_dataset)  # todo: Determine best scheduler first  # todo: Determine best scheduler first, Remember to put the POS NER hyperparameter to true
            scores = evaluate_model(model, val_data, device, tokenizer)
            bleu_score, _ = scores.values()
            list_bleu.append(bleu_score)
            print(f"The penalized BLEU-score for alpha {a} of the model is: {bleu_score:.3f}")
            del model
        print(f"alpha, bleu: {list_alpha, list_bleu}.")


    if normal_mode == True:  # Todo: This code below is to check loss function engineering, BUT see below
        print("Adam:")
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", local_files_only=True)
        model.to(device)
        model = train_model(model, train_data, val_data, device, tokenizer,
                            learning_rate=hyperparams['learning_rate'], batch_size=hyperparams['batch_size'],
                            patience=hyperparams['patience'], print_messages=True, alpha_ngram=0.0,
                            alpha_diversity=0.0, optimizer="Adam",
                            use_scheduler='ReduceLROnPlateau')
        scores = evaluate_model(model, val_data, device, tokenizer)
        bleu_score, _ = scores.values()
        print(f"The penalized BLEU-score of the model is: {bleu_score:.3f}")
        del model
        print("EAdam:")
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", local_files_only=True)
        model.to(device)
        model = train_model(model, train_data, val_data, device, tokenizer,
                            learning_rate=hyperparams['learning_rate'], batch_size=hyperparams['batch_size'],
                            patience=hyperparams['patience'], print_messages=True, alpha_ngram=0.0,
                            alpha_diversity=0.0, optimizer="EAdam",
                            use_scheduler='ReduceLROnPlateau')
        scores = evaluate_model(model, val_data, device, tokenizer)
        bleu_score, _ = scores.values()
        print(f"The penalized BLEU-score of the model is: {bleu_score:.3f}")
        del model


# print(f"The METEOR-score of the model is: {meteor_score:.3f}")
# print(f"Without training: \n BLEU: {bleu_score_before_training:.3f}")# \n METEOR: {meteor_score_before_training}")
# Clear GPU memory
# del model
# torch.cuda.empty_cache()

# paraphrases = generate_paraphrases(model, val_data, device, tokenizer)
# paraphrases.to_csv("predictions/bart/generation_predict.csv", index=False, sep="\t")


# test_ids = test_dataset["id"]
# bleu_test = evaluate_model_on_testset(model, test_dataset, device, tokenizer)
# print(f"Bleu test: {bleu_test}")


# if not DEV_MODE:
# test_results = test_model(test_data, test_ids, device, model, tokenizer)
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