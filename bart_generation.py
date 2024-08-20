import argparse
import random
import numpy as np
import pandas as pd
import spacy
import torch
from optimizer import AdamW
import warnings
import socket
import os
from bart_with_gcn import BartWithGCN, train_gcn_bart_model, transform_data_gcn, finetune_paraphrase_generation_gcn, evaluate_model_gcn
from gcn import GraphConvolution

try:
    local_hostname = socket.gethostname()
except:
    local_hostname = None

DEV_MODE = False
if local_hostname in ['Corinna-PC', "TABLET-TTS0K9R0", "DESKTOP-3D9LKBO"]:
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


#print(f"The METEOR-score of the model is: {meteor_score:.3f}")
    #print(f"Without training: \n BLEU: {bleu_score_before_training:.3f}")# \n METEOR: {meteor_score_before_training}")
    # Clear GPU memory
    #del model
    #torch.cuda.empty_cache()

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
    finetune_paraphrase_generation_gcn(args)
    # Delete the saved model file
    if os.path.exists(model_save_path):
        os.remove(model_save_path)
        print(f"Deleted model file at {model_save_path}")
    else:
        print(f"Model file at {model_save_path} not found.")
