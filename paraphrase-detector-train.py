import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, BartForConditionalGeneration, BertTokenizer
import pandas as pd
import paraphrase_detector
import multitask_classifier
import bart_generation
import warnings
from tqdm import tqdm
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

warnings.filterwarnings("ignore", category=FutureWarning)

# Evaluator function to compute the paraphrase probability
def predict_paraphrase(model, sentence1, sentence2, tokenizer, device):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(sentence1, sentence2, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        logits = model.predict_paraphrase(input_ids, attention_mask, input_ids, attention_mask)
        probability = torch.sigmoid(logits).item()
        return probability

def check_paraphrase(model_path, sentence1, sentence2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(model_path)
    config = saved["model_config"]

    model = multitask_classifier.MultitaskBERT(config)
    model.load_state_dict(saved["model"])
    model = model.to(device)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    probability = predict_paraphrase(model, sentence1, sentence2, tokenizer, device)
    return probability

# Reinforcement Learning fine-tuning step
def fine_tune_generator(model, evaluator_model_path, train_data, device, tokenizer, num_epochs=3):
    optimizer = AdamW(model.parameters(), lr=5e-6)
    evaluator = multitask_classifier.MultitaskBERT(torch.load(evaluator_model_path)["model_config"])
    evaluator.load_state_dict(torch.load(evaluator_model_path)["model"])
    evaluator = evaluator.to(device)
    evaluator_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    model.train()
    for epoch in tqdm(range(num_epochs)):
        for batch in train_data:
            input_ids, attention_mask, labels = [tensor.to(device) for tensor in batch]
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50, num_beams=5, early_stopping=True)
            generated_sentences = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in outputs]
            rewards = [predict_paraphrase(evaluator, tokenizer.decode(i, skip_special_tokens=True), gs, evaluator_tokenizer, device) for i, gs in zip(input_ids, generated_sentences)]
            rewards = torch.tensor(rewards, device=device)

            log_probs = torch.sum(torch.log(torch.cat([model(input_ids=i.unsqueeze(0), attention_mask=a.unsqueeze(0))[0] for i, a in zip(input_ids, attention_mask)])), dim=1)
            loss = -torch.mean(log_probs * rewards)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


# Main function to train and fine-tune the generator
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large").to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t") if not DEV_MODE else pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t")[:10]
    val_dataset = train_dataset.sample(frac=0.2, random_state=42)
    train_data = bart_generation.transform_data(train_dataset)
    val_data = bart_generation.transform_data(val_dataset)
    evaluator_model_path = "models/pretrain-10-0.001-sst.pt"

    print('Training generator. \n')
    #bart_generation.train_model(model, train_data, val_data, device, tokenizer)
    score_before_finetune = bart_generation.evaluate_model(model, val_data, device, tokenizer)
    print(f'Score before fine-tuning with evaluator: {score_before_finetune} \n')
    print('Training generator with feedback from evaluator. \n')
    fine_tune_generator(model, evaluator_model_path, train_data, device, tokenizer, num_epochs=3)
    score_after_finetune = bart_generation.evaluate_model(model, val_data, device, tokenizer)
    print(f'Score after fine-tuning with evaluator: {score_after_finetune} \n')

if __name__ == "__main__":
    main()
