import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, BartForConditionalGeneration, BertTokenizer
import multitask_classifier
import pandas as pd
import bart_generation
import warnings
from tqdm import tqdm
import socket

# Determine the mode based on the hostname
try:
    local_hostname = socket.gethostname()
except Exception:
    local_hostname = None

DEV_MODE = local_hostname in ['Corinna-PC', "TABLET-TTS0K9R0"]  # Add any other hostname if needed
TQDM_DISABLE = not DEV_MODE

warnings.filterwarnings("ignore", category=FutureWarning)

evaluator_model_path = "models/finetune-10-1e-05-qqp.pt"

# Evaluator function to compute the paraphrase probability
def predict_paraphrase(model, sentence1, sentence2, tokenizer, device):
    """Predict whether two sentences are paraphrases using the given model."""
    model.eval()
    with torch.no_grad():
        # Tokenize input sentences and move to the device
        inputs = tokenizer(sentence1, sentence2, return_tensors="pt", padding=True, truncation=True).to(device)

        # Forward pass through the model
        logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

        # If the model returns a tensor directly, it should be your logits
        # Assuming binary classification, extract the logit for the positive class
        probability = torch.sigmoid(logits[:, 0]).item()

        return probability


def load_evaluator(path, device):
    evaluator_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    saved = torch.load(path, map_location=device)
    evaluator = multitask_classifier.MultitaskBERT(saved["model_config"]).to(device)
    evaluator.load_state_dict(saved["model"])
    return evaluator, evaluator_tokenizer


# Fine-tuning step with reinforcement learning
def fine_tune_generator(model, evaluator, evaluator_tokenizer, train_data, device, tokenizer, num_epochs=3):
    optimizer = AdamW(model.parameters(), lr=1e-8)

    model.train()
    for epoch in tqdm(range(num_epochs), disable=TQDM_DISABLE):
        for batch in train_data:
            input_ids, attention_mask, labels = [tensor.to(device) for tensor in batch]
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50, num_beams=5,
                                     early_stopping=True)
            generated_sentences = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for
                                   g in outputs]

            rewards = [
                predict_paraphrase(evaluator, tokenizer.decode(i, skip_special_tokens=True), gs, evaluator_tokenizer,
                                   device)
                for i, gs in zip(input_ids, generated_sentences)]
            rewards = torch.tensor(rewards, device=device)

            # Compute the log probabilities for the generated tokens
            # Forward pass to get logits
            model_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = model_outputs.logits

            # Apply softmax to get probabilities and then log
            log_probs = torch.log_softmax(logits, dim=-1)  # Shape: [batch_size, sequence_length, vocab_size]

            # Use the generated token indices to extract the relevant log probabilities
            # Here assuming you're working with tokenized outputs and logits
            # You'll need to handle this based on how generated sentences are processed
            # Example assumes the generated token indices are provided in `outputs`
            generated_token_indices = outputs  # Adjust this based on actual output format
            log_probs_for_generated_tokens = torch.gather(log_probs, dim=-1,
                                                          index=generated_token_indices.unsqueeze(-1)).squeeze(-1)

            # Average log_probs over the sequence length
            mean_log_probs = log_probs_for_generated_tokens.mean(dim=1)

            # Compute the loss
            loss = -torch.mean(mean_log_probs * rewards)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return model

# Main function to train and fine-tune the generator
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large").to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

    evaluator, evaluator_tokenizer = load_evaluator(evaluator_model_path, device)

    # Load datasets
    data_path = "data/etpc-paraphrase-train.csv"
    train_dataset = pd.read_csv(data_path, sep="\t") if not DEV_MODE else pd.read_csv(data_path, sep="\t")[:10]
    val_dataset = train_dataset.sample(frac=0.2, random_state=42)
    train_dataset = train_dataset.drop(val_dataset.index)

    # Transform data for training and evaluation
    train_data = bart_generation.transform_data(train_dataset)
    val_data = bart_generation.transform_data(val_dataset)

    print('Training generator.\n')
    #model = bart_generation.train_model(model, train_data, val_data, device, tokenizer)
    print('Finished training generator.')

    score_before_finetune = bart_generation.evaluate_model(model, val_data, device, tokenizer)
    print(f'Score before fine-tuning with evaluator: {score_before_finetune}\n')

    print('Training generator with feedback from evaluator.\n')
    model = fine_tune_generator(model, evaluator, evaluator_tokenizer, train_data, device, tokenizer, num_epochs=5)

    score_after_finetune = bart_generation.evaluate_model(model, val_data, device, tokenizer)
    print(f'Score after fine-tuning with evaluator: {score_after_finetune}\n')

if __name__ == "__main__":
    main()
