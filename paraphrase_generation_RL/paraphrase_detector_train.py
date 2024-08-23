import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, BartForConditionalGeneration, BertTokenizer
from multitask_classifier import MultitaskBERT
import pandas as pd
#import bart_generation
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

# Evaluator function to compute the paraphrase probability
def predict_paraphrase(evaluator, sentence1, sentence2, evaluator_tokenizer, device):
    """
    Predict whether two sentences are paraphrases using the given model.

    Parameters:
    - evaluator (torch.nn.Module): The trained model to use for prediction.
    - sentence1 (str): The first sentence to compare.
    - sentence2 (str): The second sentence to compare with the first.
    - evaluator_tokenizer (PreTrainedTokenizer): Tokenizer for encoding the input sentences.
    - device (torch.device): Device to run the model on (e.g., 'cuda' or 'cpu').

    Returns:
    - float: The predicted probability that the two sentences are paraphrases.
    """
    evaluator.eval()
    with torch.no_grad():
        # Tokenize input sentences and move to the device
        inputs = evaluator_tokenizer(sentence1, sentence2, return_tensors="pt", padding=True, truncation=True).to(device)

        # Forward pass through the model
        logits = evaluator(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

        # If the model returns a tensor directly, it should be your logits
        # Assuming binary classification, extract the logit for the positive class
        probability = torch.sigmoid(logits[:, 0]).item()
        #print(f"probability: {probability}, sentences: {sentence1}, {sentence2}")
        return probability

def load_evaluator(path, device):
    """
    This function loads a pre-trained paraphrase evaluation model and its tokenizer from a saved file.

    Parameters:

    path: Path to the saved model file (type: str).
    device: Device to load the model on (e.g., 'cuda' or 'cpu') (type: torch.device).
    Returns:

    tuple: A tuple containing two elements:
    evaluator: The loaded paraphrase evaluation model (type: MultitaskBERT).
    evaluator_tokenizer: The tokenizer used by the model (type: PreTrainedTokenizer).
    """
    evaluator_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    saved = torch.load(path, map_location=device)
    evaluator = MultitaskBERT(saved["model_config"]).to(device)
    evaluator.load_state_dict(saved["model"])
    return evaluator, evaluator_tokenizer


# Fine-tuning step with reinforcement learning
def fine_tune_generator(model, evaluator, evaluator_tokenizer, train_data, device, tokenizer, train_dataset, learning_rate, num_epochs=10 if not DEV_MODE else 2):
    """
    This function fine-tunes a text generation model using reinforcement learning based on paraphrase evaluation.

    Parameters:

    model: The text generation model to be fine-tuned (type: nn.Module).
    evaluator: The pre-trained paraphrase evaluation model (type: MultitaskBERT).
    evaluator_tokenizer: Tokenizer for the paraphrase evaluation model (type: PreTrainedTokenizer).
    train_data: A PyTorch dataloader for the training data (type: DataLoader).
    device: Device to run the model on (e.g., 'cuda' or 'cpu') (type: torch.device).
    tokenizer: Tokenizer for the text generation model (type: PreTrainedTokenizer).
    train_dataset: The training dataset object (type depends on the dataset format).
    learning_rate: Learning rate for the fine-tuning optimizer (type: float).
    num_epochs: Number of epochs for fine-tuning (type: int, default: 10).
    Returns:

    nn.Module: The fine-tuned text generation model.
    """

    optimizer = AdamW(model.parameters(), lr=learning_rate) #1e-8 is the learning rate that is typically at the end of standard training. We are close to optimum and want to improve the model

    input_texts = train_dataset["sentence1"].tolist()

    model.train()
    for epoch in range(num_epochs):
        for _ , batch in enumerate(train_data):
            input_ids, attention_mask, labels, indices = [tensor.to(device) for tensor in batch]

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50, num_beams=5, early_stopping=True)

            generated_sentences = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in outputs]

            batch_indices = indices.tolist()
            input_texts_batch = [input_texts[idx] for idx in batch_indices]
            rewards = [
                predict_paraphrase(evaluator, i, gs, evaluator_tokenizer,
                                   device)
                for i, gs in zip(input_texts_batch, generated_sentences)]

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