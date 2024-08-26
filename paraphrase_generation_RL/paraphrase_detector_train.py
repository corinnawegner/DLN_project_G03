import torch
from torch.optim import AdamW
from transformers import BertTokenizer
from multitask_classifier_task import MultitaskBERT
import warnings
import socket
from torch.utils.data import DataLoader
from data.datasets import preprocess_string
import numpy as np

# Determine the mode based on the hostname
try:
    local_hostname = socket.gethostname()
except Exception:
    local_hostname = None

DEV_MODE = local_hostname in ['Corinna-PC', "TABLET-TTS0K9R0", "DESKTOP-3D9LKBO"]
TQDM_DISABLE = not DEV_MODE

warnings.filterwarnings("ignore", category=FutureWarning)

# Evaluator function to compute the paraphrase probability
def predict_paraphrase_RL(evaluator, sentences1, sentences2, evaluator_tokenizer):
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

    # Whoever reads this, I want to let you know that this multitask bert pipeline is overengineered
    # I present to you, a rodeo through the sts pipeline:

    evaluator.eval()

    sentences1 = [preprocess_string(s) for s in sentences1]
    sentences2 = [preprocess_string(s) for s in sentences2]

    encoding1 = evaluator_tokenizer(sentences1, return_tensors="pt", padding=True, truncation=True)
    encoding2 = evaluator_tokenizer(sentences2, return_tensors="pt", padding=True, truncation=True)

    token_ids = torch.LongTensor(encoding1["input_ids"])
    attention_mask = torch.LongTensor(encoding1["attention_mask"])

    token_ids2 = torch.LongTensor(encoding2["input_ids"])
    attention_mask2 = torch.LongTensor(encoding2["attention_mask"])

    logits = evaluator.predict_similarity(token_ids, attention_mask, token_ids2, attention_mask2)

    probabilities = logits/5

    print(probabilities)

    return probabilities

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

    config = saved['model_config']

    print(config)

    model = MultitaskBERT(config)
    model.load_state_dict(saved["model"])
    model = model.to(device)

    evaluator = MultitaskBERT(config).to(device)
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
            rewards = predict_paraphrase_RL(evaluator, input_texts_batch, generated_sentences, evaluator_tokenizer)

            rewards = torch.tensor(rewards, device=device)

            # Compute the log probabilities for the generated sentences

            model_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = model_outputs.logits

            # Apply softmax to get probabilities and then take the log
            log_probs = torch.log_softmax(logits, dim=-1)

            # Ensure generated_token_indices are the correct size
            generated_token_indices = outputs[:, :log_probs.size(1)]  # Ensure matching dimensions

            log_probs_for_generated_tokens = torch.gather(log_probs, dim=-1, index=generated_token_indices.unsqueeze(-1)).squeeze(-1)

            # Average log_probs over the sequence length
            mean_log_probs = log_probs_for_generated_tokens.mean(dim=1)

            # Compute the loss (Reinforcement Learning Loss)
            loss = -torch.mean(mean_log_probs * rewards)

            # Compute the loss
            loss = -torch.mean(mean_log_probs * rewards)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return model