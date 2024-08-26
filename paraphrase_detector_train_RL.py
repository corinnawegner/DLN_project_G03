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
def predict_paraphrase_RL(evaluator, sentences1, sentences2, evaluator_tokenizer, device):
    """
    Predict whether two sentences are paraphrases using the given model.

    Parameters:
    - evaluator (torch.nn.Module): The trained model to use for prediction.
    - sentences1 (list of str): The first set of sentences to compare.
    - sentences2 (list of str): The second set of sentences to compare with the first.
    - evaluator_tokenizer (PreTrainedTokenizer): Tokenizer for encoding the input sentences.

    Returns:
    - float: The predicted probability that the two sentences are paraphrases.
    """
    evaluator.eval()

    sentences1 = [preprocess_string(s) for s in sentences1]
    sentences2 = [preprocess_string(s) for s in sentences2]

    encoding1 = evaluator_tokenizer(sentences1, return_tensors="pt", padding=True, truncation=True)
    encoding2 = evaluator_tokenizer(sentences2, return_tensors="pt", padding=True, truncation=True)

    # Move tensors to the correct device
    token_ids = encoding1["input_ids"].to(device)
    attention_mask = encoding1["attention_mask"].to(device)

    token_ids2 = encoding2["input_ids"].to(device)
    attention_mask2 = encoding2["attention_mask"].to(device)

    with torch.no_grad():
        logits = evaluator.predict_similarity(token_ids, attention_mask, token_ids2, attention_mask2)

    probabilities = logits / 5

    return probabilities


# Fine-tuning step with reinforcement learning
def fine_tune_generator(model: object, evaluator_path: object, train_data: object, device: object, tokenizer: object, train_dataset: object, learning_rate: object,
                        num_epochs: object = 10 if not DEV_MODE else 2) -> object:
    """
    Fine-tune a text generation model using reinforcement learning based on paraphrase evaluation.

    Parameters:
    - model: The text generation model to be fine-tuned.
    - evaluator: The pre-trained paraphrase evaluation model.
    - evaluator_tokenizer: Tokenizer for the paraphrase evaluation model.
    - train_data: A PyTorch dataloader for the training data.
    - device: Device to run the model on (e.g., 'cuda' or 'cpu').
    - tokenizer: Tokenizer for the text generation model.
    - train_dataset: The training dataset object.
    - learning_rate: Learning rate for the fine-tuning optimizer.
    - num_epochs: Number of epochs for fine-tuning.

    Returns:
    - nn.Module: The fine-tuned text generation model.
    """

    saved = torch.load(evaluator_path, map_location=device)


    evaluator_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    config = saved['model_config']
    evaluator = MultitaskBERT(config)  # Create a single model instance
    evaluator.load_state_dict(saved["model"])
    evaluator.to(device)  # Move to the correct device

    print("Device:")
    print(next(model.parameters()).device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    input_texts = train_dataset["sentence1"].tolist()

    model.train()
    for epoch in range(num_epochs):
        for _, batch in enumerate(train_data):
            input_ids, attention_mask, labels, indices = [tensor.to(device) for tensor in batch]

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50, num_beams=5, early_stopping=True)

            generated_sentences = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in outputs]

            batch_indices = indices.tolist()
            input_texts_batch = [input_texts[idx] for idx in batch_indices]
            rewards = predict_paraphrase_RL(evaluator, input_texts_batch, generated_sentences, evaluator_tokenizer, device)

            rewards = torch.tensor(rewards, device=device)

            # Compute the log probabilities for the generated sentences
            model_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = model_outputs.logits

            log_probs = torch.log_softmax(logits, dim=-1)
            generated_token_indices = outputs[:, :log_probs.size(1)]
            log_probs_for_generated_tokens = torch.gather(log_probs, dim=-1, index=generated_token_indices.unsqueeze(-1)).squeeze(-1)
            mean_log_probs = log_probs_for_generated_tokens.mean(dim=1)

            loss = -torch.mean(mean_log_probs * rewards)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return model
