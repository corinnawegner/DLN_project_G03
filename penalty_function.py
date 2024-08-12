from collections import Counter
import torch


def ngram_penalty(predictions, n=3):
    """
    Compute the n-gram penalty for a batch of predictions.
    :param predictions: Tensor of shape (batch_size, seq_length) containing the predicted tokens.
    :param n: The size of the n-grams to consider.
    :return: The penalty term to be added to the loss function.
    """
    batch_size, seq_length = predictions.size()
    penalty = 0.0

    for i in range(batch_size):
        tokens = predictions[i].cpu().tolist()
        token_counter = Counter()

        # Collect n-grams
        for start in range(seq_length - n + 1):
            ngram = tuple(tokens[start:start + n])
            token_counter[ngram] += 1

        # Calculate penalty based on frequency of n-grams
        for count in token_counter.values():
            if count > 1:
                penalty += (count - 1) * (n - 1)  # Penalize repeated n-grams

    return penalty / batch_size


def length_penalty(predictions, max_length=50):
    batch_size, seq_length = predictions.size()
    penalty = 0.0

    for i in range(batch_size):
        length = (predictions[i] != 0).sum().item()  # Assuming 0 is the padding token
        if length < 0.8 * max_length:
            penalty += (0.8 * max_length - length)  # Penalize short sentences
        elif length > max_length:
            penalty += (length - max_length)  # Penalize long sentences

    return penalty / batch_size


def diversity_penalty(predictions):
    batch_size, seq_length = predictions.size()
    penalty = 0.0

    for i in range(batch_size):
        tokens = predictions[i].cpu().tolist()
        unique_tokens = len(set(tokens))
        if unique_tokens < 0.5 * seq_length:  # Penalize if less than 50% of tokens are unique
            penalty += (0.5 * seq_length - unique_tokens)

    return penalty / batch_size
