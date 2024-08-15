from collections import Counter
from nltk import ngrams

def ngram_penalty(pred_texts, input_texts, n=3):
    """
    Compute the n-gram penalty for a batch of predictions compared to inputs.
    :param pred_texts: List of strings containing the generated texts.
    :param input_texts: List of strings containing the input texts.
    :param n: The size of the n-grams to consider.
    :return: The penalty term to be added to the loss function.
    """
    penalty = 0.0

    for pred_text, input_text in zip(pred_texts, input_texts):
        pred_tokens = pred_text.split()
        input_tokens = input_text.split()

        # Collect n-grams from predictions and inputs
        pred_ngrams = Counter(ngrams(pred_tokens, n))
        input_ngrams = Counter(ngrams(input_tokens, n))

        # Calculate penalty based on the difference in n-gram frequencies
        for ngram, count in pred_ngrams.items():
            if ngram in input_ngrams:
                if count > input_ngrams[ngram]:
                    penalty += (count - input_ngrams[ngram]) * (n - 1)  # Penalize excessive n-grams
            else:
                penalty += count * (n - 1)  # Penalize n-grams not found in the input

    return penalty / len(pred_texts)


def length_penalty(pred_texts, input_texts, max_length=50):
    """
    Compute the length penalty for a batch of predictions compared to inputs.
    :param pred_texts: List of strings containing the generated texts.
    :param input_texts: List of strings containing the input texts.
    :param max_length: The maximum allowable length.
    :return: The penalty term to be added to the loss function.
    """
    penalty = 0.0

    for pred_text, input_text in zip(pred_texts, input_texts):
        pred_length = len(pred_text.split())
        input_length = len(input_text.split())

        if pred_length < 0.8 * input_length:
            penalty += (0.8 * input_length - pred_length)  # Penalize short sentences
        elif pred_length > max_length:
            penalty += (pred_length - max_length)  # Penalize long sentences

    return penalty / len(pred_texts)


def diversity_penalty(pred_texts, input_texts):
    """
    Compute the diversity penalty for a batch of predictions compared to inputs.
    :param pred_texts: List of strings containing the generated texts.
    :param input_texts: List of strings containing the input texts.
    :return: The penalty term to be added to the loss function.
    """
    penalty = 0.0

    for pred_text, input_text in zip(pred_texts, input_texts):
        pred_tokens = pred_text.split()
        input_tokens = input_text.split()
        unique_pred_tokens = len(set(pred_tokens))
        unique_input_tokens = len(set(input_tokens))

        # Penalize if generated tokens are less diverse compared to input tokens
        if unique_pred_tokens < 0.5 * unique_input_tokens:
            penalty += (0.5 * unique_input_tokens - unique_pred_tokens)

    return penalty / len(pred_texts)
