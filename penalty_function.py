from collections import Counter
from nltk import ngrams

def ngram_penalty(pred_texts, input_texts, n=4):
    """
    Compute the n-gram penalty for a batch of predictions compared to inputs.
    Penalizes n-grams that are present in both input and prediction texts.
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

        # Calculate penalty for n-grams that are in both prediction and input
        for ngram, pred_count in pred_ngrams.items():
            if ngram in input_ngrams:
                # Penalize each occurrence of the n-gram that is shared between input and prediction
                penalty += pred_count * (n - 1)

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

def diversity_penalty(pred_texts, input_texts, threshold=0.5):
    """
    Compute the diversity penalty for a batch of predictions compared to inputs.
    :param pred_texts: List of strings containing the generated texts.
    :param input_texts: List of strings containing the input texts.
    :param threshold: The threshold for common word frequency.
    :return: The penalty term to be added to the loss function.
    """
    penalty = 0.0

    for pred_text, input_text in zip(pred_texts, input_texts):
        pred_tokens = pred_text.split()
        input_tokens = input_text.split()

        # Combine tokens and calculate frequency
        all_tokens = pred_tokens + input_tokens
        token_counts = Counter(all_tokens)

        # Find common words (above threshold)
        common_words = [word for word, count in token_counts.items() if count / len(all_tokens) > threshold]

        # Calculate penalty based on common word frequency in predictions
        pred_token_counts = Counter(pred_tokens)
        for word in common_words:
            penalty += pred_token_counts[word]

    return penalty / len(pred_texts)
