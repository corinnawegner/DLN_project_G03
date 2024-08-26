import torch
import warnings
import spacy
from datasets import load_metric
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
import Levenshtein

nlp = spacy.load('en_core_web_sm')

warnings.filterwarnings("ignore", category=FutureWarning)

def compute_bleurt_score(references, candidates):
    """Compute BLEURT score for semantic similarity."""

    config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20-D12')
    model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20-D12')
    tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20-D12')

    model.eval()
    with torch.no_grad():
        inputs = tokenizer(references, candidates, padding='longest', return_tensors='pt')
        res = model(**inputs).logits.flatten().tolist()
    return res

def word_level_levenshtein_distance(str1, str2):
    """Compute Levenshtein distance at the word level."""
    words1 = str1.split()
    words2 = str2.split()
    return Levenshtein.distance(' '.join(words1), ' '.join(words2))

def quality_vector(s, s_prime):
    """Compute the quality vector for two sentences."""
    # Semantic quality
    qsem = compute_bleurt_score(s, s_prime)[0]

    # Syntactic quality
    metric = load_metric("Quality_predictor/syntdiv_measure", trust_remote_code=True)
    qsyn = metric.compute(predictions=[s], references=[s_prime])['scores'][0]

    # Lexical quality
    max_word_length = max(len(s.split()), len(s_prime.split()))
    word_level_lev = word_level_levenshtein_distance(s, s_prime)
    qlex = word_level_lev / max_word_length if max_word_length > 0 else 0
    qlex = qlex

    return qsem, qsyn, qlex


# Example usage
s = "The quick brown fox jumps over the lazy dog."
s_prime = "A fast, dark-colored fox leaps over a sleepy dog."

quality = quality_vector(s, s_prime)
print(f"Quality Vector: {quality}")