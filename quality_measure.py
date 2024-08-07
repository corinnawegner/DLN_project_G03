import numpy as np
import difflib
import torch
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
import warnings
import nltk
from nltk import word_tokenize, Tree

# Download necessary NLTK data files (run these lines once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
warnings.filterwarnings("ignore", category=FutureWarning)

def parse_sentence(sentence):
    """Parse sentence into a constituency parse tree."""
    # Tokenize and tag part-of-speech
    tokens = word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    # Create a simple POS-based parse tree
    # Todo: Note: This is a placeholder for real parsing; a more sophisticated parser is recommended
    parse_tree = Tree('S', [Tree(pos, [token]) for token, pos in pos_tags])
    return parse_tree

def tree_edit_distance(tree1, tree2):
    """Compute the tree edit distance between two trees."""
    # Convert trees to string representations
    str_tree1 = tree1.pformat()
    str_tree2 = tree2.pformat()
    # Use difflib to compute the edit distance
    diff = difflib.ndiff(str_tree1, str_tree2)
    return sum(1 for _ in diff if _[0] in ('+', '-'))


def normalized_tree_edit_distance(tree1, tree2):
    """Compute normalized tree edit distance."""
    edit_distance = tree_edit_distance(tree1, tree2)
    max_distance = max(len(tree1.pformat()), len(tree2.pformat()))
    return edit_distance / max_distance if max_distance > 0 else 0


def character_edit_distance(str1, str2):
    """Compute character-level minimal edit distance."""
    seq_matcher = difflib.SequenceMatcher(None, str1, str2)
    return 1 - seq_matcher.ratio()


def normalized_character_edit_distance(words1, words2):
    """Compute normalized character-level edit distance between bag-of-words."""
    str1 = ''.join(words1)
    str2 = ''.join(words2)
    return character_edit_distance(str1, str2)


def compute_bleurt_score(reference, candidate, model, tokenizer):
    """Compute BLEURT score for semantic similarity."""
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(reference, candidate, padding='longest', return_tensors='pt')
        scores = model(**inputs).logits.flatten().tolist()
    return scores[0]


def quality_vector(s, s_prime):
    """Compute the quality vector for two sentences."""
    # Load BLEURT model and tokenizer
    config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20-D12')
    model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20-D12')
    tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20-D12')

    # Semantic quality
    qsem = compute_bleurt_score([s], [s_prime], model, tokenizer)

    # Syntactic quality
    tree_s = parse_sentence(s)
    tree_s_prime = parse_sentence(s_prime)
    qsyn = normalized_tree_edit_distance(tree_s, tree_s_prime)

    # Lexical quality
    words_s = word_tokenize(s)
    words_s_prime = word_tokenize(s_prime)
    qlex = normalized_character_edit_distance(words_s, words_s_prime)

    # Normalize the scores to [0, 100]
    qsem_normalized = 100 * (1 / (1 + np.exp(-qsem)))  # Sigmoid normalization to [0, 100]
    qsyn_normalized = 100 * (1 - qsyn)
    qlex_normalized = 100 * (1 - qlex)

    return (qsem_normalized, qsyn_normalized, qlex_normalized)

"""
# Example usage
s = "The quick brown fox jumps over the lazy dog."
s_prime = "A fast, dark-colored fox leaps over a sleepy dog."

quality = quality_vector(s, s_prime)
print(f"Quality Vector: {quality}")
"""
