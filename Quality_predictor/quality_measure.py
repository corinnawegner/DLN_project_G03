import difflib
import torch
import warnings
import nltk
from nltk import word_tokenize, Tree

# Download necessary NLTK data files (run these lines once)
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
warnings.filterwarnings("ignore", category=FutureWarning)

def parse_sentence_simple(sentence):
    """Create a simple parse tree with each word treated as its own leaf."""
    words = sentence.split()
    parse_tree = Tree('S', [Tree('WORD', [word]) for word in words])
    return parse_tree

def tree_edit_distance_simple(tree1, tree2):
    """Compute the tree edit distance between two simple trees."""
    str_tree1 = tree1.pformat()
    str_tree2 = tree2.pformat()
    diff = difflib.ndiff(str_tree1, str_tree2)
    return sum(1 for _ in diff if _[0] in ('+', '-'))

def normalized_tree_edit_distance_simple(tree1, tree2):
    """Compute normalized tree edit distance for simple trees."""
    edit_distance = tree_edit_distance_simple(tree1, tree2)
    max_distance = max(len(tree1.pformat()), len(tree2.pformat()))
    return edit_distance / max_distance if max_distance > 0 else 0

def character_edit_distance_simple(str1, str2):
    """Compute character-level minimal edit distance."""
    seq_matcher = difflib.SequenceMatcher(None, str1, str2)
    return 1 - seq_matcher.ratio()

def normalized_character_edit_distance_simple(str1, str2):
    """Compute normalized character-level edit distance."""
    return character_edit_distance_simple(str1, str2)

def compute_bleurt_score_simple(reference, candidate):
    """Compute a simple BLEURT-like score based on basic similarity."""
    # Use a simple ratio of matching words as a proxy for semantic similarity
    ref_words = set(reference.split())
    cand_words = set(candidate.split())
    overlap = ref_words & cand_words
    return len(overlap) / max(len(ref_words), len(cand_words), 1)

def quality_vector(s, s_prime):
    """Compute the quality vector for two sentences without tokenizers."""

    # Semantic quality
    qsem = compute_bleurt_score_simple(s, s_prime)

    # Syntactic quality
    tree_s = parse_sentence_simple(s)
    tree_s_prime = parse_sentence_simple(s_prime)
    qsyn = normalized_tree_edit_distance_simple(tree_s, tree_s_prime)

    # Lexical quality
    qlex = normalized_character_edit_distance_simple(s, s_prime)

    # Normalize the scores to [0, 100]
    qsem_normalized = int(100 * qsem)/100
    qsyn_normalized = int(100 * (1 - qsyn))/100
    qlex_normalized = int(100 * (1 - qlex))/100

    return qsem_normalized, qsyn_normalized, qlex_normalized

"""
# Example usage
s = "The quick brown fox jumps over the lazy dog."
s_prime = "A fast, dark-colored fox leaps over a sleepy dog."

quality = quality_vector(s)
print(f"Quality Vector: {quality}")
"""