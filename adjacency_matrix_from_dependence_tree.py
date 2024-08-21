import spacy
import numpy as np

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

important_dependency_labels = (
    'ROOT', 'nsubj', 'dobj', 'amod', 'prep'
)

def give_num_labels():
    return len(important_dependency_labels)

def adjacency_matrix(sentence):
    # Process the sentence with spaCy
    doc = nlp(sentence)
    num_words = len(doc)

    # Create label set and map labels to indices
    label_set = important_dependency_labels
    num_labels = len(label_set)
    print(f"give_num_labels: {give_num_labels()}")
    label_to_idx = {label: idx for idx, label in enumerate(label_set)}

    adj_matrices = np.zeros((num_labels, num_words, num_words), dtype=int)

    # Fill the adjacency matrices based on dependency relations
    for token in doc:
        if token.head.i != token.i:  # Exclude self-dependencies
            head_idx = token.head.i
            dependent_idx = token.i
            label = token.dep_
            if label in label_to_idx:  # Check if label is in the set
                label_idx = label_to_idx[label]
                adj_matrices[label_idx, head_idx, dependent_idx] = 1

    total_sum = np.sum(adj_matrices)
    print(f"Sum of all values in adjacency matrices: {total_sum}")

    if np.all(adj_matrices == 0):
        print("Warning: All adjacency matrices are zeros. No dependencies found.")

    return adj_matrices

# Test with a sentence
#sentence1 = "The cat is on the mat"
#A = adjacency_matrix(sentence1)
#print(A)
