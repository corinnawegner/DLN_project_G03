import spacy
import numpy as np
import pandas as pd

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

import spacy
import numpy as np
import pandas as pd

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def adjacency_matrix(sentence): #
    # Process the sentence with spaCy
    doc = nlp(sentence)
    num_words = len(doc)

    label_set = nlp.get_pipe('parser').labels  #Take all possible labels
    num_labels = len(label_set)
    label_to_idx = {label: idx for idx, label in enumerate(label_set)}

    adj_matrices = np.zeros((num_labels, num_words, num_words), dtype=int)

    # Fill the adjacency matrices based on dependency relations
    for token in doc:
        if token.head.i != token.i:  # Exclude self-dependencies
            head_idx = token.head.i
            dependent_idx = token.i
            label = token.dep_
            label_idx = label_to_idx[label]
            adj_matrices[label_idx, head_idx, dependent_idx] = 1

    return adj_matrices

# Print the results
#print("Dependency Tree:")
#for token in doc:
 #   print(f"{token.text} --> {token.head.text} (dep: {token.dep_})")

#print("\nAdjacency Matrix:")
#print(adj_matrix_df)

#sentence1 = "The cat is on the mat"
#sentence2 = "Tracks the maximum number of dependency labels across all sentences."
#A = adjacency_matrix(sentence1)
#print(A)
#doc = nlp(sentence2)
#print({token.dep_ for token in doc})