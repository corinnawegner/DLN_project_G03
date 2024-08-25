import random
from Quality_predictor import quality_measure
from optimizer import AdamW
import warnings
import socket
from transformers import BertModel, AutoTokenizer
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from bart_generation import perform_pos_ner
import spacy

# Load SpaCy model for POS tagging and NER
nlp = spacy.load("en_core_web_sm")

try:
    local_hostname = socket.gethostname()
except:
    local_hostname = None

DEV_MODE = False
if local_hostname == 'Corinna-PC' or local_hostname == "TABLET-TTS0K9R0":  # Todo: Add also laptop
    DEV_MODE = True

TQDM_DISABLE = not DEV_MODE

warnings.filterwarnings("ignore", category=FutureWarning)

r = random.randint(10000, 99999)
model_save_path = f"models/bart_generation_quality_control_{r}.pt"

class QualityPredictor(nn.Module):
    """
    A neural network model for predicting quality scores based on input text using a BERT-based transformer.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer for converting text into tokens.
        bert (BertModel): Pre-trained BERT model for extracting text representations.
        regressor (nn.Linear): Linear layer for regression output based on BERT hidden states.
    """
    def __init__(self, model_name, num_labels=3):
        """
            Initializes the QualityPredictor model.

            Parameters:
                model_name (str): The name of the pre-trained BERT model to use.
                num_labels (int): The number of output labels for the regression task.
        """
        super(QualityPredictor, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.regressor = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        """
            Performs a forward pass through the model.

            Parameters:
               input_ids (torch.Tensor): Input token IDs.
               attention_mask (torch.Tensor): Attention mask to avoid padding tokens.

            Returns:
               torch.Tensor: Predicted logits.
        """
        # Get hidden states from BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Pass through regression layer
        logits = self.regressor(outputs.pooler_output)
        return logits

    def predict(self, texts, device):
        """
            Makes predictions on a list of texts.

            Parameters:
               texts (list of str): List of text inputs to predict.
               device (torch.device): The device to which the model should be moved.

            Returns:
               torch.Tensor: Probability scores for each label.
        """
        self.eval()
        # Ensure the model is on the correct device
        self.to(device)

        # Tokenize and move inputs to the correct device
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the correct device

        with torch.no_grad():
            logits = self(inputs['input_ids'], inputs['attention_mask'])

        # Ensure logits are on the correct device
        logits = logits.to(device)
        probs = torch.sigmoid(logits)  # Apply sigmoid to ensure output is between 0 and 1
        return probs

def transform_data_qp(dataset, tokenizer, max_length=256):
    """
        Transforms the dataset into a DataLoader suitable for training.

        Parameters:
            dataset (pd.DataFrame): DataFrame containing input text and indices.
            tokenizer (AutoTokenizer): Tokenizer used to convert text into tokens.
            max_length (int): Maximum length of the tokenized sequences.

        Returns:
            DataLoader: DataLoader instance with the transformed dataset.
    """

    input_ids = []
    attention_masks = []
    indices = []

    SEP = tokenizer.sep_token

    for idx, row in tqdm(dataset.iterrows(), total=len(dataset), disable=False):
        sentence_1 = row['sentence1']
        segment_location_1 = row['sentence1_segment_location']
        paraphrase_type = row['paraphrase_types']

        pos_tags, entities = perform_pos_ner(sentence_1)
        pos_tags_str = ' '.join([f"{token}/{tag}" for token, tag in pos_tags])
        entities_str = ' '.join([f"{ent}/{label}" for ent, label in entities])

        combined_input = f"{sentence_1} {SEP} {segment_location_1} {SEP} {paraphrase_type} {SEP} POS: {pos_tags_str} {SEP} NER: {entities_str}"

        encoding = tokenizer(
            combined_input,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids.append(encoding['input_ids'].squeeze())
        attention_masks.append(encoding['attention_mask'].squeeze())
        indices.append(idx)

    dataset_tensor = TensorDataset(
        torch.stack(input_ids),
        torch.stack(attention_masks),
        torch.tensor(indices)
    )

    return DataLoader(dataset_tensor, batch_size=32, shuffle=True)

def train_quality_predictor(model, train_dataset, num_epochs, device):
    """
       Trains the QualityPredictor model on the provided dataset.

       Parameters:
           model (QualityPredictor): The model to be trained.
           train_dataset (pd.DataFrame): DataFrame containing training data.
           num_epochs (int): Number of epochs to train the model.
           device (torch.device): The device to which the model and data should be moved.

       Returns:
           QualityPredictor: The trained model.
    """
    train_data = transform_data_qp(train_dataset, tokenizer=model.tokenizer)
    model.to(device)
    model.train()

    input_texts = train_dataset["sentence1"].tolist()
    reference_texts = train_dataset["sentence2"].tolist()

    # Compute true vectors
    true_vectors = []
    for s_1, s_2 in zip(input_texts, reference_texts):
        true_vector = quality_measure.quality_vector(s_1, s_2)
        true_vectors.append(true_vector)

    # Initialize optimizer and scheduler
    qp_optimizer = AdamW(model.parameters(), lr=5e-4)
    qp_scheduler = StepLR(qp_optimizer, step_size=1, gamma=0.95)
    mse = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        model.train()  # Set model to training mode
        epoch_loss = 0  # Initialize epoch loss

        for batch in tqdm(train_data, disable=TQDM_DISABLE):
            input_ids, attention_mask, indices = [tensor.to(device) for tensor in batch]

            batch_indices = indices.tolist()
            true_vectors_batch = torch.tensor([true_vectors[idx] for idx in batch_indices], dtype=torch.float).to(device)

            qp_optimizer.zero_grad()  # Reset gradients

            # Forward pass using the model inside qpmodel
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Compute loss
            loss = mse(outputs, true_vectors_batch)
            loss.backward()  # Backpropagation
            qp_optimizer.step()  # Update model parameters
            qp_scheduler.step()  # Update learning rate

            epoch_loss += loss.item()  # Accumulate loss

        avg_loss = epoch_loss / len(train_data)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

    model.eval()  # Set model to evaluation mode
    print("Quality predictor training finished")
    return model

def load_and_train_qp_model(train_dataset, device):
    """
       Loads and trains a QualityPredictor model.

       Parameters:
           train_dataset (pd.DataFrame): DataFrame containing the training data.
           device (torch.device): The device to which the model should be moved.

       Returns:
           QualityPredictor: The trained QualityPredictor model.
       """

    model_name = "textattack/bert-base-uncased-yelp-polarity"
    num_labels = 3  # Since we are predicting three labels
    qpmodel = QualityPredictor(model_name=model_name, num_labels=num_labels)
    qpmodel.to(device)

    print('Training quality predictor')
    qpmodel = train_quality_predictor(qpmodel, train_dataset, device=device, num_epochs=20 if not DEV_MODE else 1)

    return qpmodel
