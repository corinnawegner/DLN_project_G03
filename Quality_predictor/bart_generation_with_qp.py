import random
from Quality_predictor import quality_measure
from optimizer import AdamW
import warnings
import socket
import spacy
from transformers import BertModel, AutoTokenizer
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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

batch_size = 64 if not DEV_MODE else 1

warnings.filterwarnings("ignore", category=FutureWarning)

r = random.randint(10000, 99999)
model_save_path = f"models/bart_generation_quality_control_{r}.pt"

hyperparams = {
    'optimizer': AdamW,
    'learning_rate': 1e-5,
    'batch_size': 64,
    'dropout_rate': 0.1,
    'patience': 3,
    'num_epochs': 100 if not DEV_MODE else 10,
    'alpha': 0.0,
    'scheduler': "CosineAnnealingLR",
    'POS_NER_tagging': True  # Set to True to enable POS and NER tagging
}


class QualityPredictor(nn.Module):
    def __init__(self, model_name, num_labels=3):
        super(QualityPredictor, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.regressor = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # Get hidden states from BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Pass through regression layer
        logits = self.regressor(outputs.pooler_output)
        return logits

    def predict(self, texts, device):
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
    input_ids = []
    attention_masks = []
    indices = []

    for idx, row in tqdm(dataset.iterrows(), total=len(dataset), disable=False):
        sentence = row['sentence1']

        encoding = tokenizer(
            sentence,
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
    qp_optimizer = AdamW(model.parameters(), lr=5e-3)  # Find optimal learning rate later
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
    #df_train_data_qp = bart_generation.transform_data(train_dataset)

    model_name = "textattack/bert-base-uncased-yelp-polarity"
    num_labels = 3  # Since we are predicting three labels
    qpmodel = QualityPredictor(model_name=model_name, num_labels=num_labels)
    qpmodel.to(device)

    print('Training quality predictor')
    qpmodel = train_quality_predictor(qpmodel, train_dataset, device=device, num_epochs=10 if not DEV_MODE else 1)

    return qpmodel
