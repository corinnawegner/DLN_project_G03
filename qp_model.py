import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.optim import AdamW
import warnings
import socket
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import ElectraTokenizer, ElectraModel
from sklearn.metrics import mean_squared_error
import quality_measure as qm

try:
    local_hostname = socket.gethostname()
except:
    local_hostname = None

DEV_MODE = False
if local_hostname == 'Corinna-PC' or local_hostname == "TABLET-TTS0K9R0": #Todo: Add also laptop
    DEV_MODE = True

TQDM_DISABLE = not DEV_MODE

batch_size = 64 if not DEV_MODE else 1

warnings.filterwarnings("ignore", category=FutureWarning)

class QualityPredictor(nn.Module):
    def __init__(self, model_name='google/electra-base-discriminator'):
        super(QualityPredictor, self).__init__()
        self.tokenizer = ElectraTokenizer.from_pretrained(model_name)
        self.model = ElectraModel.from_pretrained(model_name)
        self.regressor = nn.Linear(self.model.config.hidden_size * 2, 3)  # Output 3 dimensions: sem, syn, lex

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        # Get embeddings for both sentences
        outputs_1 = self.model(input_ids=input_ids_1, attention_mask=attention_mask_1)
        cls_output_1 = outputs_1.last_hidden_state[:, 0, :]  # [CLS] token

        outputs_2 = self.model(input_ids=input_ids_2, attention_mask=attention_mask_2)
        cls_output_2 = outputs_2.last_hidden_state[:, 0, :]  # [CLS] token

        # Concatenate the embeddings
        combined_output = torch.cat((cls_output_1, cls_output_2), dim=1)

        # Predict the quality scores
        quality_scores = self.regressor(combined_output)
        return quality_scores

class QualityDataset(Dataset):
    def __init__(self, sentences_1, sentences_2, qualities, tokenizer, max_length):
        self.sentences_1 = sentences_1
        self.sentences_2 = sentences_2
        self.qualities = qualities
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences_1)

    def __getitem__(self, idx):
        sentence_1 = self.sentences_1[idx]
        sentence_2 = self.sentences_2[idx]
        quality = self.qualities[idx]

        # Tokenize both sentences
        inputs_1 = self.tokenizer(sentence_1, return_tensors='pt', padding='max_length', truncation=True,
                                  max_length=self.max_length)
        inputs_2 = self.tokenizer(sentence_2, return_tensors='pt', padding='max_length', truncation=True,
                                  max_length=self.max_length)

        input_ids_1 = inputs_1['input_ids'].squeeze(0)  # Remove batch dimension
        attention_mask_1 = inputs_1['attention_mask'].squeeze(0)  # Remove batch dimension

        input_ids_2 = inputs_2['input_ids'].squeeze(0)  # Remove batch dimension
        attention_mask_2 = inputs_2['attention_mask'].squeeze(0)  # Remove batch dimension

        return {'input_ids_1': input_ids_1, 'attention_mask_1': attention_mask_1,
                'input_ids_2': input_ids_2, 'attention_mask_2': attention_mask_2,
                'labels': torch.tensor(quality, dtype=torch.float)}

def prepare_data(df):
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
    sentences_1 = df['sentence_1'].tolist()
    sentences_2 = df['sentence_2'].tolist()
    qualities = df[['quality_sem', 'quality_syn', 'quality_lex']].values.tolist()
    dataset = QualityDataset(sentences_1, sentences_2, qualities, tokenizer, max_length=32)
    return dataset

def train_model(model, train_loader, num_epochs=3):
    print("Training Quality predictor \n")
    model.train()

    qp_optimizer = AdamW(model.parameters(), lr=5e-5)
    qp_scheduler = StepLR(qp_optimizer, step_size=1, gamma=0.95)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in tqdm(train_loader, disable=TQDM_DISABLE):
            input_ids_1 = batch['input_ids_1']
            attention_mask_1 = batch['attention_mask_1']
            input_ids_2 = batch['input_ids_2']
            attention_mask_2 = batch['attention_mask_2']
            labels = batch['labels']

            qp_optimizer.zero_grad()
            outputs = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
            loss = criterion(outputs, labels)
            loss.backward()
            qp_optimizer.step()
            qp_scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')
    model.eval()
    #print("Quality predictor training finished")
    return model

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, disable=TQDM_DISABLE):
            input_ids_1 = batch['input_ids_1']
            attention_mask_1 = batch['attention_mask_1']
            input_ids_2 = batch['input_ids_2']
            attention_mask_2 = batch['attention_mask_2']
            labels = batch['labels']

            outputs = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    mse_sem = mean_squared_error(all_labels[:, 0], all_preds[:, 0])
    mse_syn = mean_squared_error(all_labels[:, 1], all_preds[:, 1])
    mse_lex = mean_squared_error(all_labels[:, 2], all_preds[:, 2])

    print(f'MSE for Semantics: {mse_sem:.4f}')
    print(f'MSE for Syntax: {mse_syn:.4f}')
    print(f'MSE for Lexical: {mse_lex:.4f}')

def predict_quality(model, sentence_pairs):
    model.eval()
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
    sentences_1, sentences_2 = zip(*sentence_pairs)
    inputs_1 = tokenizer(sentences_1, padding=True, truncation=True, return_tensors='pt')
    inputs_2 = tokenizer(sentences_2, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(inputs_1['input_ids'], inputs_1['attention_mask'],
                        inputs_2['input_ids'], inputs_2['attention_mask'])
    return outputs
def load_toy_data():
    # Example data
    data = {
        'sentence_1': [
            'The quick brown fox jumps over the lazy dog.',
            'A fast brown fox leaps over a lazy canine.',
            'The quick brown fox jumps over a sleepy dog.',
            'A speedy brown fox leaps over a lazy dog.',
            'The swift brown fox jumps over the lethargic dog.'
        ],
        'sentence_2': [
            'A fast brown fox leaps over a lazy canine.',
            'The quick brown fox jumps over the lazy dog.',
            'A speedy brown fox leaps over a lazy dog.',
            'The quick brown fox jumps over a sleepy dog.',
            'A speedy brown fox leaps over a lazy dog.'
        ],
        'quality_sem': [0.8, 0.7, 0.75, 0.65, 0.85],
        'quality_syn': [0.7, 0.6, 0.7, 0.6, 0.75],
        'quality_lex': [0.65, 0.55, 0.6, 0.5, 0.7]
    }
    df = pd.DataFrame(data)
    return df

def train_tset_split(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = prepare_data(train_df)
    test_dataset = prepare_data(test_df)
    return train_dataset, test_dataset

def load_etpc_paraphrase():
    dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t")
    list_s1 = []
    list_s2 = []
    list_sem = []
    list_syn = []
    list_lex = []
    for _, row in tqdm(dataset.iterrows(), total=len(dataset), disable=TQDM_DISABLE):
        sentence_1 = row['sentence1']
        sentence_2 = row['sentence2']
        list_s1.append(sentence_1)
        list_s2.append(sentence_2)
        qv = qm.quality_vector(sentence_1, sentence_2)
        list_sem.append(qv[0])
        list_syn.append(qv[1])
        list_lex.append(qv[2])
        # (qsem_normalized, qsyn_normalized, qlex_normalized)
    data = {
        'sentence_1': list_s1,
        'sentence_2': list_s2,
        'quality_sem': list_sem,
        'quality_syn': list_syn,
        'quality_lex': list_lex
    }
    df = pd.DataFrame(data)
    return df

"""
# Example inference
df_data = load_toy_data()
train_dataset = prepare_data(df_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

qpmodel = QualityPredictor()
qpmodel = train_model(qpmodel, train_loader, num_epochs=3)
sentence_pairs = [
    ('The quick brown fox jumps over the lazy dog.', 'A fast brown fox leaps over a lazy canine.'),
    ('The quick brown fox jumps over the lazy dog.', 'A speedy brown fox leaps over a lazy dog.'),
    ("He said the foodservice pie business doesn't fit the company's long-term growth strategy.", '"The foodservice pie business does not fit our long-term growth strategy.')
]
predictions = predict_quality(qpmodel, sentence_pairs)
print(predictions)
"""