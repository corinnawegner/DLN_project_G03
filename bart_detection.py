import argparse
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartModel

from optimizer import AdamW
#from transformers import AdamW # (amin) deprecated ATTENTION!
#from torch.optim import AdamW

TQDM_DISABLE = False


class BartWithClassifier(nn.Module):
    def __init__(self, num_labels=7):
        super(BartWithClassifier, self).__init__()

        self.bart = BartModel.from_pretrained("facebook/bart-large", local_files_only=args.local_files_only) # (amin) changed local_files_only=True
        self.classifier = nn.Linear(self.bart.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None):
        # Use the BartModel to obtain the last hidden state
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]

        # Add an additional fully connected layer to obtain the logits
        logits = self.classifier(cls_output)

        # Return the probabilities
        probabilities = self.sigmoid(logits)
        return probabilities


def transform_data(dataset, max_length=512, batch_size=2, tokenizer_name="facebook/bart-large"): # (amin) add batch_size=2, tokenizer_name="facebook/bart-large"
    """
    dataset: pd.DataFrame

    Turn the data to the format you want to use.

    1. Extract the sentences from the dataset. We recommend using the already split
    sentences in the dataset.
    2. Use the AutoTokenizer from_pretrained to tokenize the sentences and obtain the
    input_ids and attention_mask.
    3. Currently, the labels are in the form of [2, 5, 6, 0, 0, 0, 0]. This means that
    the sentence pair is of type 2, 5, and 6. Turn this into a binary form, where the
    label becomes [0, 1, 0, 0, 1, 1, 0]. Be careful that the test-student.csv does not
    have the paraphrase_types column. You should return a DataLoader without the labels.
    4. Use the input_ids, attention_mask, and binary labels to create a TensorDataset.
    Return a DataLoader with the TensorDataset. You can choose a batch size of your
    choice.
    """

    # (amin) [

    #raise NotImplementedError

    # Extract sentences (problem implementing the suggestion)
    sentences1 = dataset['sentence1'].tolist()
    sentences2 = dataset['sentence2'].tolist()

    #sentences1 = dataset['sentence1_tokenized'].tolist()
    #sentences2 = dataset['sentence2_tokenized'].tolist()
    
    #print(sentences1)

    # Tokenize sentences
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    encoded = tokenizer(
        sentences1,
        sentences2,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    
    # Convert labels to binary form, if available
    if 'paraphrase_types' in dataset.columns: # since the test data does not have this column
        binary_labels = []
        for labels in dataset['paraphrase_types']:
            # print(type(labels), labels)
            binary_label = [0] * 7  # 7 classes
            for label in labels:
                if label != 0:  # Skip the padding zeros
                    binary_label[label - 1] = 1 # labels range is 1-7
            binary_labels.append(binary_label)
        
        binary_labels = torch.tensor(binary_labels)
    
        # Create TensorDataset with labels
        tensor_dataset = TensorDataset(input_ids, attention_mask, binary_labels)
    else:
        # Create TensorDataset without labels
        tensor_dataset = TensorDataset(input_ids, attention_mask)
    
    # Collate function to return dict (datasets.Dataset like) output instead of list (not compatible with the evaluation() function)
    def collate_fn(batch):
        input_ids = torch.stack([item[0] for item in batch])
        attention_mask = torch.stack([item[1] for item in batch])
        
        # Check if binary_labels are present in the batch
        if len(batch[0]) > 2:
            labels = torch.stack([item[2] for item in batch])
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

    # Create DataLoader
    #data_loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn) # uncomment if you want a dictionary
    data_loader = DataLoader(tensor_dataset, batch_size=batch_size) # , shuffle=True (nover put this, because of test data)
    
    return data_loader

    # (amin) ]

def train_model(model, train_data, dev_data, device, epochs=1, learning_rate=1e-5): # (amin) added  epochs=3, learning_rate=5e-5
    """
    Train the model. You can use any training loop you want. We recommend starting with
    AdamW as your optimizer. You can take a look at the SST training loop for reference.
    Think about your loss function and the number of epochs you want to train for.
    You can also use the evaluate_model function to evaluate the
    model on the dev set. Print the training loss, training accuracy, and dev accuracy at
    the end of each epoch.

    Return the trained model.
    """
    ### TODO

    # (amin) [
    # raise NotImplementedError

    optimizer = AdamW(model.parameters(), lr=learning_rate )
    best_dev_acc = float("-inf")

    model.to(device)

    # Run for the specified number of epochs
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        loss_fn = nn.BCELoss()

        for batch in tqdm(
            train_data, desc=f"train-{epoch+1:02}", disable=TQDM_DISABLE
        ):
            #b_ids, b_mask, b_labels = (
            #    batch["input_ids"],
            #    batch["attention_mask"],
            #    batch["labels"],
            #)
            b_ids, b_mask, b_labels = batch

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.float().to(device)

            optimizer.zero_grad() # what is the difference with model.zero_grad() # check later
            logits = model(input_ids=b_ids, attention_mask=b_mask)
            loss = loss_fn(logits, b_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
        
        train_loss = train_loss / num_batches

        train_acc = evaluate_model(model, train_data, device)
        dev_acc =  evaluate_model(model, dev_data, device)

        print(
            f"Epoch {epoch+1:02}: train loss :: {train_loss:.3f}, train :: {train_acc:.3f}, dev :: {dev_acc:.3f}"
        )

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            #save_model(model, optimizer, args, config, args.filepath)
    return model

    # (amin) ]


def test_model(model, test_data, test_ids, device):
    """
    Test the model. Predict the paraphrase types for the given sentences and return the results in form of
    a Pandas dataframe with the columns 'id' and 'Predicted_Paraphrase_Types'.
    The 'Predicted_Paraphrase_Types' column should contain the binary array of your model predictions.
    Return this dataframe.
    """
    ### TODO

    # (amin) [
    #raise NotImplementedError

    model.eval()
    model.to(device)

    predictions_list = []

    with torch.no_grad():
        for batch in tqdm(test_data):
            #b_input_ids = batch['input_ids'].to(device)
            #b_attention_mask = batch['attention_mask'].to(device)
            #b_ids = batch['ids']
            b_input_ids, b_attention_mask = batch # _ is for safety

            b_input_ids = b_input_ids.to(device)
            b_attention_mask = b_attention_mask.to(device)
            
            outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask)
            
            preds = (outputs > 0.5).int().cpu().numpy()

            predictions_list.extend(preds)

    # Create DataFrame
    results_df = pd.DataFrame({
        'id': test_ids,
        'Predicted_Paraphrase_Types': predictions_list
    })

    return results_df

    # (amin) ]


def evaluate_model(model, test_data, device):
    """
    This function measures the accuracy of our model's prediction on a given train/validation set
    We measure how many of the seven paraphrase types the model has predicted correctly for each data point.
    So, if the models prediction is [1,1,0,0,1,1,0] and the true label is [0,0,0,0,1,1,0], this predicition
    has an accuracy of 5/7, i.e. 71.4% .
    """
    all_pred = []
    all_labels = []
    model.eval()

    with torch.no_grad():
        for batch in test_data:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_labels = (outputs > 0.5).int()

            all_pred.append(predicted_labels)
            all_labels.append(labels)

    all_predictions = torch.cat(all_pred, dim=0)
    all_true_labels = torch.cat(all_labels, dim=0)

    true_labels_np = all_true_labels.cpu().numpy()
    predicted_labels_np = all_predictions.cpu().numpy()

    # Compute the accuracy for each label
    accuracies = []
    for label_idx in range(true_labels_np.shape[1]):
        correct_predictions = np.sum(
            true_labels_np[:, label_idx] == predicted_labels_np[:, label_idx]
        )
        total_predictions = true_labels_np.shape[0]
        label_accuracy = correct_predictions / total_predictions
        accuracies.append(label_accuracy)

    # Calculate the average accuracy over all labels
    accuracy = np.mean(accuracies)
    model.train()
    return accuracy


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=32)

    args = parser.parse_args()
    return args


def finetune_paraphrase_detection(args):
    model = BartWithClassifier()
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model.to(device)

    #train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t") # (amin) prefer to cast datatypes here
    #test_dataset = pd.read_csv("data/etpc-paraphrase-detection-test-student.csv", sep="\t") # (amin)

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t", converters={'paraphrase_types': pd.eval, 'sentence1_tokenized': pd.eval, 'sentence2_tokenized': pd.eval})
    test_dataset = pd.read_csv("data/etpc-paraphrase-detection-test-student.csv", sep="\t", converters={'sentence1_tokenized': pd.eval, 'sentence2_tokenized': pd.eval})

    # (amin) test with fewer data ATTENTION!
    #train_dataset = train_dataset.head(20)
    #test_dataset = test_dataset.head(10)
    
    # TODO You might do a split of the train data into train/validation set here
    # (or in the csv files directly)

    # (amin) [
    #train_data = transform_data(train_dataset)
    #test_data = transform_data(test_dataset)

    # Shuffle the DataFrame
    train_df = train_dataset.sample(frac=1).reset_index(drop=True)

    # Split the DataFrame into training and validation datasets
    train_frac = 0.8
    train_size = int(len(train_df) * train_frac)

    train_data = train_df.iloc[:train_size]
    dev_data = train_df.iloc[train_size:]

    train_data = transform_data(train_data, batch_size=args.train_batch_size) # (amin) change train_dataset -> train_data
    dev_data = transform_data(dev_data, batch_size=args.val_batch_size)
    test_data = transform_data(test_dataset, batch_size=args.test_batch_size)

    # (amin) ]

    print(f"Loaded {len(train_dataset)} training samples.")

    model = train_model(model, train_data, dev_data, device, epochs=args.epochs)

    print("Training finished.")

    accuracy = evaluate_model(model, dev_data, device)
    print(f"The accuracy of the model is: {accuracy:.3f}")

    test_ids = test_dataset["id"]
    test_results = test_model(model, test_data, test_ids, device)

    #print('Still alive!') # (amin) because of wierd KILLED message ATTENTION!
    test_results.to_csv(
        "predictions/bart/etpc-paraphrase-detection-test-output.csv", index=False, sep="\t"
    )


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_detection(args)

