from transformers import BertTokenizer
import multitask_classifier
import torch
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def predict_paraphrase(self, sentence1, sentence2, tokenizer, device):
    self.eval()
    with torch.no_grad():
        # Tokenize input sentences
        inputs = tokenizer(sentence1, sentence2, return_tensors="pt", padding=True, truncation=True)

        # Extract tensors from tokenizer outputs
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Handle the case with pairs of tensors
        if len(input_ids.size()) == 2:  # Single pair of sentences
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = self.predict_paraphrase(input_ids, attention_mask, input_ids, attention_mask)
        else:
            input_ids_1, input_ids_2 = input_ids[0], input_ids[1]
            attention_mask_1, attention_mask_2 = attention_mask[0], attention_mask[1]

            input_ids_1 = input_ids_1.unsqueeze(0).to(device)
            input_ids_2 = input_ids_2.unsqueeze(0).to(device)
            attention_mask_1 = attention_mask_1.unsqueeze(0).to(device)
            attention_mask_2 = attention_mask_2.unsqueeze(0).to(device)

            logits = self.predict_paraphrase(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)

        # Apply sigmoid to get probability
        probability = torch.sigmoid(logits).item()

        return probability

def check_paraphrase(model_path, sentence1, sentence2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    saved = torch.load(model_path)
    config = saved["model_config"]

    model = multitask_classifier.MultitaskBERT(config)
    model.load_state_dict(saved["model"])
    model = model.to(device)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    probability = predict_paraphrase(model, sentence1, sentence2, tokenizer, device)

    return probability

model_path = f"models/pretrain-10-0.001-sst.pt"

#sentence1 = "How do I use this code to get a response if two input sentences are paraphrases?"
#sentence2 = "How can I determine if two sentences mean the same using this code?"
#probability = check_paraphrase(model_path, sentence1, sentence2)
#print(f"Probability that the sentences are paraphrases: {probability:.4f}")
