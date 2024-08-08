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
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Handle the case with pairs of tensors
        if len(input_ids.size()) == 2:  # Single pair of sentences
            logits = self.predict_paraphrase(input_ids, attention_mask, input_ids, attention_mask)
        else:
            input_ids_1, input_ids_2 = input_ids[0].unsqueeze(0), input_ids[1].unsqueeze(0)
            attention_mask_1, attention_mask_2 = attention_mask[0].unsqueeze(0), attention_mask[1].unsqueeze(0)

            logits = self.predict_paraphrase(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)

        # Apply sigmoid to get probability
        probability = torch.sigmoid(logits).item()

        return probability

def check_paraphrase(model_path, sentence1, sentence2):
    # Assuming the class has an attribute named "config" to store the configuration
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    if torch.cuda.is_available():
        evaluator = multitask_classifier.MultitaskBERT(torch.load(model_path)["model_config"])
        evaluator.load_state_dict(torch.load(model_path)["model"])

    else:
        evaluator = multitask_classifier.MultitaskBERT(
            torch.load(model_path, map_location=torch.device('cpu'))["model_config"])
        evaluator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))["model"])

    probability = predict_paraphrase(evaluator, sentence1, sentence2, tokenizer, device)
    return probability

model_path = f"models/finetune-10-1e-05-qqp.pt"

sentence1 = "How do I use this code to get a response if two input sentences are paraphrases?"
sentence2 = "How can I determine if two sentences mean the same using this code?"

sentence1 = "AAAA"
sentence2 = "AAAA"
probability = check_paraphrase(model_path, sentence1, sentence2)
print(f"Probability that the sentences are paraphrases: {probability:.4f}")
