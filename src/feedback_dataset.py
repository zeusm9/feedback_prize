import pandas as pd
import torch
from torch.utils.data import Dataset

class FeedbackDataset(Dataset):
    def __init__(self, file, tokenizer, max_len, is_test = False):
        self.data = pd.read_csv(file)
        self.is_test = is_test
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        text = self.data["full_text"].values
        labels = self.data[["cohesion","syntax","vocabulary","phraseology","grammar","conventions"]].values

        inputs = self.tokenize_text(text=text[idx])

        if not self.is_test:
            targets = torch.tensor(labels[idx], dtype=torch.float)
            return inputs, targets
        
        return inputs

    def tokenize_text(self, text):
        inputs = self.tokenizer.encode_plus(
            text,
            return_tensors = None,
            add_special_tokens = True,
            max_length = self.max_len,
            padding = 'max_length',
            truncation = True
        )
        for k,v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        return inputs