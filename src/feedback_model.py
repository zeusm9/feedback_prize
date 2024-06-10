import torch.nn as nn
import transformers

class FeedbackModel(nn.Module):
    def __init__(self, model_name, n_labels):
        super(FeedbackModel,self).__init__()
        self.model_name = model_name
        self.backbone = transformers.AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(1024, n_labels)
    
    def forward(self, input_dict):
        _, output = self.backbone(**input_dict, return_dict = False)
        output = self.drop(output)
        output = self.fc(output)
        return output