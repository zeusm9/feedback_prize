import torch
import platform
import transformers
from feedback_dataset import FeedbackDataset
from feedback_model import FeedbackModel
from trainer import Trainer
from torch.utils.data import DataLoader

def main():

    MODEL_NAME = 'roberta-large'
    N_LABELS = 6

    if torch.cuda.is_available():
        print("\n[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
        DEVICE = torch.device('cuda:0')
    else:
        print("\n[INFO] GPU not found. Using CPU: {}\n".format(platform.processor()))
        DEVICE = torch.device('cpu')
    
    file_train = "/home/matteocana/projects/my_projects/feedback_prize/data/train.csv"
    file_test = "/home/matteocana/projects/my_projects/feedback_prize/data/test.csv"

    tokenizer = transformers.AutoTokenizer.from_pretrained('roberta-large', use_fast=True)

    fs_train = FeedbackDataset(file=file_train, tokenizer=tokenizer, max_len = 200, is_test=False)
    fs_test = FeedbackDataset(file=file_test, tokenizer=tokenizer, max_len=200, is_test=True)

    train_loader = DataLoader(dataset=fs_train, batch_size= 8, shuffle=True)
    test_loader = DataLoader(dataset=fs_test, batch_size=8, shuffle=True)

    feedback_model = FeedbackModel(model_name=MODEL_NAME, n_labels=N_LABELS)

    trainer = Trainer(dataloaders=(train_loader,test_loader), model=feedback_model)
    trainer.train_one_epoch()

if __name__ == "__main__":
    main()