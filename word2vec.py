import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import EmailDataset
from torch.utils.data import DataLoader

file_path = "data/emails.train"
gpu = torch.device("cuda")
context_size = 1

email_data = EmailDataset(file_path, context_size)




