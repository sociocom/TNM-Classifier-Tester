import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import torch


def load_dataset(data_dir, label_file_name='label.csv'):
    df = pd.read_csv(data_dir.joinpath(label_file_name))
    df['text'] = None
    for idx, id in enumerate(df['ID']):
        text = Path(data_dir).joinpath(f'{id}.txt').read_text()
        df.loc[idx, 'text'] = text
    return df

def set_seed(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def display_score(labels, preds):
    f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    print(f'TNM | macro F1: {f1:.4f} Accuracy: {acc:.4f}')
