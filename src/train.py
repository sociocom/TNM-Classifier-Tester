import numpy as np
from torch.utils.data import DataLoader
from utils import set_seed
from dataset import *
from model import *


class MultiLabelBERTTester:
    def __init__(self, model_name, tokenizer, device, seed=2023):
        set_seed(seed)
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.device = device

    def load_model(self, model_path):
        self.model = BERTMultiLabelModel(model_name=self.model_name).to(self.device)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()

    def inferencing(self, df, batch_size=32, max_length=512):
        test_dataset = MultiLabelDataset(
            self.tokenizer,
            texts=df["text"].to_list(),
            max_length=max_length,
            train=False,
        )
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False
        )

        predsT, predsN, predsM = np.empty(0), np.empty(0), np.empty(0)
        with torch.no_grad():
            for step, batch in enumerate(test_loader):
                ids = batch["ids"].to(self.device)
                mask = batch["mask"].to(self.device)

                outputT, outputN, outputM = self.model(ids, mask)
                predsT = np.concatenate(
                    (predsT, outputT.cpu().detach().numpy().argmax(axis=1))
                )
                predsN = np.concatenate(
                    (predsN, outputN.cpu().detach().numpy().argmax(axis=1))
                )
                predsM = np.concatenate(
                    (predsM, outputM.cpu().detach().numpy().argmax(axis=1))
                )

        id2labelT = {0: 1, 1: 2, 2: 3, 3: 4}
        predsT = [id2labelT[v] for v in predsT]

        preds = [
            str(int(t)) + str(int(n)) + str(int(m))
            for t, n, m in zip(predsT, predsN, predsM)
        ]

        return preds
