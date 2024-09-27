import torch
from torch.utils.data import Dataset


class MultiLabelDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        texts,
        labelT=None,
        labelN=None,
        labelM=None,
        max_length=512,
        train=True,
    ):
        self.texts = texts
        self.inputs = tokenizer(
            texts, max_length=max_length, padding="max_length", truncation=True
        )
        if train:
            labelT2id = {1: 0, 2: 1, 3: 2, 4: 3}
            self.labelT = [labelT2id[v] for v in labelT]
            self.labelN = labelN
            self.labelM = labelM
        self.train = train

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = torch.tensor(self.inputs["input_ids"][idx], dtype=torch.long)
        mask = torch.tensor(self.inputs["attention_mask"][idx], dtype=torch.long)

        if self.train:
            labelT = torch.tensor(self.labelT[idx], dtype=torch.int64)
            labelN = torch.tensor(self.labelN[idx], dtype=torch.int64)
            labelM = torch.tensor(self.labelM[idx], dtype=torch.int64)
            return {
                "ids": ids,
                "mask": mask,
                "labelT": labelT,
                "labelN": labelN,
                "labelM": labelM,
            }
        else:
            return {"ids": ids, "mask": mask}
