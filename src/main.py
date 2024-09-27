import pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer
import fire
import datetime

# from utils import load_dataset
from train import MultiLabelBERTTester


def main(
    pretrained_model: str = "model",
    input_path: str = "data/tnm_report.csv",
    option_dic_path: str = "data/MANBYO_202106_dic-utf8.dic",
    seed: int = 2023,
    device: str = torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
    batch_size: int = 32,
    max_length: int = 512,
):
    # Read Data
    # df = load_dataset(train_data_dir)
    df = pd.read_csv(input_path)
    # df["TNM"] = df["T"].astype(str) + df["N"].astype(str) + df["M"].astype(str)
    # labelTNM2id = {v: i for i, v in enumerate(df["TNM"].unique())}

    # TOKENIZER = AutoTokenizer.from_pretrained(pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model, **{"mecab_kwargs": {"mecab_option": option_dic_path}}
    )
    tester = MultiLabelBERTTester(
        model_name=pretrained_model, tokenizer=tokenizer, device=device, seed=seed
    )
    tester.load_model(pretrained_model)
    predsTNM = tester.inferencing(df=df, batch_size=batch_size, max_length=max_length)
    df["TNM_pred"] = predsTNM

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    df.to_csv(f"outputs/predicted_{timestamp}.csv", index=False)


if __name__ == "__main__":
    fire.Fire(main)
