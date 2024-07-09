import argparse
import os
import pickle
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_bert_cls_embedding(sentences: List[str]) -> torch.Tensor:
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)

    tokens = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**tokens)

    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.detach().cpu()

def split_and_apply(lst: List[str], batch_size: int) -> torch.Tensor:
    sublists = [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]
    results = []
    for sublist in tqdm(sublists, desc="Processing batches"):
        results.append(get_bert_cls_embedding(sublist))
    return torch.cat(results, dim=0)

def process_datasubset(datasubset: List[Dict[str, Any]], subset_type: str, le: preprocessing.LabelEncoder, output_folder: str, batch_size: int) -> preprocessing.LabelEncoder:
    bios = [bio["hard_text"] for bio in datasubset]
    text_labels = [bio["p"] for bio in datasubset]
    gender_list = [0 if bio["g"] == "m" else 1 for bio in datasubset]

    if subset_type == "train":
        le.fit(text_labels)

    labels = le.transform(text_labels)
    input_ids = split_and_apply(bios, batch_size)

    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, f"{subset_type}_labels.pkl"), "wb") as fp:
        pickle.dump(labels, fp)
    with open(os.path.join(output_folder, f"{subset_type}_gender_list.pkl"), "wb") as fp:
        pickle.dump(gender_list, fp)
    np.save(os.path.join(output_folder, f'{subset_type}_input_ids.npy'), input_ids.numpy())

    return le

def process_data(input_folder: str, output_folder: str, batch_size: int):
    train_data = pd.read_pickle(os.path.join(input_folder, "train.pickle"))
    dev_data = pd.read_pickle(os.path.join(input_folder, "dev.pickle"))
    test_data = pd.read_pickle(os.path.join(input_folder, "test.pickle"))

    le = preprocessing.LabelEncoder()

    le = process_datasubset(train_data, "train", le, output_folder, batch_size)
    process_datasubset(dev_data, "dev", le, output_folder, batch_size)
    process_datasubset(test_data, "test", le, output_folder, batch_size)

def main():
    parser = argparse.ArgumentParser(description='BERT Embedding for Profession Classification')
    parser.add_argument('--input_folder', type=str, default="../data/biasbios/", help='Path to the input data folder')
    parser.add_argument('--output_folder', type=str, default="../data/biasbios/", help='Path to the output data folder')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for processing with BERT')
    args = parser.parse_args()

    process_data(args.input_folder, args.output_folder, args.batch_size)

if __name__ == '__main__':
    main()