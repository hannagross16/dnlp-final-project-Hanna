import argparse
import random

import numpy as np
import pandas as pd
import torch
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration
from optimizer import AdamW
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)
from multitask_classifier import get_args




TQDM_DISABLE = False



class SentencePairDataset_Generation(Dataset):
    def __init__(self, dataset, args,isRegression=False,max_length=256):
        self.dataset = dataset
        self.p = args
        self.isRegression = isRegression
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors="pt", padding=True, truncation=True,max_length=self.max_length,add_special_tokens=True)
        encoding2 = self.tokenizer(sent2, return_tensors="pt", padding=True, truncation=True,max_length=self.max_length,add_special_tokens=True)
        token_ids = torch.LongTensor(encoding1["input_ids"])
        attention_mask = torch.LongTensor(encoding1["attention_mask"])

        token_ids2 = torch.LongTensor(encoding2["input_ids"])
        attention_mask2 = torch.LongTensor(encoding2["attention_mask"])
        return (
            token_ids,
            attention_mask,
            token_ids2
        )

    def collate_fn(self, all_data):
        (
            token_ids,
            attention_mask,
            token_ids2,
        ) = self.pad_data(all_data)

        batched_data = (
            token_ids,
            attention_mask,
            token_ids2,
        )

        return batched_data
    
class SentencePairTestDataset_Generation(Dataset):
    def __init__(self, dataset, args,isRegression=False,max_length=256):
        self.dataset = dataset
        self.p = args
        self.isRegression = isRegression
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors="pt", padding=True, truncation=True,max_length=self.max_length,add_special_tokens=True)

        token_ids = torch.LongTensor(encoding1["input_ids"])
        attention_mask = torch.LongTensor(encoding1["attention_mask"])

        return (
            token_ids,
            attention_mask,
        )

    def collate_fn(self, all_data):
        (
            token_ids,
            attention_mask
        ) = self.pad_data(all_data)

        batched_data = (
            token_ids,
            attention_mask,
        )

        return batched_data

def transform_data(dataset, max_length=256):
    """
    Turn the data to the format you want to use.
    Use AutoTokenizer to obtain encoding (input_ids and attention_mask).
    Tokenize the sentence pair in the following format:
    sentence_1 + SEP + sentence_1 segment location + SEP + paraphrase types.
    Return Data Loader.
    # SEP from tokenizer not string, no strip/preprocess_string
    """
    ### TODO
    etpc_data = []
    if 'sentence2' in dataset.columns:
        for _, row in dataset.iterrows():
            try:
                #label = "".join(row["paraphrase_types"].strip("][").replace(' ',''))
                etpc_data.append((
                        row["sentence1"].strip("][") + tokenizer.sep_token + row["sentence1_segment_location"]+ tokenizer.sep_token + row["paraphrase_types"].strip("]["),
                        row["sentence2"].strip("]["),
                ))
                
            except:
                pass
        etpc_train_data = SentencePairDataset_Generation(etpc_data, args, isRegression = True, max_length=max_length)
        etpc_train_dataloader = DataLoader(
                etpc_train_data,
                shuffle=True,
                batch_size=32,#TODO
                collate_fn=etpc_train_data.collate_fn,
            )
        return etpc_train_dataloader
    
    else:
        for _, row in dataset.iterrows():
            try:
                #label = "".join(row["paraphrase_types"].strip("][").replace(' ',''))

                etpc_data.append(
                    (
                        row["sentence1"].strip("][") + tokenizer.sep_token + row["sentence1_segment_location"]+ tokenizer.sep_token + row["paraphrase_types"].strip("][")
                    )
                )
            except:
                pass
        etpc_test_data = SentencePairTestDataset_Generation(etpc_data, args)
        etpc_test_data.tokenizer = tokenizer
        etpc_test_data.tokenizer.model_max_length = max_length #TODO padding max len trunc true add special token True
        etpc_test_dataloader = DataLoader(
                etpc_test_data,
                shuffle=False,
                batch_size=1,
                collate_fn=etpc_test_data.collate_fn,
            )
        return etpc_test_dataloader


def train_model(model, train_data, dev_data, device, tokenizer):
    """
    Train the model. Return and save the model.
    """
    ### TODO
    num_epoch = 5
    lr = 1e-5
    optimizer = AdamW(model.parameters(), lr=lr)
    
    for epoch in range(num_epoch):
        model.train()
        train_loss = 0
        num_batches = 0

        for batch in tqdm(
            train_data, desc=f"train-{epoch+1:02}", disable=TQDM_DISABLE
        ):
            b_ids1, b_mask1, b_ids2 = batch
            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2= b_ids2.to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=b_ids1, attention_mask=b_mask1, labels=b_ids2)

            
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
        print("Loss:",train_loss)
        print("Train BLEU:",evaluate_model(model, train_data, device,tokenizer))
        print("Dev BLEU:",evaluate_model(model, dev_data, device,tokenizer))

    return model
    


def test_model(test_data, test_ids, device, model, tokenizer):
    """
    Test the model. Generate paraphrases for the given sentences (sentence1) and return the results
    in form of a Pandas dataframe with the columns 'id' and 'Generated_sentence2'.
    The data format in the columns should be the same as in the train dataset.
    Return this dataframe.
    """
    ### TODO
    model.eval()
    all_pred = []
    with torch.no_grad():
        for batch in test_data:
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=5,
                early_stopping=True,
            )
            pred_text = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in outputs
            ]
            all_pred.extend(pred_text)
    return pd.DataFrame({'id':test_ids, 'Generated_sentence2':all_pred})


def evaluate_model(model, test_data, device, tokenizer):
    """
    You can use your train/validation set to evaluate models performance with the BLEU score.
    """
    model.eval()
    bleu = BLEU()
    predictions = []
    references = []

    with torch.no_grad():
        for batch in test_data:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Generate paraphrases
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=5,
                early_stopping=True,
            )

            pred_text = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in outputs
            ]
            ref_text = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in labels
            ]

            predictions.extend(pred_text)
            references.extend(ref_text)
        print(predictions[0])
        print(references[0])
        
    model.train()
    # Calculate BLEU score
    bleu_score = bleu.corpus_score(predictions, [references])
    return bleu_score.score


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True




def finetune_paraphrase_generation(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    print("Using device:",device)

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", local_files_only=True)
    model.to(device)
    print("loaded")

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t")
    dev_dataset = pd.read_csv("data/etpc-paraphrase-dev.csv", sep="\t")
    test_dataset = pd.read_csv("data/etpc-paraphrase-generation-test-student.csv", sep="\t")

    # You might do a split of the train data into train/validation set here
    # ...

    train_data = transform_data(train_dataset)
    dev_data = transform_data(dev_dataset)
    test_data = transform_data(test_dataset)

    print(f"Loaded {len(train_dataset)} training samples.")

    model = train_model(model, train_data, dev_data, device, tokenizer)

    print("Training finished.")

    bleu_score = evaluate_model(model, dev_data, device, tokenizer)
    print(f"The BLEU-score of the model is: {bleu_score:.3f}")

    test_ids = test_dataset["id"]
    test_results = test_model(test_data, test_ids, device, model, tokenizer)
    test_results.to_csv(
        "predictions/bart/etpc-paraphrase-generation-test-output.csv", index=False, sep="\t"
    )


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_generation(args)
