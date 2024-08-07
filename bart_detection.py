import argparse
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, BartModel
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large',local_files_only=True)
from optimizer import AdamW
from datasets import (
    SentencePairDataset,SentencePairTestDataset)
from multitask_classifier import get_args


TQDM_DISABLE = False


class BartWithClassifier(nn.Module):
    def __init__(self, num_labels=7):
        super(BartWithClassifier, self).__init__()

        self.bart = BartModel.from_pretrained("facebook/bart-large", local_files_only=True)
        for param in self.bart.parameters():
            param.requires_grad = False
            
        self.classifier = nn.Linear(self.bart.config.hidden_size*2, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_id1, attention_mask1, input_id2, attention_mask2):
        # Use the BartModel to obtain the last hidden state
        embedding_1 = self.bart(input_ids=input_id1, attention_mask=attention_mask1)["last_hidden_state"][:,0]
        embedding_2 = self.bart(input_ids=input_id2, attention_mask=attention_mask2)["last_hidden_state"][:,0]
        logits = self.classifier(torch.concatenate((embedding_1,embedding_2),dim=1))
        probabilities = self.sigmoid(logits)
        return probabilities


def preprocess_string(s):
    return " ".join(
        s.lower()
        .replace(".", " .")
        .replace("?", " ?")
        .replace(",", " ,")
        .replace("'", " '")
        .split()
    )

def transform_data(dataset, max_length=512):
    """
    dataset: pd.DataFrame

    Turn the data to the format you want to use.

    1. Extract the sentences from the dataset. We recommend using the already split
    sentences in the dataset.
    2. Use the AutoTokenizer from_pretrained to tokenize the sentences and obtain the
    input_ids and attention_mask.
    3. Currently, the labels are in the form of [2, 5, 6, 0, 0, 0, 0]. This means that
    the sentence pair is of type 2, 5, and 6. Turn this into a binary form, where the
    label becomes [0, 1, 0, 0, 1, 1, 0]. Be careful that the test-student.csv does not
    have the paraphrase_types column. You should return a DataLoader without the labels.
    4. Use the input_ids, attention_mask, and binary labels to create a TensorDataset.
    Return a DataLoader with the TensorDataset. You can choose a batch size of your
    choice.
    """
    etpc_data = []
    if 'paraphrase_types' in dataset.columns:
        for _, row in dataset.iterrows():
            try:
                sent_id = row["id"].lower().strip()
                label = np.zeros(7)
                dat = np.array(row["paraphrase_types"].strip("][").split(", "),dtype=int)
                label[dat[dat!=0]-1] = 1
                label = list(label)
                etpc_data.append(
                    (
                        preprocess_string(row["sentence1"]),
                        preprocess_string(row["sentence2"]),
                        label,
                        sent_id,
                    )
                )
            except:
                pass
        etpc_train_data = SentencePairDataset(etpc_data, args, isRegression = True)
        etpc_train_data.tokenizer.model_max_length = max_length
        etpc_train_dataloader = DataLoader(
                etpc_train_data,
                shuffle=True,
                batch_size=args.batch_size,
                collate_fn=etpc_train_data.collate_fn,
            )
        return etpc_train_dataloader
    
    else:
        for _, row in dataset.iterrows():
            try:
                sent_id = row["id"].lower().strip()
                etpc_data.append(
                    (
                        preprocess_string(row["sentence1"]),
                        preprocess_string(row["sentence2"]),
                        sent_id,
                    )
                )
            except:
                pass
        etpc_test_data = SentencePairTestDataset(etpc_data, args)
        etpc_test_data.tokenizer.model_max_length = max_length
        etpc_test_dataloader = DataLoader(
                etpc_test_data,
                shuffle=True,
                batch_size=1,
                collate_fn=etpc_test_data.collate_fn,
            )
        return etpc_test_dataloader



def train_model(model, train_data, dev_data, device):
    """
    Train the model. You can use any training loop you want. We recommend starting with
    AdamW as your optimizer. You can take a look at the SST training loop for reference.
    Think about your loss function and the number of epochs you want to train for.
    You can also use the evaluate_model function to evaluate the
    model on the dev set. Print the training loss, training accuracy, and dev accuracy at
    the end of each epoch.

    Return the trained model.
    """
    num_epoch = 10
    lr = 3e-4
    optimizer = AdamW(model.parameters(), lr=lr)
    
    for epoch in range(num_epoch):
        model.train()
        train_loss = 0
        num_batches = 0

        for batch in tqdm(
            train_data, desc=f"train-{epoch+1:02}", disable=TQDM_DISABLE
        ):
            b_ids1, b_mask1,b_ids2, b_mask2, b_labels = (
                    batch["token_ids_1"],
                    batch["attention_mask_1"],
                    batch["token_ids_2"],
                    batch["attention_mask_2"],
                    batch["labels"],
            )

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2= b_ids2.to(device)
            b_mask2 = b_mask2.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()

            probabilities = model(b_ids1, b_mask1, b_ids2, b_mask2)
            loss = F.binary_cross_entropy(probabilities.squeeze(), b_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
        print("Loss:",train_loss)
        print("Train acc:",evaluate_model(model, train_data, device))
        print("Dev acc:",evaluate_model(model, dev_data, device))
        
    return model
    ### TODO
    raise NotImplementedError



def test_model(model, test_data, test_ids, device):
    """
    Test the model. Predict the paraphrase types for the given sentences and return the results in form of
    a Pandas dataframe with the columns 'id' and 'Predicted_Paraphrase_Types'.
    The 'Predicted_Paraphrase_Types' column should contain the binary array of your model predictions.
    Return this dataframe.
    """
    ### TODO
    all_pred = []
    all_ids = []
    with torch.no_grad():
        for batch in test_data:
            input_id1, attention_mask1,input_id2, attention_mask2, test_ids = (
                    batch["token_ids_1"],
                    batch["attention_mask_1"],
                    batch["token_ids_2"],
                    batch["attention_mask_2"],
                    batch["sent_ids"]
                                            )
            input_id1 = input_id1.to(device)
            attention_mask1 = attention_mask1.to(device)
            input_id2 = input_id2.to(device)
            attention_mask2 = attention_mask2.to(device)

            outputs = model(input_id1, attention_mask1, input_id2, attention_mask2)
            predicted_labels = (outputs > 0.5).int()

            all_pred.append(predicted_labels.detach().cpu().numpy().tolist()[0])
            all_ids.append(test_ids[0])
    return pd.DataFrame({'id':all_ids, 'Predicted_Paraphrase_Types':all_pred})



def evaluate_model(model, test_data, device):
    """
    This function measures the accuracy of our model's prediction on a given train/validation set
    We measure how many of the seven paraphrase types the model has predicted correctly for each data point.
    So, if the models prediction is [1,1,0,0,1,1,0] and the true label is [0,0,0,0,1,1,0], this predicition
    has an accuracy of 5/7, i.e. 71.4% .
    """
    all_pred = []
    all_labels = []
    model.eval()

    with torch.no_grad():
        for batch in test_data:
            input_id1, attention_mask1,input_id2, attention_mask2, labels = (
                    batch["token_ids_1"],
                    batch["attention_mask_1"],
                    batch["token_ids_2"],
                    batch["attention_mask_2"],
                    batch["labels"],
            )
            input_id1 = input_id1.to(device)
            attention_mask1 = attention_mask1.to(device)
            input_id2 = input_id2.to(device)
            attention_mask2 = attention_mask2.to(device)

            outputs = model(input_id1, attention_mask1, input_id2, attention_mask2)
            predicted_labels = (outputs > 0.5).int()

            all_pred.append(predicted_labels)
            all_labels.append(labels)

    all_predictions = torch.cat(all_pred, dim=0)
    all_true_labels = torch.cat(all_labels, dim=0)

    true_labels_np = all_true_labels.cpu().numpy()
    predicted_labels_np = all_predictions.cpu().numpy()

    # Compute the accuracy for each label
    accuracies = []
    for label_idx in range(true_labels_np.shape[1]):
        correct_predictions = np.sum(
            true_labels_np[:, label_idx] == predicted_labels_np[:, label_idx]
        )
        total_predictions = true_labels_np.shape[0]
        label_accuracy = correct_predictions / total_predictions
        accuracies.append(label_accuracy)

    # Calculate the average accuracy over all labels
    accuracy = np.mean(accuracies)
    model.train()
    return accuracy


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


"""def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()
    return args
"""

def finetune_paraphrase_detection(args):

    model = BartWithClassifier()
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    print(sum(p.numel() for p in model.parameters()))
    print("using device:",device)
    model.to(device)

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t")
    dev_dataset = pd.read_csv("data/etpc-paraphrase-dev.csv", sep="\t")
    test_dataset = pd.read_csv("data/etpc-paraphrase-detection-test-student.csv", sep="\t")
    
    # TODO You might do a split of the train data into train/validation set here
    # (or in the csv files directly)
    train_data = transform_data(train_dataset)
    dev_data = transform_data(dev_dataset)
    test_data = transform_data(test_dataset)
    

    print(f"Loaded {len(train_dataset)} training samples.")

    model = train_model(model, train_data, dev_data, device)

    print("Training finished.")

    accuracy = evaluate_model(model, dev_data, device)
    print(f"The accuracy of the model is: {accuracy:.3f}")

    test_ids = test_dataset["id"]
    test_results = test_model(model, test_data, test_ids, device)
    test_results.to_csv(
        "predictions/bart/etpc-paraphrase-detection-test-output.csv", index=False, sep="\t"
    )


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_detection(args)
