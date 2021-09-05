import os
import sys

import math
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers as T
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger


DATA_DIR = "../dataset/"
OUTPUT_DIR = "../output/"
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER = init_logger()


def init_logger(log_file=OUTPUT_DIR + "train.log"):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


class BaseDataset(Dataset):
    def __init__(self, df, model_name, include_labels=True):
        tokenizer = T.BertTokenizer.from_pretrained(model_name)

        self.df = df
        self.include_labels = include_labels

        self.title = df["title"].tolist()
        self.encoded = tokenizer.batch_encode_plus(
            self.title,
            padding="max_length",
            max_length=72,
            truncation=True,
            return_attention_mask=True,
        )

        if self.include_labels:
            self.labels = df["judgement"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.encoded["input_ids"][idx])
        attention_mask = torch.tensor(self.encoded["attention_mask"][idx])

        if self.include_labels:
            label = torch.tensor(self.labels[idx]).float()
            return input_ids, attention_mask, label

        return input_ids, attention_mask


class BaseModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.model = T.BertForSequenceClassification.from_pretrained(
            model_name, num_labels=1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        out = self.sigmoid(out.logits).squeeze()

        return out


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))


def train_fn(train_loader, model, criterion, optimizer, epoch, device):
    start = end = time.time()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    for step, (input_ids, attention_mask, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        y_preds = model(input_ids, attention_mask)

        loss = criterion(y_preds, labels)

        # record loss
        losses.update(loss.item(), batch_size)
        loss.backward()

        optimizer.step()

        if step % 100 == 0 or step == (len(train_loader) - 1):
            print(
                f"Epoch: [{epoch + 1}][{step}/{len(train_loader)}] "
                f"Elapsed {timeSince(start, float(step + 1) / len(train_loader)):s} "
                f"Loss: {losses.avg:.4f} "
            )

    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    start = end = time.time()
    losses = AverageMeter()

    # switch to evaluation mode
    model.eval()
    preds = []

    for step, (input_ids, attention_mask, labels) in enumerate(valid_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        # compute loss
        with torch.no_grad():
            y_preds = model(input_ids, attention_mask)

        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)

        # record score
        preds.append(y_preds.to("cpu").numpy())

        if step % 100 == 0 or step == (len(valid_loader) - 1):
            print(
                f"EVAL: [{step}/{len(valid_loader)}] "
                f"Elapsed {timeSince(start, float(step + 1) / len(valid_loader)):s} "
                f"Loss: {losses.avg:.4f} "
            )

    predictions = np.concatenate(preds)
    return losses.avg, predictions


def inference():
    """推論用関数
    """
    predictions = []

    test_dataset = BaseDataset(test, "bert-base-uncased", include_labels=False)
    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
    )

    for fold in range(5):
        LOGGER.info(
            f"========== model: bert-base-uncased fold: {fold} inference =========="
        )
        model = BaseModel("bert-base-uncased")
        model.to(device)
        model.load_state_dict(
            torch.load(OUTPUT_DIR + f"bert-base-uncased_fold{fold}_best.pth")["model"]
        )
        model.eval()
        preds = []
        for i, (input_ids, attention_mask) in tqdm(
            enumerate(test_loader), total=len(test_loader)
        ):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            with torch.no_grad():
                y_preds = model(input_ids, attention_mask)
            preds.append(y_preds.to("cpu").numpy())
        preds = np.concatenate(preds)
        predictions.append(preds)
    predictions = np.mean(predictions, axis=0)

    return predictions


def train_loop(train, fold):
    """学習用関数
    """

    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # Data Loader
    # ====================================================
    trn_idx = train[train["fold"] != fold].index
    val_idx = train[train["fold"] == fold].index

    train_folds = train.loc[trn_idx].reset_index(drop=True)
    valid_folds = train.loc[val_idx].reset_index(drop=True)

    train_dataset = BaseDataset(train_folds, "bert-base-uncased")
    valid_dataset = BaseDataset(valid_folds, "bert-base-uncased")

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # ====================================================
    # Model
    # ====================================================
    model = BaseModel("bert-base-uncased")
    model.to(device)

    optimizer = T.AdamW(model.parameters(), lr=2e-5)

    criterion = nn.BCELoss()

    # ====================================================
    # Loop
    # ====================================================
    best_score = -1
    best_loss = np.inf

    for epoch in range(3):
        start_time = time.time()

        # train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, device)

        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)
        valid_labels = valid_folds["judgement"].values

        # scoring
        score = fbeta_score(valid_labels, np.where(preds < border, 0, 1), beta=7.0)

        elapsed = time.time() - start_time
        LOGGER.info(
            f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
        )
        LOGGER.info(f"Epoch {epoch+1} - Score: {score}")

        if score > best_score:
            best_score = score
            LOGGER.info(f"Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model")
            torch.save(
                {"model": model.state_dict(), "preds": preds},
                OUTPUT_DIR + f"bert-base-uncased_fold{fold}_best.pth",
            )

    check_point = torch.load(OUTPUT_DIR + f"bert-base-uncased_fold{fold}_best.pth")

    valid_folds["preds"] = check_point["preds"]

    return valid_folds


def get_result(result_df):
    preds = result_df["preds"].values
    labels = result_df["judgement"].values
    score = fbeta_score(labels, np.where(preds < border, 0, 1), beta=7.0)
    LOGGER.info(f"Score: {score:<.5f}")


def main():
    # Training
    oof_df = pd.DataFrame()
    for fold in range(5):
        _oof_df = train_loop(train, fold)
        oof_df = pd.concat([oof_df, _oof_df])
        LOGGER.info(f"========== fold: {fold} result ==========")
        get_result(_oof_df)

    # CV result
    LOGGER.info(f"========== CV ==========")
    get_result(oof_df)

    # Save OOF result
    oof_df.to_csv(OUTPUT_DIR + "oof_df.csv", index=False)

    # Inference
    predictions = inference()
    predictions = np.where(predictions < border, 0, 1)

    # submission
    sub["judgement"] = predictions
    sub.to_csv(OUTPUT_DIR + "submission.csv", index=False, header=False)


if __name__ == "__main__":
    main()
