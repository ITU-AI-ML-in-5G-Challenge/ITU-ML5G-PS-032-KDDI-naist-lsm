import os
import time
import configparser
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, random_split, DataLoader, Subset
from tqdm import tqdm
import argparse
from model import MLPClassifier


events = {
    'normal': 0,
    'ixnetwork-bgp-hijacking-start': 1,
    'ixnetwork-bgp-injection-start': 2,
    'node-down': 3,
    'interface-down': 4,
    'tap-loss-delay': 5,
}

config = configparser.ConfigParser()
config.read('config/mlp_config.ini')

model_path = config.get('MAIN', 'model_path')
model_dir = model_path
window = config.getint('MAIN', 'window')
seed = config.getint('MAIN', 'seed')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.manual_seed(seed)

if not os.path.exists(model_path):
    os.makedirs(model_path)


class PreProcessing:
    def __init__(self, is_train=True):
        self.transfermer = dict()
        self.is_train = is_train
        if not self.is_train:
            self.transfermer = self.load()

    def __call__(self, df, s_type='minmax', is_timeseries=False):
        if self.is_train:
            return self.fit_transform(df, s_type, is_timeseries)
        else:
            return self.transform(df, is_timeseries)

    def fit_transform(self, df, s_type, is_timeseries):
        for column in df.columns:
            # feature scaling
            if s_type == 'minmax':
                self.transfermer[column] = MinMaxScaler()
            elif s_type == 'standard':
                self.transfermer[column] = StandardScaler()
            value = self.transfermer[column].fit_transform(
                pd.DataFrame(df[column]))
            df.loc[:, column] = value
            # lag feature
            if is_timeseries:
                df[column + "_diff"] = df[column].diff()
                df[column + "_mean_5"] = df[column].rolling(5).mean()
                df = df.fillna(df.median())
        return df

    def transform(self, df, is_timeseries):
        for column in df.columns:
            value = self.transfermer[column].transform(
                pd.DataFrame(df[column]))
            df.loc[:, column] = value
            if is_timeseries:
                df[column + "_diff"] = df[column].diff()
                df[column + "_mean_5"] = df[column].rolling(5).mean()
                df = df.fillna(df.median())
        return df

    def dump(self, filename='/tmp/mlp_transfer.bin'):
        with open(filename, 'wb') as f:
            joblib.dump(self.transfermer, f)

    def load(self, filename='/tmp/mlp_transfer.bin'):
        with open(filename, 'rb') as f:
            data = joblib.load(f)
        return data


class MyDataset(Dataset):
    def __init__(self, data_path, label_path, metrics, transform=None):
        self.transform = transform
        self.metrics = metrics

        data = []
        for metric in tqdm(self.metrics):
            df = pd.read_csv(os.path.join(data_path, metric + '.tsv'), sep="\t", index_col=0)
            df = df.fillna(0)
            df = df.sort_values("timestamp")
            df = df.set_index("timestamp")
            columns_name = {name: metric + '_' + name for name in df.columns}
            df.rename(columns=columns_name, inplace=True)
            if self.transform:
                if metric != "admin-status":
                    df = self.transform(df, s_type='standard')
            data.append(df)
        self.dataframe = pd.concat(data, axis=1)
        self.dataframe = self.dataframe.reindex(columns=sorted(self.dataframe.columns))
        self.data = self.dataframe.values
        self.data_num = len(self.dataframe)
        self.label = pd.read_csv(label_path, sep="\t", index_col=0).set_index("timestamp").values

        for idx in range(len(self.label)):
            if self.label[idx] in [5, 6]:
                self.label[idx] = 5

        # for idx in range(len(self.label)):
        #     if self.label[idx] in [1, 2]:
        #         self.label[idx] = 1
        #     elif self.label[idx] in [4, 5, 6]:
        #         self.label[idx] = 2
        #     elif self.label[idx] == 3:
        #         self.label[idx] = 3

    def __len__(self):
        return self.data_num - window + 1

    def __getitem__(self, idx):
        if window == 1:
            out_data = self.data[idx]
        else:
            out_data = self.data[idx: idx + window, :]
        out_data = torch.tensor(out_data, dtype=torch.float, device=device)

        out_label = self.label[idx + window - 1]
        out_label = torch.tensor(out_label, dtype=torch.long, device=device)

        return out_data, out_label


class Environment:
    def train(self):
        # losses = []
        np.random.seed(seed)

        data_path = config.get('TRAIN', 'data_path')
        batch_size = config.getint('TRAIN', 'batch_size')
        max_epoches = config.getint('TRAIN', 'max_epoches')
        checkpoint = config.getint('TRAIN', 'checkpoint')
        is_checkpoint = config.getboolean('TRAIN', 'is_checkpoint')
        log_dir = config.get('TRAIN', 'logdir')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        writer = SummaryWriter(logdir=log_dir)  # tensorboardX logger

        metrics = ["cpu_util", "tx-pps", "rx-pps", "admin-status", "network-incoming-packets-rate",
                   "network-outgoing-packets-rate", "prefix-activity-received-current-prefixes", "as-path"]
        transfermer = PreProcessing(is_train=True)
        label_path = os.path.join(data_path, "label.tsv")

        trainval_dataset = MyDataset(data_path, label_path, metrics, transfermer)
        transfermer.dump()

        train_indices, val_indices = train_test_split(
            list(range(len(trainval_dataset))),
            test_size=0.2,
            stratify=trainval_dataset.label,
            random_state=seed,
        )
        train_dataset = Subset(trainval_dataset, train_indices)
        train_size = len(train_dataset)
        val_dataset = Subset(trainval_dataset, val_indices)
        val_size = len(val_dataset)
        print(f'train size : {train_size} val size: {val_size}')

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset))

        val_data, val_labels = iter(val_dataloader).next()
        val_data = val_data.to(device)
        val_labels = val_labels.to(device).view(-1)

        input_dim = list(train_dataset[0][0].shape)[-1]
        target_dim = len(events.keys())
        model = MLPClassifier(input_dim, target_dim).to(device)

        if is_checkpoint:
            print('./{:}/pytorch_{:}.model'.format(model_dir, checkpoint))
            model.load_state_dict(torch.load('./{:}/pytorch_{:}.model'.format(model_dir, checkpoint)))
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        li_times = []

        for epoch in range(1 + checkpoint, max_epoches + 1):
            if device == "cuda":
                torch.cuda.synchronize()
            since = int(round(time.time() * 1000))
            running_loss, correct, total = (0, 0, 0)
            for train_data, train_labels in train_dataloader:
                train_data = train_data.to(device)
                train_labels = train_labels.to(device).view(-1)

                model.zero_grad()
                train_scores = model(train_data)
                loss = loss_function(train_scores, train_labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predict = torch.max(train_scores.data, 1)
                correct += (predict == train_labels).sum().item()
                total += train_labels.size(0)
            train_loss = running_loss / len(train_dataloader)
            train_acc = correct / total

            with torch.no_grad():
                val_scores = model(val_data)
                val_loss = loss_function(val_scores, val_labels)

                bi_scores = torch.argmax(val_scores, dim=1).to('cpu').numpy()
                y_val_scores = val_labels.to('cpu').numpy()
                val_acc = accuracy_score(y_val_scores, bi_scores)

            if device == "cuda":
                torch.cuda.synchronize()
            time_elapsed = int(round(time.time() * 1000)) - since
            li_times.append(time_elapsed)
            print(f'EPOCH: [{epoch}/{max_epoches}] train loss: {train_loss:.4f} train acc: {train_acc:.4f} val loss: {val_loss:.4f} val acc: {val_acc:.4f} elapsed: {time_elapsed:.4f}ms')
            writer.add_scalars('data/loss', {
                "train loss": train_loss,
                "validation loss": val_loss
            }, epoch)
            writer.add_scalars('data/metric', {
                'train accuracy': train_acc,
                "validation accuracy": val_acc
            }, epoch)

            if epoch % 10 == 0:
                print("save model")
                torch.save(model.state_dict(), "./{:}/pytorch_{:}.model".format(model_dir, epoch))
            writer.close()
        print(np.sum(li_times))

    def test(self):
        # losses = []
        np.random.seed(seed)
        data_path = config.get('TEST', 'data_path')
        result_dir = config.get('TEST', 'result_dir')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        # model_file = config.get('TEST', 'model_file')

        metrics = ["cpu_util", "tx-pps", "rx-pps", "admin-status", "network-incoming-packets-rate",
                   "network-outgoing-packets-rate", "prefix-activity-received-current-prefixes", "as-path"]
        transfermer = PreProcessing(is_train=False)
        label_path = os.path.join(data_path, "label.tsv")

        test_dataset = MyDataset(data_path, label_path, metrics, transfermer)
        input_dim = list(test_dataset[0][0].shape)[-1]
        target_dim = len(events.keys())

        test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))
        test_data, test_label = iter(test_dataloader).next()
        test_data = test_data.to(device)
        test_label = test_label.to(device).view(-1)

        model_paths = os.listdir(model_dir)
        model_paths = [model_path for model_path in model_paths if 'pytorch' in model_path]

        acc_scores = []
        losses = []
        li_time = []
        logs = []
        _logs = []
        for model_path in model_paths:
            if device == "cuda":
                torch.cuda.synchronize()
            since = int(round(time.time() * 1000))
            model = MLPClassifier(input_dim, target_dim).to(device)
            model.load_state_dict(torch.load(os.path.join(model_dir, model_path)))
            loss_function = nn.CrossEntropyLoss()

            with torch.no_grad():
                test_scores = model(test_data)
                loss = loss_function(test_scores, test_label)

                bi_scores = torch.argmax(test_scores, dim=1).to('cpu').numpy()
                y_test_scores = test_label.to('cpu').numpy()
                acc_scores.append(accuracy_score(y_test_scores, bi_scores))
                losses.append(loss)

                logs.append(classification_report(y_test_scores, bi_scores, target_names=list(events.keys())))
                _logs.append(classification_report(y_test_scores, bi_scores,
                                                   target_names=list(events.keys()), output_dict=True))
            if device == "cuda":
                torch.cuda.synchronize()
            time_elapsed = int(round(time.time() * 1000)) - since
            li_time.append(time_elapsed)
        max_index = acc_scores.index(max(acc_scores))
        print(
            f"Test acc : {max(acc_scores):.4f} loss : {losses[max_index]:.4f} model : {model_paths[max_index]} elapsed: {li_time[max_index]:.4f}")
        print(logs[max_index])
        df = pd.DataFrame.from_dict(_logs[max_index])
        df.T.to_csv(os.path.join(result_dir, 'classification_report.csv'), index=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MLP Classification")
    parser.add_argument('--train', action="store_true",
                    help='is train mode')
    parser.add_argument('--test', action="store_true",
                    help='is test mode')
    parser.add_argument('--both', action="store_true",
                    help='is both train and test mode')
    args = parser.parse_args()
    env = Environment()
    if args.both:
        env.train()
        env.test()
    elif args.test:
        env.test()
    elif args.train:
        env.train()
    else:
        parser.print_help()

