import os
import time

import argparse
import configparser
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm import tqdm


events = {
    'normal': 0,
    'ixnetwork-bgp-hijacking-start': 1,
    'ixnetwork-bgp-injection-start': 2,
    'node-down': 3,
    'interface-down': 4,
    'tap-loss-delay': 5,
}

config = configparser.ConfigParser()
config.read('config/rf_config.ini')

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

    def dump(self, filename='/tmp/rf_transfer.bin'):
        with open(filename, 'wb') as f:
            joblib.dump(self.transfermer, f)

    def load(self, filename='/tmp/rf_transfer.bin'):
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
        self.label = self.label.ravel()

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
        # out_data = torch.tensor(out_data, dtype=torch.float, device=device)

        out_label = self.label[idx + window - 1]
        # out_label = torch.tensor(out_label, dtype=torch.long, device=device)

        return out_data, out_label


class Environment:
    def train(self):
        # losses = []
        np.random.seed(seed)

        data_path = config.get('TRAIN', 'data_path')
        # batch_size = config.getint('TRAIN', 'batch_size')
        # max_epoches = config.getint('TRAIN', 'max_epoches')
        # checkpoint = config.getint('TRAIN', 'checkpoint')
        # is_checkpoint = config.getboolean('TRAIN', 'is_checkpoint')
        log_dir = config.get('TRAIN', 'logdir')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # writer = SummaryWriter(logdir=log_dir)  # tensorboardX logger

        metrics = ["cpu_util", "tx-pps", "rx-pps", "admin-status", "network-incoming-packets-rate",
                   "network-outgoing-packets-rate", "prefix-activity-received-current-prefixes", "as-path"]
        transfermer = PreProcessing(is_train=True)
        label_path = os.path.join(data_path, "label.tsv")

        trainval_dataset = MyDataset(data_path, label_path, metrics, transfermer)
        transfermer.dump()

        train_data, val_data, train_label, val_label = train_test_split(
            trainval_dataset.data,
            trainval_dataset.label,
            test_size=0.2,
            stratify=trainval_dataset.label,
            random_state=seed,
        )
        train_size = len(train_data)
        val_size = len(val_data)
        print(f'train size : {train_size} val size: {val_size}')

        since = int(round(time.time() * 1000))
        model = RandomForestClassifier()
        model.fit(train_data, train_label)
        time_elapsed = int(round(time.time() * 1000)) - since
        print(time_elapsed)

        val_pred = model.predict(val_data)
        print(classification_report(val_pred, val_label, target_names=list(events.keys())))

        joblib.dump(model, f"{model_path}/rf.bin")

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
        test_data = test_dataset.data
        test_label = test_dataset.label

        model = joblib.load(f"{model_path}/rf.bin")

        since = int(round(time.time() * 1000))
        test_pred = model.predict(test_data)
        time_elapsed = int(round(time.time() * 1000)) - since
        print(time_elapsed)

        print(classification_report(test_pred, test_label, target_names=list(events.keys())))
        result_dict = classification_report(test_pred, test_label, target_names=list(events.keys()), output_dict=True)

        df = pd.DataFrame.from_dict(result_dict)
        df.T.to_csv(os.path.join(result_dir, 'classification_report.csv'), index=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Random Forest")
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

