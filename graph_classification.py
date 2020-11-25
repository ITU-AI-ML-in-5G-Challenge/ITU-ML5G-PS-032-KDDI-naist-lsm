import os
import time
import argparse
import configparser
import joblib
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch.utils.data import random_split, Dataset, DataLoader, Subset
from tqdm import tqdm
import dgl
from model import GraphClassifier

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    # feats1, graphs1, graphs2, labels = map(list, zip(*samples))
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    # batched_graph2 = dgl.batch(graphs2)
    # return (feats1, batched_graph1, batched_graph2), torch.tensor(labels)
    return batched_graph, torch.tensor(labels)


def generate_physical_network():
    link_list = [
        ['IntGW-01_Gi2', 'TR-01_Gi2'],
        ['IntGW-01_Gi3', 'TR-02_Gi2'],
        ['IntGW-02_Gi2', 'TR-01_Gi3'],
        ['IntGW-02_Gi3', 'TR-02_Gi3'],
        ['RR-01_Gi2', 'TR-01_Gi4'],
        ['RR-01_Gi3', 'TR-02_Gi4'],
        ['TR-01_Gi5', 'TR-02_Gi5'],
        ['IntGW-01_Gi2', 'IntGW-01_Gi3'],
        ['IntGW-01_Gi2', 'IntGW-01_Gi5'],
        ['IntGW-01_Gi2', 'IntGW-01_Gi6'],
        ['IntGW-01_Gi2', 'IntGW-01_Gi7'],
        ['IntGW-01_Gi3', 'IntGW-01_Gi5'],
        ['IntGW-01_Gi3', 'IntGW-01_Gi6'],
        ['IntGW-01_Gi3', 'IntGW-01_Gi7'],
        ['IntGW-01_Gi5', 'IntGW-01_Gi6'],
        ['IntGW-01_Gi5', 'IntGW-01_Gi7'],
        ['IntGW-01_Gi6', 'IntGW-01_Gi7'],
        ['IntGW-02_Gi2', 'IntGW-02_Gi3'],
        ['IntGW-02_Gi2', 'IntGW-02_Gi5'],
        ['IntGW-02_Gi2', 'IntGW-02_Gi6'],
        ['IntGW-02_Gi2', 'IntGW-02_Gi7'],
        ['IntGW-02_Gi3', 'IntGW-02_Gi5'],
        ['IntGW-02_Gi3', 'IntGW-02_Gi6'],
        ['IntGW-02_Gi3', 'IntGW-02_Gi7'],
        ['IntGW-02_Gi5', 'IntGW-02_Gi6'],
        ['IntGW-02_Gi5', 'IntGW-02_Gi7'],
        ['IntGW-02_Gi6', 'IntGW-02_Gi7'],
        ['RR-01_Gi2', 'RR-01_Gi3'],
        ['TR-01_Gi2', 'TR-01_Gi3'],
        ['TR-01_Gi2', 'TR-01_Gi4'],
        ['TR-01_Gi2', 'TR-01_Gi5'],
        ['TR-01_Gi2', 'TR-01_Gi6'],
        ['TR-01_Gi3', 'TR-01_Gi4'],
        ['TR-01_Gi3', 'TR-01_Gi5'],
        ['TR-01_Gi3', 'TR-01_Gi6'],
        ['TR-01_Gi4', 'TR-01_Gi5'],
        ['TR-01_Gi4', 'TR-01_Gi6'],
        ['TR-01_Gi5', 'TR-01_Gi6'],
        ['TR-02_Gi2', 'TR-02_Gi3'],
        ['TR-02_Gi2', 'TR-02_Gi4'],
        ['TR-02_Gi2', 'TR-02_Gi5'],
        ['TR-02_Gi2', 'TR-02_Gi6'],
        ['TR-02_Gi3', 'TR-02_Gi4'],
        ['TR-02_Gi3', 'TR-02_Gi5'],
        ['TR-02_Gi3', 'TR-02_Gi6'],
        ['TR-02_Gi4', 'TR-02_Gi5'],
        ['TR-02_Gi4', 'TR-02_Gi6'],
        ['TR-02_Gi5', 'TR-02_Gi6'],
    ]

    edge_list = pd.DataFrame(link_list, columns=['node_from', 'node_to'])
    g = nx.from_pandas_edgelist(edge_list, source='node_from', target='node_to')

    return g

class PreProcessing:
    def __init__(self, is_train=False):
        self.transformer = dict()
        self.is_train = is_train
        if not self.is_train:
            self.transformer = self.load()

    def __call__(self, df, metric, s_type):
        if self.is_train:
            return self.fit_transform(df, metric, s_type)
        else:
            df[:] = self.transformer[metric].transform(df)
            return df

    def fit_transform(self, df, metric, s_type):
        if s_type == 'minmax':
            self.transformer[metric] = MinMaxScaler()
        if s_type == 'standard':
            self.transformer[metric] = StandardScaler()
        df[:] = self.transformer[metric].fit_transform(df)
        self.dump()
        return df

    def dump(self, filename='/tmp/graph_transformer.bin'):
        with open(filename, 'wb') as f:
            joblib.dump(self.transformer, f)

    def load(self, filename='/tmp/graph_transformer.bin'):
        with open(filename, 'rb') as f:
            data = joblib.load(f)
        return data

class NetworkDataset(Dataset):
    def __init__(self, data_path, gcn_path, interface_metrics, gcn_metrics, transform=None, is_train=False):
        self.transform = transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_train = is_train

        mapping_table = {
            'IntGW-01+link-tr-intgw-a-1': 'IntGW-01_Gi2',
            'IntGW-01+link-tr-intgw-b-1': 'IntGW-01_Gi3',
            'IntGW-01+link-intgw-exgw-a-1': 'IntGW-01_Gi5',
            'IntGW-01+link-intgw-exgw-a-2': 'IntGW-01_Gi6',
            'IntGW-01+link-intgw-exgw-a-3': 'IntGW-01_Gi7',
            'IntGW-02+link-tr-intgw-a-2': 'IntGW-02_Gi2',
            'IntGW-02+link-tr-intgw-b-2': 'IntGW-02_Gi3',
            'IntGW-02+link-intgw-exgw-b-1': 'IntGW-02_Gi5',
            'IntGW-02+link-intgw-exgw-b-2': 'IntGW-02_Gi6',
            'IntGW-02+link-intgw-exgw-b-3': 'IntGW-02_Gi7',
            'RR-01+link-tr-intgw-a-3': 'RR-01_Gi2',
            'RR-01+link-tr-intgw-b-3': 'RR-01_Gi3',
            'TR-01+link-tr-intgw-a-1': 'TR-01_Gi2',
            'TR-01+link-tr-intgw-a-2': 'TR-01_Gi3',
            'TR-01+link-tr-intgw-a-3': 'TR-01_Gi4',
            'TR-01+link-tr-tr-a-1': 'TR-01_Gi5',
            'TR-01+link-tr-ssm-a-1': 'TR-01_Gi6',
            'TR-02+link-tr-intgw-b-1': 'TR-02_Gi2',
            'TR-02+link-tr-intgw-b-2': 'TR-02_Gi3',
            'TR-02+link-tr-intgw-b-3': 'TR-02_Gi4',
            'TR-02+link-tr-tr-a-1': 'TR-02_Gi5',
            'TR-02+link-tr-ssm-b-1': 'TR-02_Gi6',
            'timestamp': 'timestamp',
        }

        node_list = [
            'AS10_GW1', 'AS20_GW1', 'AS30_GW1',
            'AS10_GW2', 'AS20_GW2', 'AS30_GW2',
            '5G_GW1', '5G_GW2'
        ]

        mapping_table2 = {
            'IntGW-01-10.30.2.2-10': ('IntGW-01', 'AS10GW-01'),
            'IntGW-01-10.30.2.6-20': ('IntGW-01', 'AS20GW-01'),
            'IntGW-01-10.30.2.10-30': ('IntGW-01', 'AS30GW-01'),
            'IntGW-02-10.30.2.14-10': ('IntGW-02', 'AS10GW-02'),
            'IntGW-02-10.30.2.18-20': ('IntGW-02', 'AS20GW-02'),
            'IntGW-02-10.30.2.22-30': ('IntGW-02', 'AS30GW-02'),
            'RR-01-10.30.2.14-10': ('RR-01', 'AS10GW-02'),
            'RR-01-10.30.2.2-10': ('RR-01', 'AS10GW-01'),
            'RR-01-10.30.2.18-20': ('RR-01', 'AS20GW-02'),
            'RR-01-10.30.2.6-20': ('RR-01', 'AS20GW-01'),
            'RR-01-10.30.2.10-30': ('RR-01', 'AS30GW-01'),
            'RR-01-10.30.2.22-30': ('RR-01', 'AS30GW-02'),
        }

        node_list2 = [
            'IntGW-01', 'IntGW-02', 'RR-01',
            'AS10GW-01', 'AS10GW-02',
            'AS20GW-01', 'AS20GW-02',
            'AS30GW-01', 'AS30GW-02',
        ]

        label_path = os.path.join(data_path, 'label.tsv')
        label_df = pd.read_csv(label_path, sep='\t', index_col=0)
        label_df['status'] = label_df['status'].replace(6, 5)

        if self.is_train:
            sample_num = 100
            normal_num = len(label_df[label_df['status'] == 0])
            delete_index = np.random.choice(
                label_df[label_df['status'] == 0].index,
                normal_num - sample_num,
                replace=False
            )
            label_df = label_df.drop(delete_index).sort_values('timestamp')
            timestamps = label_df['timestamp']
            self.label = label_df.set_index("timestamp").values
        else:
            timestamps = label_df['timestamp']
            self.label = label_df.set_index('timestamp').values

        interface_df = []
        for metric in interface_metrics:
            df = pd.read_csv(os.path.join(data_path, metric + '.tsv'), sep='\t', index_col=0)
            df = df.fillna(0)
            df = df.sort_values('timestamp').set_index('timestamp')
            df = df.loc[timestamps]
            df = df.reset_index()
            df.rename(columns={'index': 'timestamp'}, inplace=True)
            df.columns = df.columns.map(mapping_table)
            if transform:
                timestamp = df['timestamp']
                df = df.drop('timestamp', axis=1)
                if metric in ["tx-pps", "rx-pps", "tx-kbps", "rx-kbps","network-incoming-packets-rate", "network-outgoing-packets-rate", "in-octets", "out-octets"]:
                    df = self.transform(df, metric, s_type='standard')
                df = pd.concat([df, timestamp], axis=1)
            df['index'] = metric + '_' + df.index.astype(str)
            df = df.set_index('index', drop=True)
            interface_df.append(df)
        for metric in gcn_metrics:
            df = pd.read_csv(os.path.join(gcn_path, metric + '.tsv'), sep='\t', index_col=0)
            df = df.fillna(0)
            df = df.sort_values('timestamp').set_index('timestamp')
            df = df.loc[timestamps]
            df = df.reset_index()
            df.rename(columns={'index': 'timestamp'}, inplace=True)
            if transform:
                timestamp = df['timestamp']
                df = df.drop('timestamp', axis=1)
                if metric in ['cpu-util']:
                    df = self.transform(df, metric, s_type='minmax')
                elif metric in ['prefix-activity-received-current-prefixes', 'prefix-activity-received-bestpaths']:
                    df = df.replace(to_replace=0, method='ffill')
                    df = self.transform(df, metric, s_type='standard')
                else:
                    df = self.transform(df, metric, s_type='standard')
                df = pd.concat([df, timestamp], axis=1)
            df['index'] = metric + '_' + df.index.astype(str)
            df = df.set_index('index', drop=True)
            interface_df.append(df)

        self.data = []
        interface_df = pd.concat(interface_df, sort=True).fillna(0)
        grouping_interface_df = interface_df.groupby('timestamp')
        for timestamp in tqdm(interface_df["timestamp"].unique()):
            tdf = grouping_interface_df.get_group(timestamp)
            tdf.index = tdf.index.map(lambda x: x.split('_')[0])
            tdf = tdf.drop("timestamp", axis=1)
            pg = generate_physical_network()
            nx.set_node_attributes(
                pg, values={column: {'h': tdf[column].values} for column in tdf.columns}
            )
            tdf = tdf.T
            _pg = pg.copy()
            for n in tdf[tdf['admin-status'] == 0].index.values:
                for neighbor in _pg.neighbors(n):
                    if pg.has_edge(n, neighbor):
                        pg.remove_edge(n, neighbor)
            pg = dgl.from_networkx(pg, node_attrs=['h'])
            pg = dgl.add_self_loop(pg)
            self.data.append(pg)
        self.column_dim = len(interface_metrics + gcn_metrics)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_data = out_data.to(self.device)
        # out_data3 = out_data3.to(self.device)
        out_label = self.label[idx]
        out_label = torch.tensor(out_label, dtype=torch.int64, device=self.device)
        # return out_data1, out_data2, out_data3, out_label
        return out_data, out_label


class Environment:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config/gcn_config.ini')

        self.model_dir = self.config.get('MAIN', 'model_dir')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.seed = self.config.getint('MAIN', 'seed')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        dgl.random.seed(self.seed)

        # self.node_metrics = ["cpu_util"]
        self.interface_metrics = ["admin-status", "tx-pps", "rx-pps", "network-incoming-packets-rate", "network-outgoing-packets-rate"]
        # self.bgp_metrics = ["prefix-activity-received-current-prefixes-changed", "as-path-changed"]
        self.gcn_metrics = ["cpu-util", "prefix-activity-received-current-prefixes"]
            # , "as-path"]

        self.events = {
            'normal': 0,
            'ixnetwork-bgp-hijacking-start': 1,
            'ixnetwork-bgp-injection-start': 2,
            'node-down': 3,
            'interface-down': 4,
            'packet-loss-delay': 5,
        }

    def train(self):
        data_path     = self.config.get('TRAIN', 'data_path')
        batch_size    = self.config.getint('TRAIN', 'batch_size')
        max_epoches   = self.config.getint('TRAIN', 'max_epoches')
        checkpoint    = self.config.getint('TRAIN', 'checkpoint')
        is_checkpoint = self.config.getboolean('TRAIN', 'is_checkpoint')
        gcn_path = data_path

        transformer = PreProcessing(is_train=True)

        trainval_dataset = NetworkDataset(
            data_path,
            gcn_path,
            # self.node_metrics,
            self.interface_metrics,
            self.gcn_metrics,
            # self.bgp_metrics,
            transformer,
            # is_train=True
        )

        train_indices, val_indices = train_test_split(
            list(range(len(trainval_dataset))),
            test_size=0.1,
            stratify=trainval_dataset.label,
            random_state=self.seed
        )
        train_dataset = Subset(trainval_dataset, train_indices)
        train_size = len(train_dataset)
        val_dataset = Subset(trainval_dataset, val_indices)
        val_size = len(val_dataset)
        print(f'train size : {train_size} val size: {val_size}')

        train_dataloader = DataLoader(trainval_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
        val_dataloader = DataLoader(val_dataset, batch_size=val_size, shuffle=True, collate_fn=collate)

        input_dim = trainval_dataset.column_dim
        # input_dim2 = len(self.interface_metrics)
        # input_dim3 = 2
        # input_dim4 = len(self.bgp_metrics)
        target_dim = len(self.events.keys())
        model = GraphClassifier(input_dim, target_dim).to(self.device)
        # model = DataParallel(model)
        model.double()

        if is_checkpoint:
            print('./models/gcn_{:}.model'.format(checkpoint))
            model.load_state_dict(torch.load('./models/gcn_{:}.model'.format(checkpoint)))

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        li_times = []

        for epoch in range(1 + checkpoint, max_epoches + 1):
            if self.device == "cuda":
                torch.cuda.synchronize()
            since = int(round(time.time()*1000))
            running_loss, correct, total = (0, 0, 0)
            for train_inputs, train_labels in train_dataloader:
                train_scores = model(train_inputs)
                train_labels = train_labels.to(self.device)
                loss = loss_function(train_scores, train_labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predict = torch.max(train_scores, 1)
                correct += (predict == train_labels).sum().item()
                total += train_labels.size(0)
            train_loss = running_loss / len(train_dataloader)
            train_acc = correct / total

            with torch.no_grad():
                val_inputs, val_labels = iter(val_dataloader).next()
                val_scores = model(val_inputs)
                val_labels = val_labels.to(self.device)
                val_loss = loss_function(val_scores, val_labels)

                bi_scores = torch.argmax(val_scores, dim=1).to('cpu')
                y_val_scores = val_labels.to('cpu').numpy()
                val_acc = accuracy_score(y_val_scores, bi_scores)

            if self.device == "cuda":
                torch.cuda.synchronize()
            time_elapsed = int(round(time.time()*1000)) - since
            li_times.append(time_elapsed)
            print('EPOCH [{}/{}] train loss: {} train acc: {} val loss: {} val acc: {}, elapsed: {}ms'.format(
                epoch, max_epoches, train_loss, train_acc, val_loss, val_acc, time_elapsed))

            if epoch % 10 == 0:
                print("save model")
                torch.save(model.state_dict(), "{:}/gcn_{:}.model".format(self.model_dir, epoch))
        print(np.sum(li_times))
    def test(self):
        data_path = self.config.get('TEST', 'data_path')
        result_dir = self.config.get('TEST', 'result_dir')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        gcn_path = data_path

        transformer = PreProcessing(is_train=False)

        test_dataset = NetworkDataset(
            data_path,
            gcn_path,
            self.interface_metrics,
            self.gcn_metrics,
            transformer,
            is_train=False
        )

        test_size = len(test_dataset)
        test_dataloader = DataLoader(test_dataset, batch_size=test_size, collate_fn=collate)

        input_dim = test_dataset.column_dim
        target_dim = len(self.events.keys())

        model_dir = self.config.get('MAIN', 'model_dir')
        model_paths = os.listdir(model_dir)
        model_paths = [model_path for model_path in model_paths]

        acc_scores = []
        bi_scores = []
        y_test_scores = []
        li_time = []
        losses = []
        logs = []
        _logs = []
        for model_path in tqdm(model_paths):
            if self.device == "cuda":
                torch.cuda.synchronize()
            since = int(round(time.time()*1000))

            model = GraphClassifier(input_dim, target_dim).to(self.device)
            model.load_state_dict(torch.load(os.path.join(model_dir, model_path)))
            model.double()
            loss_function = nn.CrossEntropyLoss()

            with torch.no_grad():
                test_inputs, test_labels = iter(test_dataloader).next()
                test_scores = model(test_inputs)
                test_labels = test_labels.to(self.device)
                test_loss   = loss_function(test_scores, test_labels)

                bi_score = torch.argmax(test_scores, dim=1).to('cpu')
                y_test_score = test_labels.to('cpu').numpy()
                bi_scores.append(bi_score)
                y_test_scores.append(y_test_score)
                acc_scores.append(accuracy_score(y_test_score, bi_score))
                losses.append(test_loss)

                logs.append(classification_report(y_test_score, bi_score, target_names=list(self.events.keys())))
                _logs.append(classification_report(y_test_score, bi_score, target_names=list(self.events.keys()), output_dict=True))
            if self.device == "cuda":
                torch.cuda.synchronize()
            time_elapsed = int(round(time.time()*1000)) - since
            li_time.append(time_elapsed)
        max_index = acc_scores.index(max(acc_scores))
        print(f"Test acc : {max(acc_scores):.4f} loss : {losses[max_index]:.4f} model : {model_paths[max_index]}, elapsed: {li_time[max_index]:.4f}")
        print(logs[max_index])

        df = pd.DataFrame.from_dict(_logs[max_index])
        df.T.to_csv(os.path.join(result_dir, 'classification_report.csv'), index=True)

        out_df = pd.read_csv(os.path.join(data_path, 'label.tsv'), sep='\t', index_col=0)
        out_df['status'] = out_df.apply(lambda x: bi_scores[max_index][x.index])
        out_df.to_csv(os.path.join(result_dir, 'label.tsv'), sep='\t', index=True)

        res_df = pd.read_csv(os.path.join(data_path, 'label.tsv'), sep='\t', index_col=0)
        res_df['pred'] = 0
        res_df['pred'] = res_df.apply(lambda x: bi_scores[max_index][x.index])
        res_df.to_csv(os.path.join(result_dir, 'classification.tsv'), sep='\t', index=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Graph Classification")
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

