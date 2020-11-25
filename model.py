import torch
import torch.nn as nn

import dgl
from dgl.nn.pytorch import GraphConv, SAGEConv
from dgl.nn.pytorch.glob import SortPooling, MaxPooling, AvgPooling, SumPooling


def init(module, gain):
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0)
    return module


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, target_dim):
        super(MLPClassifier, self).__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, target_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        return self.fc3(x)


class GraphClassifier(nn.Module):
    def __init__(self, input_dim, target_dim):
        super(GraphClassifier, self).__init__()

        hidden_dim = 8
        hidden_dim2 = 16
        self.graph_conv1 = GraphConv(input_dim, hidden_dim)
        self.graph_conv2 = GraphConv(hidden_dim, hidden_dim2)
        self.relu = nn.ReLU()
        self.pooling = AvgPooling()

        fc_hidden_dim = 32
        self.fc1 = nn.Linear(hidden_dim2, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, target_dim)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, g):
        h = g.ndata['h']
        h = self.graph_conv1(g, h)
        h = self.relu(h)
        h = self.graph_conv2(g, h)
        h = self.relu(h)
        h = self.pooling(g, h)
        h = self.fc1(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return h


class InterfaceClassifier(nn.Module):
    def __init__(self, input_dim, target_dim):
        super(InterfaceClassifier, self).__init__()

        hidden_dim = 32
        self.graph_conv1 = SAGEConv(input_dim, hidden_dim, aggregator_type='pool')
        self.graph_conv2 = SAGEConv(hidden_dim, hidden_dim, aggregator_type='pool')
        self.graph_conv3 = SAGEConv(hidden_dim, target_dim, aggregator_type='pool')
        # self.graph_conv1 = GraphConv(input_dim, hidden_dim)
        # self.graph_conv2 = GraphConv(hidden_dim, hidden_dim)
        # self.graph_conv3 = GraphConv(hidden_dim, target_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g):
        h = g.ndata['h']
        h = self.graph_conv1(g, h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.graph_conv2(g, h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.graph_conv3(g, h)
        h = self.sigmoid(h)
        return h.view(-1, 22)


class NodeClassifier(nn.Module):
    def __init__(self, input_dim, target_dim):
        super(NodeClassifier, self).__init__()

        hidden_dim = 32
        self.graph_conv1 = SAGEConv(input_dim, hidden_dim, aggregator_type='pool')
        self.graph_conv2 = SAGEConv(hidden_dim, hidden_dim, aggregator_type='pool')
        self.graph_conv3 = SAGEConv(hidden_dim, target_dim, aggregator_type='pool')

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g):
        h = g.ndata['h']
        h = self.graph_conv1(g, h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.graph_conv2(g, h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.graph_conv3(g, h)
        h = self.sigmoid(h)
        return h.view(-1, 5)
