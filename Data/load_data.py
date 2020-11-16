import os
import json
from collections import namedtuple
import pandas as pd
import numpy as np
import scipy.sparse as sp
import tensorflow as tf

Data = namedtuple('Data', ['x', 'y', 'adjacency', 'train_mask', 'test_mask'])


class DataLoader:
    def __init__(self, data_root):
        """
        处理之后的数据可以通过属性 .data 获得，它将返回一个数据对象，包括如下几部分：
            * x: 节点的特征，维度为 7624 * 7842，类型为 np.ndarray
            * y: 节点的标签，总共包括 18 个类别，类型为 np.ndarray
            * adjacency: 邻接矩阵，维度为 7624 * 7624，类型为 scipy.sparse.coo.coo_matrix
            * train_mask: 训练集掩码向量，维度为 7624，当节点属于训练集时，相应位置为True，否则False
            * test_mask: 测试集掩码向量，维度为 7624，当节点属于测试集时，相应位置为True，否则False
        """
        self.data_root = data_root
        self._data = self.process_data()

    @property
    def data(self):
        """返回Data数据对象，包括x, y, adjacency, train_mask, val_mask, test_mask"""
        return self._data

    def process_data(self):
        """
        处理数据，得到节点特征和标签，邻接矩阵，训练集、验证集以及测试集
        """
        print("Process data ...")
        features = self.read_data(os.path.join(self.data_root, 'lastfm_asia_features.json'))
        edges = self.read_data(os.path.join(self.data_root, 'lastfm_asia_edges.csv'))
        target = self.read_data(os.path.join(self.data_root, 'lastfm_asia_target.csv'))

        num_nodes = len(features)
        num_training_nodes = int(num_nodes * 0.8)
        num_testing_nodes = num_nodes - num_training_nodes
        train_index = np.arange(num_training_nodes)
        test_index = np.arange(num_training_nodes, num_training_nodes + num_testing_nodes)
        train_mask = np.zeros(num_nodes, dtype=np.bool)
        test_mask = np.zeros(num_nodes, dtype=np.bool)
        train_mask[train_index] = True
        test_mask[test_index] = True

        inputs = self._make_inputs(features)
        adjacency = self.build_adjacency(edges, num_nodes)
        target = self._make_targets(target)
        print("Node's feature shape: ", inputs.shape)
        print("Node's label shape: ", target.shape)
        print("Adjacency's shape: ", adjacency.shape)
        print("Number of training nodes: ", train_mask.sum())
        print("Number of test nodes: ", test_mask.sum())

        return Data(x=inputs, y=target, adjacency=adjacency, train_mask=train_mask, test_mask=test_mask)

    @staticmethod
    def _make_inputs(features):
        inputs = np.zeros([len(features), 7842])
        for i in features:
            for j in features[i]:
                inputs[int(i), int(j)] = 1
        return inputs

    @staticmethod
    def _make_targets(target):
        target = np.array(target['target'])
        return target

    @staticmethod
    def build_adjacency(edges, num_nodes):
        """根据邻接表创建邻接矩阵"""
        num_edges = len(edges)
        from_index = np.array(edges['node_1'])
        to_index = np.array(edges['node_2'])
        adjacency = sp.coo_matrix((np.ones(num_edges),
                                   (from_index, to_index)),
                                  shape=(num_nodes, num_nodes), dtype="float32")
        return adjacency

    @staticmethod
    def read_data(path):
        """使用不同的方式读取原始数据以进一步处理"""
        name = os.path.basename(path)
        if name == "lastfm_asia_features.json":
            with open(path, 'r') as f:
                out = json.load(f)
            return out
        if name == 'lastfm_asia_edges.csv':
            out = pd.read_csv(path)
            return out
        if name == 'lastfm_asia_target.csv':
            out = pd.read_csv(path)
            return out

    @staticmethod
    def normalization(adjacency):
        """计算 L=D^-0.5 * (A+I) * D^-0.5"""
        adjacency += sp.eye(adjacency.shape[0])  # 增加自连接
        degree = np.array(adjacency.sum(1))
        d_hat = sp.diags(np.power(degree, -0.5).flatten())
        return d_hat.dot(adjacency).dot(d_hat).tocoo()


def load_data(data_root):
    dataset = DataLoader(data_root).data
    # node_feature = dataset.x / dataset.x.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1
    tensor_x = tf.constant(dataset.x, dtype=tf.float32)
    tensor_y = tf.constant(dataset.y)
    tensor_train_mask = tf.constant(dataset.train_mask)
    tensor_test_mask = tf.constant(dataset.test_mask)
    normalize_adjacency = DataLoader.normalization(dataset.adjacency)  # 规范化邻接矩阵

    num_nodes, input_dim = dataset.x.shape
    indices = tf.transpose(tf.constant(np.asarray([normalize_adjacency.row, normalize_adjacency.col]).astype('int64')))
    values = tf.constant(normalize_adjacency.data.astype(np.float32))
    tensor_adjacency = tf.sparse.SparseTensor(indices, values, [num_nodes, num_nodes])

    return tensor_x, tensor_y, tensor_train_mask, tensor_test_mask, tensor_adjacency


if __name__ == '__main__':
    data_root = r'C:\Users\62307\Desktop\GraphModel\Data\lastfm_asia'
    print('try to load data.')
    load_data(data_root)
    print('load data successfully.')
