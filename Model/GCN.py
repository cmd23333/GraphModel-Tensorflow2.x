import tensorflow as tf
from Layer.VanillaGCN import GraphConvolution


class GCNNet(tf.keras.Model):
    """
    定义一个包含两层GraphConvolution的模型
    """
    def __init__(self, input_dim, output_dim):
        super(GCNNet, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, 128)
        self.gcn2 = GraphConvolution(128, output_dim)

    def call(self, adjacency, feature):
        h = tf.nn.relu(self.gcn1(adjacency, feature))
        logits = self.gcn2(adjacency, h)
        return logits