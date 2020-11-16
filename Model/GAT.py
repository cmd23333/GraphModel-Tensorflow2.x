import tensorflow as tf
from Layer.GraphAttentionNetwork import GraphAttention


class GATNet(tf.keras.Model):
    """
    定义一个包含两层GraphConvolution的模型
    """
    def __init__(self, input_dim, output_dim):
        super(GATNet, self).__init__()
        self.gat1 = GraphAttention(input_dim, 128, 2)
        self.gat2 = GraphAttention(256, output_dim, 1, 'average')

    def call(self, adjacency, feature):
        h = tf.nn.elu(self.gat1(adjacency, feature))
        logits = self.gat2(adjacency, h)
        return logits
