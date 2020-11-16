import tensorflow as tf


class MLP(tf.keras.Model):
    """
    定义一个包含三层的全连接网络
    """
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = tf.keras.layers.Dense(input_dim=input_dim, units=128)
        self.fc2 = tf.keras.layers.Dense(64)
        self.fc3 = tf.keras.layers.Dense(output_dim)

    def call(self, adjacency, feature):
        h = tf.nn.relu(self.fc1(feature))
        h = tf.nn.relu(self.fc2(h))
        logits = self.fc3(h)
        return logits