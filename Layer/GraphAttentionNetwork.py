import tensorflow as tf


class GraphAttention(tf.keras.layers.Layer):

    def __init__(self, input_dim, hidden_dim, num_of_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 use_bias=True, kernel_initializer='glorot_uniform',
                 ):

        super(GraphAttention, self).__init__()
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possible reduction methods: concat, average')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # Number of output features (F' in the paper)
        self.num_of_heads = num_of_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.use_bias = use_bias
        self.supports_masking = False

        # Populated by build()
        self.kernels = []       # Layer kernels for attention heads
        self.biases = []        # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim = self.hidden_dim * self.num_of_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.hidden_dim

        # Initialize weights for each attention head
        for head in range(self.num_of_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(self.input_dim, self.hidden_dim),
                                     initializer=self.kernel_initializer,
                                     name='kernel_{}'.format(head))
            self.kernels.append(kernel)

            # Layer bias
            if self.use_bias:
                bias = self.add_weight(shape=(self.hidden_dim, ),
                                       name='bias_{}'.format(head))
                self.biases.append(bias)

            # Attention kernels
            attn_kernel_self = self.add_weight(shape=(self.hidden_dim, 1),
                                               initializer=self.kernel_initializer,
                                               name='attn_kernel_self_{}'.format(head),)
            attn_kernel_neighs = self.add_weight(shape=(self.hidden_dim, 1),
                                                 initializer=self.kernel_initializer,
                                                 name='attn_kernel_neigh_{}'.format(head))
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])

    def call(self, adjacency, input_feature):
        # Node features (N x F)
        # Adjacency matrix (N x N)
        num_of_nodes = input_feature.shape[0]

        outputs = []
        for head in range(self.num_of_heads):
            kernel = self.kernels[head]  # W in the paper (F x F')
            attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network
            features = tf.matmul(input_feature, kernel)  # (N x F')

            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = tf.matmul(features, attention_kernel[0])    # (N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = tf.matmul(features, attention_kernel[1])  # (N x 1), [a_2]^T [Wh_j]

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            dense = attn_for_self + tf.transpose(attn_for_neighs)  # (N x N) via broadcasting

            # Add nonlinearty
            dense = tf.nn.leaky_relu(dense)
            # Mask values before activation (Vaswani et al., 2017)
            mask = 10e9 * tf.sparse.add(-tf.ones([num_of_nodes, num_of_nodes], dtype=tf.float32), adjacency)
            dense += mask

            # Apply softmax to get attention coefficients
            dense = tf.nn.softmax(dense)  # (N x N)

            # Linear combination with neighbors' features
            node_features = tf.matmul(dense, features)  # (N x F')

            if self.use_bias:
                node_features = node_features + self.biases[head]

            # Add output of attention head to final output
            outputs.append(node_features)

        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = tf.concat(outputs, axis=-1)  # (N x KF')
        else:
            output = tf.reduce_mean(tf.stack(outputs), axis=0)  # N x F')

        return output


if __name__ == '__main__':
    # test
    GATlayer = GraphAttention(input_dim=10, hidden_dim=8, num_of_heads=3)
    x = tf.random.normal([4, 10])
    A = tf.sparse.SparseTensor(indices=[[0, 0], [1, 1], [2, 2], [3, 3], [0, 3], [1, 2]],
                               values=[1., 1., 1., 1., 1., 1.], dense_shape=[4, 4])
    print(A.dtype)
    out = GATlayer(A, x)
    print(out)
