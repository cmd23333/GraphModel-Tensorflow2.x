import tensorflow as tf


class GraphConvolution(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, use_bias=True, kernel_initializer='glorot_uniform'):
        """图卷积：L*X*\theta

        Args:
        ----------
            input_dim: int: 节点输入特征的维度
            output_dim: int： 输出特征维度
            use_bias : bool, optional： 是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)

        self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
                                      initializer=self.kernel_initializer,
                                      name='kernel')
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        name='bias')
        else:
            self.bias = None

    def call(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法

        Args:
        -------
            adjacency: 邻接矩阵
            input_feature: 输入特征
        """
        support = tf.matmul(input_feature, self.kernel)
        if isinstance(adjacency, tf.sparse.SparseTensor):
            output = tf.sparse.sparse_dense_matmul(adjacency, support)
        else:
            output = tf.matmul(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'


if __name__ == '__main__':
    # test
    GCNlayer = GraphConvolution(10, 5)
    x = tf.random.normal([4, 10])
    A = tf.sparse.SparseTensor(indices=[[0, 0], [1, 1], [2, 2], [3, 3], [0, 3], [1, 2]],
                               values=[1., 1., 1., 1., 1., 1.], dense_shape=[4, 4])
    print(A.dtype)
    print(GCNlayer)
    out = GCNlayer(A, x)
    print(out)
