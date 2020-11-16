import tensorflow as tf
from Data.load_data import load_data
from Model.GAT import GATNet
from Model.GCN import GCNNet
from Model.MLP import MLP
from sklearn.metrics import f1_score


def main(mode, num_epoch=500, learning_rate=0.005):
    # load data
    data_root = r'C:\Users\62307\Desktop\GraphModel\Data\lastfm_asia'
    tensor_x, tensor_y, tensor_train_mask, tensor_test_mask, tensor_adjacency = load_data(data_root)
    num_nodes, input_dim = tensor_x.shape

    # build model
    if mode == 'gcn':
        # 模型定义：Model, Loss, Optimizer
        model = GCNNet(input_dim, 18)
    elif mode == 'gat':
        # 模型定义：Model, Loss, Optimizer
        model = GATNet(input_dim, 18)
    elif mode == 'mlp':
        model = MLP(input_dim, 18)
    else:
        raise ValueError('mode should be included in {gcn, gat, mlp}')

    # compile
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # record
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, directory='Saved_Weights/' + mode, max_to_keep=5)
    writer = tf.summary.create_file_writer('Logs/' + mode)

    # training
    train_y = tensor_y[tensor_train_mask]

    best_test_macro_f1 = 0
    for epoch in range(num_epoch):
        with tf.GradientTape() as t:
            logits = model(tensor_adjacency, tensor_x)
            train_mask_logits = logits[tensor_train_mask]
            loss = loss_fn(train_y, train_mask_logits)
        grads = t.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # training metrics
        predict_y = tf.argmax(train_mask_logits, axis=-1).numpy()
        num_accurate = tf.cast(tf.equal(predict_y, tensor_y[tensor_train_mask]), dtype=tf.float32)
        train_accuracy = tf.reduce_mean(num_accurate).numpy()
        train_f1_macro = f1_score(tensor_y[tensor_train_mask].numpy(), predict_y, average='macro')

        # testing metrics
        test_mask_logits = logits[tensor_test_mask]
        predict_y = tf.argmax(test_mask_logits, axis=-1).numpy()
        num_accurate = tf.cast(tf.equal(predict_y, tensor_y[tensor_test_mask]), dtype=tf.float32)
        test_accuracy = tf.reduce_mean(num_accurate).numpy()
        test_f1_macro = f1_score(tensor_y[tensor_test_mask].numpy(), predict_y, average='macro')

        with writer.as_default():
            tf.summary.scalar('train_loss', loss.numpy(), step=epoch)
            tf.summary.scalar('train_acc', train_accuracy, step=epoch)
            tf.summary.scalar('train_f1_macro', train_f1_macro, step=epoch)
            tf.summary.scalar('test_acc', test_accuracy, step=epoch)
            tf.summary.scalar('test_f1_macro', test_f1_macro, step=epoch)

        if test_f1_macro > best_test_macro_f1:
            best_test_macro_f1 = test_f1_macro
            manager.save(checkpoint_number=epoch)

        print('epoch {},  train loss {:.4f},  train accuracy {:.4f},  train macro f1 {:.4f},  '
              'test accuracy {:.4f},  test macro f1 {:.4f}'.format(
                epoch, loss.numpy(),
                train_accuracy, train_f1_macro,
                test_accuracy, test_f1_macro))


if __name__ == '__main__':
    main(mode='mlp', learning_rate=0.005)
