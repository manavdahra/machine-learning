import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import sys

tf.compat.v1.disable_eager_execution()

n_classes = 10
batch_size = 128

# one hot encode for 10 MNIST classes


def one_hot(feature, label):
    return feature, tf.one_hot(label, depth=10)


n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

mnist, info = tfds.load("mnist", with_info=True, as_supervised=True)
train_data, test_data = mnist['train'], mnist['test']
train_data = train_data.shuffle(100, reshuffle_each_iteration=True)
train_data = train_data.map(one_hot)
test_data = test_data.map(one_hot)
train_data = train_data.batch(batch_size)

iterator = train_data.make_initializable_iterator()
# test_iterator = test_data.make_initializable_iterator()

next_element = iterator.get_next()
# next_test_element = test_iterator.get_next()

x = tf.compat.v1.placeholder(tf.float32, [None, 784])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.compat.v1.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.compat.v1.random_normal([n_nodes_hl1]))}
    
    hidden_2_layer = {'weights':tf.Variable(tf.compat.v1.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.compat.v1.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.compat.v1.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.compat.v1.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.compat.v1.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.compat.v1.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def convolutional_neural_network(data):
    weights = {
        'W_conv1': tf.Variable(tf.compat.v1.random_normal([5, 5, 1, 32])),
        'W_conv2': tf.Variable(tf.compat.v1.random_normal([5, 5, 32, 64])),
        'W_fc': tf.Variable(tf.compat.v1.random_normal([7*7*64, 1024])),
        'out': tf.Variable(tf.compat.v1.random_normal([1024, n_classes])),
    }

    biases = {
        'b_conv1': tf.Variable(tf.compat.v1.random_normal([32])),
        'b_conv2': tf.Variable(tf.compat.v1.random_normal([64])),
        'b_fc': tf.Variable(tf.compat.v1.random_normal([1024])),
        'out': tf.Variable(tf.compat.v1.random_normal([n_classes])),
    }
    data = tf.reshape(data, shape=[-1, 28, 28, 1])
    conv1 = tf.nn.relu(conv2d(data, weights['W_conv1'])) + biases['b_conv1']
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2'])) + biases['b_conv2']
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

    output = tf.matmul(fc, weights['out']) + biases['out']
    tf.print(output)
    return output

def train_neural_network(x):
    # prediction = convolutional_neural_network(x)
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    cost = tf.compat.v1.Print(cost, [cost], 'cost: ')
    optimizer = tf.compat.v1.train.AdamOptimizer(
        learning_rate=0.001).minimize(cost)
    optimizer = tf.compat.v1.Print(optimizer, [optimizer], 'cost: ')

    num_epochs = 5
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.initialize_all_variables())

        for epoch in range(num_epochs):
            epoch_loss = 0
            sess.run(iterator.initializer)
            batch_train_x, batch_train_y = sess.run(next_element)

            total_batch = int(info.splits["train"].num_examples/batch_size)
            for idx in range(total_batch):
                images, labels = batch_train_x, batch_train_y
                images = images.reshape(-1, 1, 784)
                for j in range(len(labels)):
                    l = labels[j].reshape(-1, 10)
                    _, c = sess.run([optimizer, cost], feed_dict={
                                    x: images[j], y: l})
                    epoch_loss += c
            print('Epoch: ', epoch, ' completed out of ',
                  num_epochs, ' loss: ', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        
        batch_test_x = np.array([])
        batch_test_y = np.array([])
        for sample in tfds.as_numpy(test_data.take(1)):
          sample_image, sample_label = sample
          sample_image = sample_image.reshape(-1, 784)
          sample_label = sample_label.reshape(-1, 10)
          batch_test_x = np.append(batch_test_x, sample_image)
          batch_test_y = np.append(batch_test_y, sample_label)
        print('Accuracy: ', accuracy.eval({x: sample_image, y: sample_label}))


train_neural_network(x)
