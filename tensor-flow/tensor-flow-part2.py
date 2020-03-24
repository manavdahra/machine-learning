import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

tf.compat.v1.disable_eager_execution()

'''
input 
> weight > hidden layer 1 (activation) 
> weight > hidden layer 2 (activation) 
> weight > output layer 

compare output to intdented output > cost function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer)

backpropagation 

feed forward + backprop = epoch

'''

mnist, info = tfds.load("mnist", with_info=True)
train_data, test_data = mnist['train'], mnist['test']
print(test_data)
iterator = train_data.make_one_shot_iterator()
image = iterator.get_next()


assert isinstance(train_data, tf.data.Dataset)
assert info.features['label'].num_classes == 10
assert info.splits['train'].num_examples == 60000

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

# height x width
x = tf.compat.v1.placeholder('float', [None, 784])
y = tf.compat.v1.placeholder('float')


def neural_network_model(data):
    # (input_data * weights) + biases
    hidden_1_layer = {
        'weights': tf.Variable(tf.random.normal([784, n_nodes_hl1])),
        'biases': tf.Variable(tf.random.normal([n_nodes_hl1]))
    }

    hidden_2_layer = {
        'weights': tf.Variable(tf.random.normal([n_nodes_hl1, n_nodes_hl2])),
        'biases': tf.Variable(tf.random.normal([n_nodes_hl2]))
    }

    hidden_3_layer = {
        'weights': tf.Variable(tf.random.normal([n_nodes_hl2, n_nodes_hl3])),
        'biases': tf.Variable(tf.random.normal([n_nodes_hl3]))
    }

    hidden_4_layer = {
        'weights': tf.Variable(tf.random.normal([n_nodes_hl3, n_classes])),
        'biases': tf.Variable(tf.random.normal([n_classes]))
    }

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases'])
    return output

def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)

def train_neural_network(x):
  prediction = neural_network_model(x)
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))

  optimizer = tf.compat.v1.train.AdamOptimizer().minimize(cost)

  num_epochs = 10
  with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.initialize_all_variables())

    i = 0
    for epoch in range(num_epochs):
      epoch_loss = 0
      # ds = train_data.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
      for example in tfds.as_numpy(train_data.skip(i * batch_size).take((i + 1) * batch_size)):
        image, label = example["image"], example["label"]
        image = image.reshape([-1, 784])
        label = bin_array(label, 10)
        # print(label, bin_array(label, 10))
        _, c = sess.run([optimizer, cost], feed_dict= {x: image, y: label})
        epoch_loss += c 
        i += 1
      print('Epoch: ', epoch, ' completed out of ', num_epochs, ' loss: ', epoch_loss)
    

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    images = np.empty([1, 784], dtype=float)
    labels = np.empty([10], dtype=float)
    for example in tfds.as_numpy(test_data.take(10)):
      image, label = example["image"], example["label"]
      image = image.reshape([-1, 784])
      label = bin_array(label, 10)
      # print(image.shape, label.shape)
      images = np.append(images, image, axis=0)
      labels = np.append(labels, label, axis=0)
    
    print('Accuracy: ', accuracy.eval({x: images, y: labels}))

train_neural_network(x)