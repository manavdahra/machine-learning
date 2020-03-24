from random import randint

import matplotlib.pyplot as plt
import tensorflow as tf

'''
Label	Class
0	    0
1	    1
2	    2
3	    3
4	    4
5	    5
6	    6
7	    7
8	    8
9	    9
'''

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test, verbose=2)

for _ in range(10):
    image_index = randint(1, 10000)
    plt.imshow(x_test[image_index].reshape(28, 28), cmap='Greys')
    pred = model.predict(x_test[image_index].reshape(1, 28, 28))
    print(pred.argmax())
    plt.show()
