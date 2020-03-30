from collections import Counter
from statistics import mean, median

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

env = gym.make('CartPole-v1')
score_requirement = 100

RIGHT_CMD = [0, 1]
LEFT_CMD = [1, 0]


def initial_population():
    training_data = []
    accepted_scores = []
    while True:
        score = 0
        game_memory = []
        action = env.action_space.sample()
        env.reset()
        while True:
            observation, reward, done, info = env.step(action)
            action = env.action_space.sample()
            game_memory.append([observation, LEFT_CMD if action == 0 else RIGHT_CMD])
            if done: break
            score += reward

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                training_data.append(data)
        if len(accepted_scores) > 100:
            break

        env.reset()

    print('Average accepted score: ', mean(accepted_scores))
    print('Median accepted score: ', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data


def neural_network_model():
    model = models.Sequential()
    model.add(layers.Flatten())

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(2, activation='sigmoid'))
    model.add(layers.Dropout(0.2))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data])
    y = np.array([i[1] for i in training_data]).astype(int)

    if not model:
        model = neural_network_model()

    history = model.fit(X, y, epochs=5, verbose=True)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.4, 1.0])
    plt.legend(loc='lower right')

    return model


model = train_model(initial_population())
plt.show()

scores = []
choices = []
for each_game in range(5):
    score = 0
    observation = env.reset()
    while True:
        env.render()
        action = np.argmax(model.predict(np.array(observation).reshape(1, len(observation)))[0])
        choices.append(action)

        observation, reward, done, info = env.step(action)
        score += reward
        if done: break

    scores.append(score)

print('Average Score:', sum(scores) / len(scores))
print(score_requirement)
