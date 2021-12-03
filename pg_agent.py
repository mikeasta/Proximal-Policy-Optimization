import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Model, load_model
import keras.backend as K
import numpy as np

class Agent(object):
    def __init__(self, ALPHA, GAMMA=0.99, n_actions=3, input_dims=2\
        ,layer_size=3, layer2_size=3, fname="a2c.h5"):

        self.gamma = GAMMA
        self.R     = 0
        self.learning_rate = ALPHA

        self.input_dims  = input_dims
        self.dense_dims  = layer_size
        self.dense2_dims = layer2_size
        self.n_actions   = n_actions
        
        self.state_memory  = []
        self.action_memory = []
        self.reward_memory = []

        self.policy, self.predict = self.build_policy_network()

        self.action_space = [i for i in range(n_actions)]
        self.model_file   = fname

    def build_policy_network(self):
        input_l    = Input(shape=(self.input_dims,))
        advantages = Input(shape=[1])
        dense      = Dense(self.dense_dims, activation="relu")(input_l)
        dense2     = Dense(self.dense2_dims, activation="relu")(dense)
        probs      = Dense(self.n_actions, activation="softmax")(dense2)

        # Определяем функцию потерь
        def loss_func(y_true, y_pred):
            out     = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true * K.log(out)

            return K.sum(-log_lik * advantages)

        policy = Model([input_l, advantages], probs)
        policy.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss=loss_func)

        predict = Model(input_l, probs)

        return policy, predict

    # Метод выбора действия исходя из
    def choose_action(self, observation):
        state         = observation[np.newaxis, :]
        probabilities = self.predict.predict(state)[0]
        action        = np.random.choice(self.action_space, p=probabilities)

        return action

    def store_observation(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def learn(self):
        state_memory  = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)

        actions = np.zeros([len(action_memory), self.n_actions])
        actions[np.arange(len(action_memory)), action_memory] = 1

        R = np.zeros_like(reward_memory)

        # Считаем суммы дисконтированных наград
        for t in range(len(reward_memory)):
            R_sum    = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                R_sum    += reward_memory[k]*discount
                discount *= self.gamma

            R[t] = R_sum
        
        # Вычисляем коэффициенты целесообразности (advantage)
        # Дла каждой вычисленной СДН
        mean   = np.mean(R)
        std    = np.std(R) if np.std(R) > 0 else 1
        self.R = (R - mean) / std

        # Проносим данные о состоянии + коэффициенты целесообразности через 
        # нейросеть и проводим обучение
        _ = self.policy.train_on_batch([state_memory, self.R], actions)

        self.state_memory  = []
        self.action_memory = []
        self.reward_memory = []

    def save_model(self):
        self.policy.save(self.model_file)

    def load_model(self):
        self.policy = load_model(self.model_file)