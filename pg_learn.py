import gym
from json_impl import json_get_data, json_wrt_data
from pg_agent import Agent
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()



# ! Гиперпараметры для обучения
GAMMA             = 0.99
ALPHA             = 0.0005
EPISODES          = 500

GYM_NAME          = "CartPole-v0"
N_ACTION          = 2
INPUT_DIMS        = 4
N_NEURONS_D1      = 64
N_NEURONS_D2      = 64

TAKE_LAST_AVG     = 100


savefilename      = "pg.h5"


# Initialize environment
env           = gym.make(GYM_NAME)
agent         = Agent(ALPHA, GAMMA, N_ACTION, INPUT_DIMS, N_NEURONS_D1, N_NEURONS_D2, savefilename)



# Train
score_history     = []
avg_history       = [] 

for i in range(EPISODES):
    done  = False
    score = 0
    observation = env.reset()
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        agent.store_observation(observation_, action, reward)
        observation = observation_
        score       += reward
        # env.render()

    score_history.append(score)
    avg_score = np.mean(score_history[-TAKE_LAST_AVG:])
    avg_history.append(avg_score)
    agent.learn()

    print ("Episode:", i,
        "Score:", score,
        "Avg_score:", avg_score)


# ? Созранение массива средних значений в JSON файл
data        = json_get_data()
data["pg_scores"] = score_history
data["pg_avg"]    = avg_history
json_wrt_data(data)


# ? Сохранение нейросети агента 
agent.save_model()
