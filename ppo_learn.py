import gym
from json_impl import json_get_data, json_wrt_data
from ppo_agent import Agent
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()



# ! Гиперпараметры для обучения
GAMMA             = 0.99
ALPHA             = 0.0005
EPOCHS            = 4
EPISODES          = 500

GYM_NAME          = "CartPole-v0"
N_ACTION          = 2
INPUT_DIMS        = 4
BATCH_SIZE        = 5
N                 = 20

TAKE_LAST_AVG     = 100



env         = gym.make(GYM_NAME)
agent       = Agent(n_actions=N_ACTION, batch_size=BATCH_SIZE, alpha=ALPHA,\
                n_epochs=EPOCHS, input_dims=INPUT_DIMS)


figure_file = "plots/cartpole_ppo.png"
best_score  = env.reward_range[0]
score_history = []
avg_history   = []

learn_iters = 0
avg_score   = 0
n_steps     = 0

for i in range(EPISODES):
    observation = env.reset()
    done  = False
    score = 0
    while not done:
        action, prob, val = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        n_steps += 1
        score += reward
        agent.remember(observation, action, prob, val, reward, done)

        if n_steps % N == 0:
            agent.learn()
            learn_iters += 1

        observation = observation_

    score_history.append(score)
    avg_score = np.mean(score_history[-TAKE_LAST_AVG:])
    avg_history.append(avg_score)

    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()

    print ("Episode:", i,
        "Score:", score,
        "Avg_score:", avg_score,
        "Time steps:", n_steps,
        "Learning steps:", learn_iters)

# ? Созранение массива средних значений в JSON файл
data        = json_get_data()
data["ppo_scores"] = score_history
data["ppo_avg"]    = avg_history
json_wrt_data(data)