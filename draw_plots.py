from json_impl import json_get_data
import matplotlib.pyplot as plt
data = json_get_data()

plt.subplot(1, 3, 1)
plt.title("Policy Gradient")
plt.ylabel("Score")
plt.xlabel("Episodes")
plt.plot(data["pg_scores"], label="Current score")
plt.plot(data["pg_avg"], label="Average score")
plt.grid(True)
plt.legend()
plt.subplot(1, 3, 2)
plt.title("PPO")
plt.ylabel("Score")
plt.xlabel("Episodes")
plt.plot(data["ppo_scores"], label="Current score")
plt.plot(data["ppo_avg"], label="Average score")
plt.grid(True)
plt.legend()
plt.subplot(1, 3, 3)
plt.title("PG vs PPO")
plt.ylabel("Score")
plt.xlabel("Episodes")
plt.plot(data["pg_avg"], label="PG")
plt.plot(data["ppo_avg"], label="PPO")
plt.grid(True)
plt.legend()
plt.show()