import gym
import numpy as np
import matplotlib.pyplot as plt
# sdasad
env = gym.make("MountainCar-v0")
episodes =2000
lr=0.1
Show_every = 500
discount = 0.95

  
# For stats
STATS_EVERY =10
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

# env.reset()

# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.action_space.n)

disc_os_size = [20] *len(env.observation_space.high)
disc_os_window = (env.observation_space.high - env.observation_space.low) / disc_os_size
# print(disc_os_window)

epsilon =0.5
start_epsilon = 1
end_epsilon = episodes // 2
epsilon_decay = (epsilon)/(end_epsilon-start_epsilon)
q_table = np.random.uniform(low=-2 , high= 0 , size = (disc_os_size+[env.action_space.n]))
# print(q_table.shape)
# print(q_table)

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/disc_os_window
    return tuple(discrete_state.astype(np.int))


for episode in range(episodes):
    episode_reward = 0

    if episode%Show_every ==0:
        print(episode)
        render = True

    else:
        render =False

    discrete_state= get_discrete_state(env.reset())
    done = False
    while not done:
        if np.random.random() >epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0,env.action_space.n)
        new_state,reward,done,_ = env.step(action)
        episode_reward+=reward
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()
        if not done:
            max_q = np.max(q_table[new_discrete_state])
            curr_q = np.max(q_table[discrete_state+(action,)])

            new_q = (1-lr)*curr_q + lr*(reward+discount*max_q)
            q_table[discrete_state+(action,)] = new_q
        elif new_state[0] >= env.goal_position:
            print(f"We made it at episdoe{episode}")
            q_table[discrete_state,(action,)]=0
        
        discrete_state = new_discrete_state

    if end_epsilon>=episode >=start_epsilon:
        epsilon-=epsilon_decay
    ep_rewards.append(episode_reward)

    if not episode % STATS_EVERY:
        average_reward = sum(ep_rewards[-STATS_EVERY:])/STATS_EVERY
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
        print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.show()