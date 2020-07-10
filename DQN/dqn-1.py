from ModifiedTensorBoard import ModifiedTensorBoard
from BlobEnv import BlobEnv
import tensorflow as tf
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque
import numpy as np 
import time 
import random
import os 



DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
update_target = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 20_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False


replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)


env = BlobEnv()

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')



class DQNAgent:

    def __init__(self):
        #Main Model
        self.model = self.create_model()
        
        # Target Model
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory=deque(maxlen=REPLAY_MEMORY_SIZE)
        
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")

        self.target_update_counter = 0 


    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256,(3,3),input_shape=env.OBSERVATION_SPACE_VALUES,activation='relu'))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))

        model.add(Conv2D(256,(3,3),activation='relu'))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(env.ACTION_SPACE_SIZE,activation='linear'))

        model.compile(loss="mse",optimizer = Adam(lr=0.001),metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self,transition):
        replay_memory.append(transition)

    def get_qs(self,state):
        return self.model.predict(np.array(state).reshape(-1,*state.shape)/255)[0]

    
    def train(self, terminal_state,step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return 
        
        minibatch = random.sample(replay_memory,MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X=[]
        y=[]

        for index ,(current_state,action,reward,new_current_state,done) in enumerate(minibatch):
            if not done:
                max_future_q= np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q=reward 

            current_qs = current_qs_list[index]
            current_qs[action] =new_q

            X.append(current_state)
            y.append(current_qs)
        

        self.model.fit(np.array(X),np.array(y),batch_size=MINIBATCH_SIZE,verbose=0,shuffle=False,callbacks=[self.tensorboard] if terminal_state else None )

        if terminal_state :
            self.target_update_counter +=1
        
        if self.target_update_counter > update_target :
            self.target_model.set_weights(self.model.get_weights())
        

agent =DQNAgent()

for episode in tqdm(range(1,EPISODES+1),ascii=True,unit="episode"):
    agent.tensorboard.step =episode

    episode_reward =0
    step =1
    current_state =env.reset()

    done = False

    while not done:
        if np.random.random() <= epsilon:
            action =np.random.randint(0,env.ACTION_SPACE_SIZE)
        else:
            action =np.argmax(agent.get_qs(current_state))

        new_state,reward,done = env.step(action)
        episode_reward+=reward 

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()
        
        agent.update_replay_memory((current_state,action,reward,new_state,done))
        agent.train(done,step)

        current_state = new_state

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)