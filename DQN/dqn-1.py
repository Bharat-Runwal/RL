from DQN.ModifiedTensorBoard import ModifiedTensorBoard
from keras.models import Sqeuential
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque
import numpy as np 
import time 
import random


REPLAY_MEMORY_SIZE =50_000
MODEL_NAME = "256x2"
MIN_REPLAY_MEMORY_SIZE =10_000
MINIBATCH_SIZE = 64
DISCOUNT = 0.99


replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

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
        model = Sqeuential()
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

            


        