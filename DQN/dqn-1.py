from keras.models import Sqeuential
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque
import time 


REPLAY_MEMORY_SIZE =50_000
MODEL_NAME = "256x2"









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
        