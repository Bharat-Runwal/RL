from keras.models import Sqeuential
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

class DQNAgent:

    def __init__(self):
        #Main Model
        self.model = self.create_model()
        
        # Target Model
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())


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
        