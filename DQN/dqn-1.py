from keras.models import Sqeuential
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D
from keras.callbacks import TensorBoard

class DQNAgent:
    def create_model(self):
        model = Sqeuential()
        
        