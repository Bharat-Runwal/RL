import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style 
import time 

style.use("ggplot")

#config env
SIZE = 10
Episodes = 25000
move_penalty = 1 
enemy_penalty =300
food_reward =25 #don't know exact just chosen

lr = 0.1
discount =0.95
epsilon =0.9 
eps_decay =0.9998
show_every =3000

start_q_table = None # or filename 

player_n =1
food_n =2
enemy_n =3

#BGR color channel
d= {1:(255,175,0),
    2:(0,255,0)
    ,3:(0,0,255)}


class Blob:
    def __init__(self):
        self.x = np.random.randint(0,SIZE)
        self.y = np.random.randint(0,SIZE)
    
    def __sub__(self,other):
        return (self.x - other.x,self.y-other.y) 

    #We are defining only diagonal movement not up and down
    def action(self,act):
        if act==0:
            self.move(x=1,y=1)
        elif act==1:
            self.move(x=-1,y=-1)
        elif act==2:
            self.move(x=-1,y=1)
        elif act==0:
            self.move(x=1,y=-1)

    def move(self,x=False,y=False):
        if not x:
            self.x += np.random.randint(-1,2)
        else:
            self.x +=x 
        if not y:
            self.y += np.random.randint(-1,2)
        else:
            self.y +=y        
       