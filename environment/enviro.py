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
    