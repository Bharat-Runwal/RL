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
    #i.e action space is discrete and # is 4 
    def action(self,act):
        if act==0:
            self.move(x=1,y=1)
        elif act==1:
            self.move(x=-1,y=-1)
        elif act==2:
            self.move(x=-1,y=1)
        elif act==3:
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

        #NON- ACCESSIBLE MOVES
        if self.x<0:
            self.x = 0
        elif self.x >SIZE-1:
            self.x =SIZE-1

        if self.y<0:
            self.y = 0
        elif self.y >SIZE-1:
            self.y =SIZE-1        
       
if start_q_table is None:
    q_table={}
    for x1 in range(-SIZE+1,SIZE):
        for y1 in range(-SIZE+1,SIZE):
            for x2 in range(-SIZE+1,SIZE):
                for y2 in range(-SIZE+1,SIZE):
                    q_table[((x1,y1),(x2,y2))] = [np.random.unifrom(-5,0) for i in range(4)]

else:
    with open(start_q_table,"rb") as f:
        q_table=pickle.load(f)

episode_rewards =[]
for episode in range(Episodes):
    player = Blob()
    food = Blob()
    enemy =Blob()

    if episode % show_every ==0:
        print("on epsiode {},epsilon:{}".format(episode,epsilon))
        print("{} ep mean :{}".format(show_every,np.mean(episode_rewards[-show_every:])))
        show =True
    else:
        show =False

    episode_reward= 0
    
    for i in range(200):
        obsv= (player-food,player-enemy)

        if np.random.random() <=epsilon:
            action = np.random.randint(0,4)
        else:
            action =np.max(q_table[obsv])

        player.action(action)

        #enemy.move()
        #food.move()

        #if you end up meeting enemy 
        if player.x ==enemy.x and player.y == enemy.y:
            reward = -enemy_penalty
        elif player.x ==food.x and player.y ==food.y:
            reward = food_reward
        else:
            reward = -move_penalty

        new_obs = (player-food , player-enemy)
        max_q_fut = np.max(q_table[new_obs])
        curr_q = q_table[obsv][action]

        if reward == food_reward:
            new_q = food_reward
        elif reward == -enemy_penalty:
            new_q = -enemy_penalty
        else:
            new_q=(1-lr)*curr_q+lr*(discount*reward+max_q_fut)

        q_table[obsv][action]= new_q
        
        # ENV SHOW 
        if show:
            env= np.zeros((SIZE,SIZE,3),dtype=np.uint8)
            env[player.y][player.x] =d[player_n]
            env[food.y][food.x] =d[food_n]
            env[enemy.y][enemy.x] =d[enemy_n]

            img = Image.fromarray(env, "RGB")
            img = img.resize((300,300))

            cv2.imshow("Small world",ap.array(img))

            if reward ==food_reward or reward == -enemy_penalty:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
                else:
                    if cv2.waitKey(1) & 0xFF ==ord("q"):
                        break

            episode_reward += reward
            if reward ==food_reward or reward == -enemy_penalty:
                break

    episode_rewards.append(episode_reward)
    epsilon *=eps_decay

moving_avg = np.convolve(episode_rewards,np.ones((show_every,))/ show_every,mode='valid')

plt.plot([i for i in range(len(moving_avg))],moving_avg)
plt.ylabel("reward{}m_avg".format(show_every))
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle","wb") as f:
    pickle.dump(q_table,f)

    


