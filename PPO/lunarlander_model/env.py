""" Game environment """

import gym
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

import numpy as np
import keras.backend as k
from keras.layers import Input
from keras.models import Model
from keras.models import load_model
from keras.callbacks import TensorBoard
import argparse
from policy_network import policy
from tensorboardX import SummaryWriter
from time import time
import matplotlib as m
from matplotlib import pyplot as plt


def load(filename):
    return load_model(filename)

def get_old_prediction(actor_model_old,state_inp,dummy_old,dummy_advantage):
    return actor_model_old.predict([state_inp,dummy_old,dummy_advantage],steps=1).flatten()

def update_network(actor_model,actor_model_old):
    actor_model_old.set_weights(actor_model.get_weights())

def get_action(state_inp,actor_model,dummy_old,dummy_advantage,action_space,episode):
    action_val=actor_model.predict([state_inp,dummy_old,dummy_advantage],steps=1).flatten()
    count=0
    for i in range(len(action_val)):
        if np.isnan(action_val[i]): action_val[i]=np.nan_to_num(action_val[i])
        if action_val[i]==0:
            count+=1
    if count >= 3:
        action=np.random.choice(action_space)
        count=0
    else:
        action=np.random.choice(action_space,p=action_val)
    return action

def draw(x,y,title,x_label,y_label,color,f_name):
    plt.figure()
    plt.plot(x,y,color=color)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.draw()
    plt.savefig(f_name+".jpg")
    plt.show()
    
    
def train_env(env,action_space,state_space,state,total_runs):
    """ training """
    # parameters
    epochs=1
    runs_completed=0
    update_thresh=5 # exploration threshold
    thresh_flag=False
    t_score=[]
    avg_reward=[]
    step_lst=[]
    dummy_old=np.zeros((1,action_space))
    dummy_advantage=np.zeros((1,1,1))
    s_writter=SummaryWriter("logs/")

    actor_model=policy().actor_network(inp_dim=state_space,out_dim=action_space,n_actions=action_space)
    critic_model=policy().critic_network(inp_dim=state_space)
    actor_model_old=policy().actor_network(inp_dim=state_space,out_dim=action_space,n_actions=action_space)
    actor_model_old.set_weights(actor_model.get_weights())

    for episode in range(total_runs):
        curr_time=time()
        t_reward=0
        steps_per_episode=0
        state=env.reset()
        store_rewards=[]
        store_actions=[]
        store_values=[]
        store_states=[]
        store_mask=[]
        action_prb=[]
        done_flag=False
        
        while not done_flag:
            state_inp=k.expand_dims(state,0)
            action=get_action(state_inp,actor_model,dummy_old,dummy_advantage,action_space,episode)
            n_state,reward,done,info=env.step(action)
            q_val=critic_model.predict([state_inp],steps=1)
            print(f"Episode: {episode}, action:{action}, reward:{reward}")
            mask=not done
            runs_completed+=1
            steps_per_episode+=1
            t_reward+=reward
            
            # memory
            store_actions.append(action)
            store_rewards.append(reward)
            store_values.append(q_val)
            store_states.append(state)
            store_mask.append(mask)
            old_prb=get_old_prediction(actor_model_old,state_inp,dummy_old,dummy_advantage)
            action_prb.append(np.array(old_prb))
           
            # commit training
            if runs_completed % update_thresh ==0:
                update_network(actor_model,actor_model_old)
                q_val=critic_model.predict([state_inp],steps=1)
                store_values.append(q_val)
                returns,advantage=policy().advantage_function(store_rewards,store_values,store_mask)
                #update
                actor_loss=actor_model.fit([store_states,action_prb,advantage],
                [np.zeros(shape=(len(store_actions),action_space))],verbose=True,epochs=epochs,shuffle=True)
                critic_loss=critic_model.fit([store_states],[np.reshape(returns,newshape=(-1,1))
                ],verbose=True, shuffle=True,epochs=epochs)
                print("---------> network upated <-----------")

                # clear old memory
                store_rewards.clear()
                store_values.clear()
                store_actions.clear()
                store_mask.clear()
                store_states.clear()
                action_prb.clear()
                runs_completed=0
                thresh_flag=True

            # new state
            state=n_state
            if done:
                done_flag=True
                runs_completed=0
                env.reset()

        # store logs
        if thresh_flag == True:
            s_writter.add_scalar('Actor_loss',actor_loss.history['loss'][-1],episode)
            s_writter.add_scalar('Critic_loss',critic_loss.history['loss'][-1],episode)

        # compute score
        t_score.append(t_reward)
        avg_score=np.mean(t_score)
        avg_reward.append(avg_score)
        step_lst.append(steps_per_episode)
        print(f"Total reward :{avg_score} @ episode:{episode}")
        end_time=time()
        print(f"Time taken @ episode {episode} --> {end_time-curr_time}")
        # save models
        if episode % 10 == 0:
            print("Models saved")
            policy().save_model(actor_model,policy().actor_fname)
            policy().save_model(critic_model,policy().critic_fname)
        env.reset()

    env.close()
    # draw plot
    draw([a for a in range(total_runs)],avg_reward,"LunarLander","episodes","Avergae reward","blue","ll_avgreward")
    draw([a for a in range(total_runs)],t_score,"LunarLander","episodes","Total reward","red","ll_totalreward")
    draw([a for a in range(total_runs)],step_lst,"LunarLander","episodes","steps per episode","green","ll_steps_per_episode")

if __name__=="__main__":
    arg=argparse.ArgumentParser(description="PPO test")
    arg.add_argument('method',default=None,help="training or testing")
    args=arg.parse_args()

    env=gym.make("LunarLander-v2")
    state=env.reset()
    action_space=env.action_space.n
    state_space=env.observation_space.shape

    print("Action space:v",action_space)
    print("State space :v",state_space[0])

    total_runs=1000

    if args.method == 'train':
        train_env(env,action_space,state_space,state,total_runs)
    

    



    



