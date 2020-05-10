import keras.backend as k
import keras
from keras.layers import Conv2D, Dense, Input, Flatten, BatchNormalization
from keras.models import Model
import numpy as np


class policy:
    def __init__(self):
        self.learning_rate=0.001
        self.epsilon=0.2
        self.gamma_val=0.99
        self.entropy_beta=0.001
        self.activation="relu"
        self.actor_fname="actor_model"
        self.critic_fname="critic_model"

    def save_model(self,model,filename):
        model.save(filename+'.h5')

    def advantage_function(self,rewards,values,mask):
        return_value=[]
        for val in reversed(range(len(rewards))):
            adv=rewards[val]+self.gamma_val*values[val+1]*mask[val] - values[val]
            return_value.insert(0,adv)
        adv=np.array(return_value) - values[:-1]
        return return_value, adv

    def ppo_loss(self,old_policy,advantage):
        def loss(y_true,y_pre):
            new_policy=y_pre
            r_theta=k.exp(k.log(new_policy + 1e-10)-k.log(old_policy + 1e-10))
            clip_val=k.clip(r_theta,min_value=1-self.epsilon,max_value=1+self.epsilon)
            actor_loss=-k.mean(k.minimum(r_theta*advantage,clip_val*advantage))
            #critic_loss=k.mean(k.square(rewards-values))
            total_loss=actor_loss-self.entropy_beta*k.mean(
            -(new_policy*k.log(new_policy+1e-10))
            )
            return total_loss
        return loss

    def actor_network(self,inp_dim,out_dim,n_actions):
        """ network model with actor policy """
        inp_state=Input(shape=(inp_dim),name="state_space_input")
        policy_old=Input(shape=(out_dim,),name="old_ploicy")
        advantage=Input(shape=(1,1,),name="advanatge_fucntion")
        dense1=Dense(64,activation=self.activation,name='fc1')(inp_state)
        dense2=Dense(32,activation=self.activation,name='fc2')(dense1)
        dense3=Dense(32,activation=self.activation,name='fc3')(dense2)
        out_predict=Dense(n_actions,activation='softmax',name='output_layer')(dense3)
        actor_model=Model(inputs=[inp_state,policy_old,advantage],output=[out_predict])
        actor_model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),loss=[
        self.ppo_loss(policy_old,advantage)
        ])
        actor_model.summary()
        self.save_model(actor_model,self.actor_fname)
        return actor_model

    def critic_network(self,inp_dim):
        """" network model with critic value """
        inp_state=Input(shape=inp_dim,name="state_space_input")
        dense1=Dense(64,activation=self.activation,name='fc1')(inp_state)
        dense2=Dense(32,activation=self.activation,name='fc2')(dense1)
        dense3=Dense(32,activation=self.activation,name='fc3')(dense2)
        out_predict=Dense(1,name='prediction')(dense3)
        critic_model=Model(inputs=[inp_state],output=[out_predict])
        critic_model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),loss='mean_squared_error')
        critic_model.summary()
        self.save_model(critic_model,self.critic_fname)
        return critic_model
