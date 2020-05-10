"""
Main file that loads the agent class, creats the environment, and trains the models, saves the models and graphs.
Author: Hari Vidharth, S4031180
"""

import gc
import time

import gym
import numpy as np
from DDQNAgent import DDQNAgent
from gym_unity.envs import UnityToGymWrapper
from keras import backend as k
from matplotlib import pyplot as plt
from mlagents_envs.environment import UnityEnvironment


def train(path):
    # env = gym.make("LunarLander-v2")
    # env = wrappers.Monitor(env, "tmp/lunar-lander", video_callable=lambda episode_id: True, force=True)
    unityenv = UnityEnvironment(path)
    env = UnityToGymWrapper(unity_env=unityenv, flatten_branched=True)
    ddqnAgent = DDQNAgent(alpha=0.0001, gamma=0.99, nActions=7, epsilon=1.0, batchSize=512, inputShape=210)
    nEpisodes = 1000
    ddqnScores = []
    ddqnAverageScores = []
    epsilonHistory = []
    stepsPerEpisode = []
    for episode in range(nEpisodes):
        StartTime = time.time()
        done = False
        score = 0
        steps = 0
        observation = env.reset()
        while not done:
            action = ddqnAgent.chooseAction(observation)
            observationNew, reward, done, info = env.step(action)
            score += reward
            ddqnAgent.remember(state=observation, stateNew=observationNew, action=action, reward=reward, done=done)
            observation = observationNew
            ddqnAgent.learn()
            steps += 1
        epsilonHistory.append(ddqnAgent.epsilon)
        ddqnScores.append(score)
        averageScore = np.mean(ddqnScores)
        ddqnAverageScores.append(averageScore)
        stepsPerEpisode.append(steps)
        ElapsedTime = time.time() - StartTime
        ElapsedTime = ElapsedTime / 60
        print("Episode:", episode, "Score: %.2f" % score, "Average Score: %.2f" % averageScore, "Run Time:",
              ElapsedTime, "Minutes", "Epsilon:", ddqnAgent.epsilon, "Steps:", steps)
        if episode > 1 and episode % 9 == 0:
            ddqnAgent.saveModel()
    env.close()
    x = [i for i in range(nEpisodes)]
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("DDQN Hallway")
    ax1.plot(x, ddqnScores, "C1")
    ax1.set_title('Episodes vs Scores')
    ax1.set(xlabel='Episodes', ylabel='Scores')
    ax2.plot(x, ddqnAverageScores, "C2")
    ax2.set_title('Episodes vs Average Scores')
    ax2.set(xlabel='Episodes', ylabel='Average Scores')
    ax3.plot(x, epsilonHistory, "C3")
    ax3.set_title('Episodes vs Epsilon Decay')
    ax3.set(xlabel='Episodes', ylabel='Epsilon Decay')
    ax4.plot(x, stepsPerEpisode, "C4")
    ax4.set_title('Episodes vs Steps Per Epsisode')
    ax4.set(xlabel='Episodes', ylabel='Steps')
    plt.savefig('Hallway.png')


if __name__ == "__main__":
    basic = "..//windowsBasic//UnityEnvironment.exe"
    pushBlock = "..//windowsPushBlock//UnityEnvironment.exe"
    hallway = "..//windowsHallway//UnityEnvironment.exe"
    k.clear_session()
    train(path=pushBlock)
    k.clear_session()
    gc.collect()
