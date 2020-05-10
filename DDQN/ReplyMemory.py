"""
File contains class that defines the experience replay.
Author: Hari Vidharth, S4031180
"""

import numpy as np


class ReplayMemory(object):
    def __init__(self, maxSize, inputShape, nActions, discrete=False):
        """
        Initialising the replay memory class with the allotted memory size to save the data.
        """
        self.memoryCounter = 0
        self.discrete = discrete
        dataType = np.int32 if self.discrete else np.float32
        self.memorySize = maxSize
        self.memoryState = np.zeros((self.memorySize, inputShape), dtype=np.float32)
        self.memoryStateNew = np.zeros((self.memorySize, inputShape), dtype=np.float32)
        self.memoryAction = np.zeros((self.memorySize, nActions), dtype=dataType)
        self.memoryReward = np.zeros(self.memorySize, dtype=np.float32)
        self.memoryTerminal = np.zeros(self.memorySize, dtype=np.float32)

    def saveTransitions(self, state, stateNew, action, reward, done):
        """
        Method to save the data into the initialsed memory to be sampled later.
        """
        index = self.memoryCounter % self.memorySize
        self.memoryState[index] = state
        self.memoryStateNew[index] = stateNew
        if self.discrete:
            actions = np.zeros(self.memoryAction.shape[1])
            actions[action] = 1.0
            self.memoryAction[index] = actions
        else:
            self.memoryAction[index] = action
        self.memoryReward[index] = reward
        self.memoryTerminal[index] = 1 - int(done)
        self.memoryCounter += 1

    def sampleMemory(self, batchSize):
        """
        Method to sample a mini batch of data stored in the memory to be used for training the agent.
        """
        memoryMax = min(self.memoryCounter, self.memorySize)
        batch = np.random.choice(memoryMax, batchSize)
        states = self.memoryState[batch]
        statesNew = self.memoryStateNew[batch]
        actions = self.memoryAction[batch]
        reward = self.memoryReward[batch]
        terminal = self.memoryTerminal[batch]

        return states, statesNew, actions, reward, terminal
