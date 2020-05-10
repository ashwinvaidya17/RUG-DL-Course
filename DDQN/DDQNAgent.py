"""
File contains the DDQN Agent that uses the DDQN neural network model and Replay memory for training.
Author: Hari Vidharth, S4031180
"""

import numpy as np
from DDQNModel import DDQNModel
from ReplyMemory import ReplayMemory
from keras.models import load_model


class DDQNAgent(object):
    def __init__(self, alpha, gamma, epsilon, nActions, batchSize, inputShape, epsilonDecay=0.99999, epsilonMin=0.01,
                 memorySize=1000000, fileName="ddqnModelBasic.h5", replaceTarget=100):
        """
        Initialize the agent with the replay memory and the ddqn model.
        """
        self.nActions = nActions
        self.actionSpace = [i for i in range(self.nActions)]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.epsilonMin = epsilonMin
        self.batchSize = batchSize
        self.fileName = fileName
        self.replaceTarget = replaceTarget
        self.memory = ReplayMemory(maxSize=memorySize, inputShape=inputShape, nActions=nActions, discrete=True)
        self.qEval = DDQNModel().buildModel(learningRate=alpha, nActions=nActions, inputShape=inputShape, fc1Dims=512,
                                            fc2Dims=512)
        self.qTarget = DDQNModel().buildModel(learningRate=alpha, nActions=nActions, inputShape=inputShape, fc1Dims=512,
                                              fc2Dims=512)

    def remember(self, state, stateNew, action, reward, done):
        """
        Agent saves the observations/data into memory.
        """
        self.memory.saveTransitions(state=state, stateNew=stateNew, action=action, reward=reward, done=done)

    def chooseAction(self, state):
        """
        Agent choosing action as per the epsilon greedy policy.
        """
        state = state[np.newaxis, :]
        randomValue = np.random.random()
        if randomValue < self.epsilon:
            action = np.random.choice(self.actionSpace)
        else:
            actions = self.qEval.predict(state)
            action = np.argmax(actions)

        return action

    def updateNetworkParameters(self):
        """
        Updating the network parameters after a set threshold.
        """
        self.qTarget.model.set_weights(self.qEval.model.get_weights())

    def learn(self):
        """
        The method used for learning, sampling batch size and training the model, updating the parameters and epsilon decay.
        """
        if self.memory.memoryCounter > self.batchSize:
            state, stateNew, action, reward, done = self.memory.sampleMemory(self.batchSize)
            actionValues = np.array(self.actionSpace, dtype=np.int8)
            actionIndeces = np.dot(action, actionValues)
            qNext = self.qTarget.predict(stateNew)
            qEval = self.qEval.predict(stateNew)
            qPredction = self.qEval.predict(state)
            maxActions = np.argmax(qEval, axis=1)
            qTarget = qPredction
            batchIndex = np.arange(self.batchSize, dtype=np.int32)
            qTarget[batchIndex, actionIndeces] = reward + self.gamma * qNext[batchIndex, maxActions.astype(int)] * done
            _ = self.qEval.fit(state, qTarget, verbose=0)
            self.epsilon = self.epsilon * self.epsilonDecay if self.epsilon > self.epsilonMin else self.epsilonMin
            if self.memory.memoryCounter % self.replaceTarget == 0:
                self.updateNetworkParameters()

    def saveModel(self):
        self.qEval.save(self.fileName)
        print("Saving Model!")

    def loadModel(self, fileName):
        self.qEval = load_model(fileName)
        if self.epsilon <= self.epsilonMin:
            self.updateNetworkParameters()
        print("Loading Model!")
