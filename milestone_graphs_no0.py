#Plays games of Pong using Open Ai Gym using a Q-learning Algorithm. 

import math, random
from collections import defaultdict
import gym
from gym import wrappers, logger
import numpy as np
from numba import jit
import sys
import collections, random

"""
Our QLearningAlgorithm Class has been adapted from the Blackjack assignment.
actions: list of all possible actions: 0, 2, or 3
discout: 1
featureExtractor: list of (feature, value) pairs extracted from current state
explorationProb: float probability of exploring (randomly or action) rather than following Qopt
"""
class QLearningAlgorithm():
    def __init__(self, actions, discount, featureExtractor, explorationProb):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        #for k, v in w:
            #self.weights[k] = v

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        features = self.featureExtractor(state, action)
        for f, v in features:
            score += self.weights[f] * v
        return score

    """Chooses an action following an epsilon greedy algorithm. Makes random choice with probability explorationProb/4,
    a baseline action with probabilty 3/4*explorationProb, and a choice based on Qopt with probability 1 - explorationProb
    """
    def getAction(self, state, done):
        if done:
            return 0
        #random choice
        if random.random() < self.explorationProb/4:
            return random.choice([2, 3])
        #baseline choice
        if random.random() < self.explorationProb:
            for i in range(35, 195):
                row = state[1][i]
                #ball is in row
                if [236, 236, 236] in row:
                    return 2 #go up
                #paddle is in row
                if [92, 186, 92] in row:
                    return 3 #go down
            #do nothing
            return 2
        #Qopt choice
        else:
            return max((self.getQ(state, action), action) for action in self.actions)[1]

    # Returns step size
    def getStepSize(self):
        return .001

    """
    state: (previous Observation, current Observation, score)
    action: action taken
    reward: +1 if scored, -1 if opponent scored, 0 otherwise
    newState: (current Observation, new Observation, score)
    Updates dictionary of weights according to state, new state, reward, and action taken.
    """
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
        if newState != None:
            Vopt = max(self.getQ(newState, a) for a in self.actions)
            for feature, value in self.featureExtractor(state, action):
                self.weights[feature] -= self.getStepSize()*(self.getQ(state, action) - (reward + self.discount*Vopt))*value
        # END_YOUR_CODE

    #Returns the dictionary current weights
    def getWeights(self):
        return self.weights
    
    """
    p = exploration Probability
    Sets the current exploration Probability to p 
    """
    def setExplorationProbability(self, p):
        self.explorationProb = p

    #returns current exploration probability
    def getExProb(self):
        return self.explorationProb

    def resetWeights(self):
        self.weights = defaultdict(float)

"""
obs = RGB array of frame
Determines ball and paddle locations based on their pixel colors. Each position is the uppermost 
pixel of eitehr the paddle or the ball 
Returns a numpy array of [row of ball, column of ball, row of paddle]
"""
@jit(nopython=True, cache=True)
def getPositions(obs):
    answer = np.zeros(3)
    for i in range(192, 37, -1):
        for j in range(0, 160):
            #ball
            if obs[i, j, 0] == 236 and obs[i, j, 1] == 236 and obs[i, j, 2] == 236:
                answer[0] = i
                answer[1] = j
            #paddle
            elif obs[i, j, 0] == 92 and obs[i, j, 1] == 186 and obs[i, j, 2] ==  92:
                answer[2] = i

            #elif obs[i, j, 0] == 213 and obs[i, j, 1] == 130 and obs[i, j, 2] ==  74:
                #answer[3] = min(i + 18, 195)
    return answer

"""
x: integer
Returns x rounded to the nearest 10
"""
def round(x):
    divided = (int(x)/10)*10
    if math.fabs(x - divided) > math.fabs(x -(divided + 10)):
        return divided + 10
    return divided

#
"""
state: tuple of previous Observation(RGB array), current Observation (RGB array), and score
action: 0, 2, or 3
Extracts features from a given state. 
Returns a feature list of (feature, value)
"""
def featureExtractorState(state, action):
    features = []
    curPos = getPositions(state[1])
    prevPos = getPositions(state[0])
    
    xvelocity = curPos[1] - prevPos[1]
    yvelocity = curPos[0] - prevPos[0]

    #whether x and y velocity are positive or negative 
    if xvelocity > 0:
        features.append((("vx+", action), 1))
    else:
        features.append((("vx-", action), 1))
    
    if yvelocity > 0:
        features.append((("vy+", action), 1))
    else:
        features.append((("vy-", action), 1))
    
    #rounded version of current Agent's paddle y position
    if round(curPos[2]) != 0:
        features.append((("agentPaddle", action, round(curPos[2])), 1))
    #rounded version of current ball's x and y position
    if round(curPos[0]) != 0:
        features.append((("ball", action, round(curPos[1]), round(curPos[0])), 1))
    #if round(curPos[3]) != 0:
        #features.append((("opPaddle", action, round(curPos[3])), 1))
    
    #Whether ball is above, below, or at same height of Agent's paddle
    if curPos[0] - curPos[2] > 36:
        features.append((("ballBelow", action), 1))
    elif curPos[0] - curPos[2] > 0:
        features.append((("ballSame", action), 1))
    else:
        features.append((("ballAbove", action), 1))

    return features

"""
Plays num_games games of Pong with actions directed by QLearningAlgorithm's getAction function, which is a function approximation by linear regression of a dictionary of features
of the current state and actions. 
"""
class QLearningAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done, q):
        return q.getAction(observation, done)

if __name__ == '__main__':
    f = open("output_milestone.txt", "w")

    logger.set_level(logger.WARN)

    #Opens a Pong environment
    env = gym.make('Pong-v0')

    #directory to output game statistics
    outdir = 'tmp/results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = QLearningAgent(env.action_space)
    
    
    cutoff = 40
    for j in range(10):
        cutoff += 10
        num_trained = 100
        for k in range(20):
            num_trained += 20
            score_list = []
            reward = 0
            done = False
            q = QLearningAlgorithm([2, 3], discount = 1, featureExtractor = featureExtractorState, explorationProb=.8)
            #plays num_games number of games 
            for i in range(num_trained + 101):
                
                new_observation = env.reset()
                observation = new_observation
                prev_obs = observation
                
                #number of points scored by agent
                score = 0
                while True:
                    #env.render() #allows video of game in progress to be shown 
                    action = agent.act((observation, new_observation, score), reward, done, q)
                    prev_obs = observation
                    observation = new_observation
                    new_observation, reward, done, _ = env.step(action) #go to next action
                    if reward == 1.0:
                        score += 1
                    
                    if i < num_trained:
                    #used during training
                	   q.incorporateFeedback((prev_obs, observation, score), action, reward, (observation, new_observation, score)) 
                    
                    #the end of the game has been reached
                    if done:
                        break

                if i == cutoff:
                    q.setExplorationProbability(0)
                
                if i > num_trained: 
                	score_list.append(score - 21)

                #print(str(i) + " games completed with current game's score " + str(score))
                if i == num_trained + 100:
                    f.write("hi")
                    f.write("\t".join([str(float(sum(score_list))/len(score_list)), str(cutoff), str(num_trained)]) + '\n')
                    print score_list
                    print sum(score_list)/len(score_list)
                    print str(cutoff)
                    print str(num_trained)
    #Closes the environment            
    f.close()
    env.close()