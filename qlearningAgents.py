# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math


class QLearningAgent(ReinforcementAgent):
    """
    Q-Learning Agent

    Functions you should fill in:
      - computeValueFromQValues
      - computeActionFromQValues
      - getQValue
      - getAction
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # qvalue를 여기서 초기화하라고?
        self.values = util.Counter()

    def getQValue(self, state, action):
        """
        Returns Q(state,action)
        Should return 0.0 if we have never seen a state
        or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.values[(state, action)]

    def computeValueFromQValues(self, state):
        """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        qValue = float("-inf")

        # 왜 리스트컴프리헨션도 통제할까?
        legalActions = self.getLegalActions(state)

        if not legalActions:
            return 0

        for action in legalActions:
            currentQvalue = self.getQValue(state, action)
            qValue = max(qValue, currentQvalue)

        return qValue

        # util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
        Compute the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
        """
        "*** YOUR CODE HERE ***"

        def pickQValue(pair):
            qValue, action = pair
            return qValue

        valueActionPairs = [
            (self.getQValue(state, action), action)
            for action in self.getLegalActions(state)
        ]

        if not valueActionPairs:
            return None

        value, action = max(valueActionPairs, key=pickQValue)

        return action
        # 그냥내놔
        # util.raiseNotDefined()

    def getAction(self, state):
        """
        Compute the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions, which is the case at the terminal state, you
        should choose None as the action.

        HINT: You might want to use util.flipCoin(prob)
        HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        action = self.getPolicy(state)

        # if random.random() < self.epsilon:
        #     legalActionIndex=random.randint(0, len(legalActions)-1)
        #     return legalActions[legalActionIndex]
        # else:
        #     return self.getPolicy(state)

        return action

    def update(self, state, action, nextState, reward):
        """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here

        NOTE: You should never call this function,
        it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        # qlearing식 그대로 옮겨쓰기
        oldQValue = self.getQValue(state, action)
        nextQValue = self.getValue(nextState)  # optimal Value6

        newQValue = (1 - self.alpha) * oldQValue + self.alpha * (
            reward + self.discount * nextQValue
        )

        self.values[(state, action)] = newQValue

        # print(newQValue)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args["epsilon"] = epsilon
        args["gamma"] = gamma
        args["alpha"] = alpha
        args["numTraining"] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
    ApproximateQLearningAgent

    You should only have to overwrite getQValue
    and update.  All other QLearningAgent functions
    should work as is.
    """

    def __init__(self, extractor="IdentityExtractor", **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
        Should return Q(state,action) = w * featureVector
        where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # identityExtractor=IdentityExtractor()

        def dotProduct(features, weights):
            return sum(features[key] * weights[key] for key in features)

        weights = self.getWeights()
        features = self.featExtractor.getFeatures(state, action)

        dp = dotProduct(features, weights)
        # print(dp)

        return dp

    def update(self, state, action, nextState, reward):
        """
        Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # 이거 그냥 식 카피하면 되잖아
        weights = self.getWeights()
        features = self.featExtractor.getFeatures(state, action)

        difference = (
            reward
            + self.discount * self.getValue(nextState)
            - self.getQValue(state, action)
        )

        for key in features:
            feature = features[key]
            oldWeight = weights[key]
            newWeight = oldWeight + (self.alpha * difference * feature)
            weights[key] = newWeight

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
