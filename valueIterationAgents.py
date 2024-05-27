# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A ValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs value iteration
    for a given number of iterations using the supplied
    discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your value iteration agent should take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.

        Some useful mdp methods you will use:
            mdp.getStates()
            mdp.getPossibleActions(state)
            mdp.getTransitionStatesAndProbs(state, action)
            mdp.getReward(state, action, nextState)
            mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        from gridworld import Gridworld

        mdp: Gridworld = self.mdp
        states = mdp.getStates()

        def getKValues():
            kValues = util.Counter()

            for state in states:
                actions = mdp.getPossibleActions(state)

                if not actions:
                    continue

                kValues[state] = max(
                    [self.computeQValueFromValues(state, action) for action in actions]
                )

            # self.values = kValues
            return kValues

        # 새로 생산된 k번째 valeus들이 변경되면 안되겠네..?
        # copy한후 대입하는거밖엔...
        for iter in range(self.iterations):
            self.values = getKValues()

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
        Compute the Q-value of action in state from the
        value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        from gridworld import Gridworld

        mdp: Gridworld = self.mdp
        stateProbs = mdp.getTransitionStatesAndProbs(state, action)

        value = sum(
            [
                prob
                * (
                    mdp.getReward(state, action, nextState)
                    + self.discount * self.getValue(nextState)
                )
                for (nextState, prob) in stateProbs
            ]
        )

        return value

    def computeActionFromValues(self, state):
        """
        The policy is the best action in the given state
        according to the values currently stored in self.values.

        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # policy extraction..?
        from gridworld import Gridworld

        def pickQValue(pair):
            qValue, action = pair
            return qValue

        mdp: Gridworld = self.mdp
        # getAction
        actions = mdp.getPossibleActions(state)

        # 만약 legal한 액션이 없으면 None을 반환해라
        if not actions:
            return None

        (qValue, action) = max(
            [(self.getQValue(state, action), action) for action in actions],
            key=pickQValue,
        )
        return action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A PrioritizedSweepingValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs prioritized sweeping value iteration
    for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
        Your prioritized sweeping value iteration agent should take an mdp on
        construction, run the indicated number of iterations,
        and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        from gridworld import Gridworld

        mdp: Gridworld = self.mdp
        priorityQueue = util.PriorityQueue()
        states = mdp.getStates()

        # # super().runValueIteration()

        def flatten(lst):
            return [item for sublist in lst for item in sublist]

        # getPrecessor
        predecessors = {}
        for state in states:
            if mdp.isTerminal(state):
                continue

            successorsPerAction = [
                mdp.getTransitionStatesAndProbs(state, action)
                for action in mdp.getPossibleActions(state)
            ]

            successors = flatten(successorsPerAction)

            for successor, prob in successors:
                if successor in predecessors:
                    predecessors[successor].add(state)
                else:
                    predecessors[successor] = set({state})

        for state in states:
            currentValue = self.getValue(state)
            actions = mdp.getPossibleActions(state)
            if not actions:
                continue
            highestQValue = max([self.getQValue(state, action) for action in actions])
            diff = abs(currentValue - highestQValue)
            priorityQueue.update(state, -diff)

        for iter in range(self.iterations):
            if priorityQueue.isEmpty():
                break

            currentState = priorityQueue.pop()
            if not mdp.isTerminal(currentState):
                values = [
                    self.getQValue(currentState, action)
                    for action in mdp.getPossibleActions(currentState)
                ]
                self.values[currentState] = max(values)

            for predecessor in predecessors[currentState]:
                if mdp.isTerminal(predecessor):
                    continue
                predecessorValue = self.getValue(predecessor)
                actions = mdp.getPossibleActions(predecessor)
                # if not actions:
                #     continue
                highestPredecessorQValue = max(
                    [self.getQValue(predecessor, action) for action in actions]
                )
                diff = abs(predecessorValue - highestPredecessorQValue)

                if diff > self.theta:
                    priorityQueue.update(predecessor, -diff)
