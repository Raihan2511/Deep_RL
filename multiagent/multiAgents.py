# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        foodList = newFood.asList()
        if len(foodList) > 0:
            distancesToFood = [manhattanDistance(newPos, foodPos) for foodPos in foodList]
            minFoodDist = min(distancesToFood)
            score += 10.0 / (minFoodDist + 1)
            score +=  - 2 * len(foodList) * 0.1
        else:
            score += 100

        for idx, ghostState in enumerate(newGhostStates):
            ghostPos = ghostState.getPosition()
            dist = manhattanDistance(newPos, ghostPos)
            scaredTime = newScaredTimes[idx]
            if scaredTime > 0:
                score += 5.0 / (dist + 1)
            else:
                if dist == 0:
                    score -= 500
                else:
                    score -= 10.0 / (dist)

        if action == Directions.STOP:
            score -= 2

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another an abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def minimax(state, agentIndex, depth):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if depth == self.depth:
                return self.evaluationFunction(state)

            nextAgent = agentIndex + 1
            nextDepth = depth
            if nextAgent == numAgents:
                nextAgent = 0
                nextDepth = depth + 1

            legalActions = state.getLegalActions(agentIndex)
            if len(legalActions) == 0:
                return self.evaluationFunction(state)

            if agentIndex == 0:
                bestValue = float('-inf')
                for action in legalActions:
                    succ = state.generateSuccessor(agentIndex, action)
                    val = minimax(succ, nextAgent, nextDepth)
                    if val > bestValue:
                        bestValue = val
                return bestValue
            else:
                bestValue = float('inf')
                for action in legalActions:
                    succ = state.generateSuccessor(agentIndex, action)
                    val = minimax(succ, nextAgent, nextDepth)
                    if val < bestValue:
                        bestValue = val
                return bestValue

        legalActionsRoot = gameState.getLegalActions(0)
        bestScore = float('-inf')
        bestAction = None
        for action in legalActionsRoot:
            succ = gameState.generateSuccessor(0, action)
            val = minimax(succ, 1 if numAgents > 1 else 0, 0 if numAgents > 1 else 1)
            if val > bestScore or bestAction is None:
                bestScore = val
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def alphabeta(state, agentIndex, depth, alpha, beta):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if depth == self.depth:
                return self.evaluationFunction(state)

            nextAgent = agentIndex + 1
            nextDepth = depth
            if nextAgent == numAgents:
                nextAgent = 0
                nextDepth = depth + 1

            legalActions = state.getLegalActions(agentIndex)
            if len(legalActions) == 0:
                return self.evaluationFunction(state)

            if agentIndex == 0:
                value = float('-inf')
                for action in legalActions:
                    succ = state.generateSuccessor(agentIndex, action)
                    value = max(value, alphabeta(succ, nextAgent, nextDepth, alpha, beta))
                    if value > alpha:
                        alpha = value
                    if value > beta:
                        return value
                return value
            else:
                value = float('inf')
                for action in legalActions:
                    succ = state.generateSuccessor(agentIndex, action)
                    value = min(value, alphabeta(succ, nextAgent, nextDepth, alpha, beta))
                    if value < beta:
                        beta = value
                    if value < alpha:
                        return value
                return value

        legalActionsRoot = gameState.getLegalActions(0)
        alpha = float('-inf')
        beta = float('inf')
        bestAction = None
        bestScore = float('-inf')
        for action in legalActionsRoot:
            succ = gameState.generateSuccessor(0, action)
            val = alphabeta(succ, 1 if numAgents > 1 else 0, 0 if numAgents > 1 else 1, alpha, beta)
            if val > bestScore or bestAction is None:
                bestScore = val
                bestAction = action
            if val > alpha:
                alpha = val
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def expectimax(state, agentIndex, depth):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if depth == self.depth:
                return self.evaluationFunction(state)

            nextAgent = agentIndex + 1
            nextDepth = depth
            if nextAgent == numAgents:
                nextAgent = 0
                nextDepth = depth + 1

            legalActions = state.getLegalActions(agentIndex)
            if len(legalActions) == 0:
                return self.evaluationFunction(state)

            if agentIndex == 0:
                bestValue = float('-inf')
                for action in legalActions:
                    succ = state.generateSuccessor(agentIndex, action)
                    val = expectimax(succ, nextAgent, nextDepth)
                    if val > bestValue:
                        bestValue = val
                return bestValue
            else:
                total = 0.0
                for action in legalActions:
                    succ = state.generateSuccessor(agentIndex, action)
                    val = expectimax(succ, nextAgent, nextDepth)
                    total += val
                return total / len(legalActions)

        legalActionsRoot = gameState.getLegalActions(0)
        bestScore = float('-inf')
        bestAction = None
        for action in legalActionsRoot:
            succ = gameState.generateSuccessor(0, action)
            val = expectimax(succ, 1 if numAgents > 1 else 0, 0 if numAgents > 1 else 1)
            if val > bestScore or bestAction is None:
                bestScore = val
                bestAction = action
        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    foodList = food.asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [g.scaredTimer for g in ghostStates]
    capsules = currentGameState.getCapsules()

    score = currentGameState.getScore()

    foodCount = len(foodList)
    if foodCount > 0:
        dists = [manhattanDistance(pos, f) for f in foodList]
        minFoodDist = min(dists)
    else:
        minFoodDist = 0

    capsuleCount = len(capsules)
    if capsuleCount > 0:
        capsuleDists = [manhattanDistance(pos, c) for c in capsules]
        minCapsuleDist = min(capsuleDists)
    else:
        minCapsuleDist = 0

    ghostDists = [manhattanDistance(pos, g.getPosition()) for g in ghostStates]
    if len(ghostDists) > 0:
        minGhostDist = min(ghostDists)
    else:
        minGhostDist = float('inf')

    evalScore = score * 1.0

    if foodCount > 0:
        evalScore += 10.0 / (minFoodDist + 1)
        evalScore += -4.0 * foodCount * 0.01

    if capsuleCount > 0:
        evalScore += 5.0 / (minCapsuleDist + 1)

    for idx, g in enumerate(ghostStates):
        gPos = g.getPosition()
        gDist = manhattanDistance(pos, gPos)
        scared = g.scaredTimer
        if scared > 0:
            evalScore += 20.0 / (gDist + 1)
        else:
            if gDist == 0:
                return -float('inf')
            evalScore -= 10.0 / (gDist)

    evalScore += -2.0 * capsuleCount

    evalScore += -1.5 * foodCount

    return evalScore

# Abbreviation
better = betterEvaluationFunction
