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

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        presentGhostStates = currentGameState.getGhostStates()
        currentPos = currentGameState.getPacmanPosition()

        # diffferent weightages given to food, ghost, capsule
        scaredTime = newScaredTimes[0];
        food = 10.0
        ghost = 10.0
        capsule = 20.0
        
        score = successorGameState.getScore()
        
        #distance to ghost should be greater than 0 when ghost is not in scared state
        distanceToGhost = manhattanDistance(newPos, newGhostStates[0].getPosition())
        if  scaredTime == 0 and distanceToGhost > 0:
            score -= ghost/ distanceToGhost

        # distance to ghost should be less than 0 when ghost is in scared state 
        if scaredTime > 0 and distanceToGhost < 0:
           score += capsule/distanceToGhost

        #distance to food should be less when scared time is  0
        distancesToFood = [manhattanDistance(newPos, x) for x in newFood.asList()]
        if  scaredTime == 0 and  len(distancesToFood) :
            score += food/ min(distancesToFood)

        return score

def scoreEvaluationFunction(currentGameState):
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
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def isPlayer(self, state, agent):
         return agent % state.getNumAgents() == 0

    def isEndState(self,state,depth,agent):
        return state.getLegalActions(agent) == 0 or state.isLose() or state.isWin() or depth == self.depth

       

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        def minimax (state, depth, agent):
          if agent == state.getNumAgents():
               return minimax(state, depth+1,0) # moving to next depth

          if self.isEndState(state, depth, agent): 
               return self.evaluationFunction(state)

          nextState = (
                  minimax(state.generateSuccessor(agent, action), depth, agent+1) # next agent
                  for action in state.getLegalActions(agent))
        
          return (max if self.isPlayer(state,agent) else min)(nextState) 
        return max(gameState.getLegalActions(0), key=lambda x:minimax(gameState.generateSuccessor(0,x),0,1))
                  
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
   

 
    def getAction(self, gameState):
         """
         Returns the minimax action using self.depth and self.evaluationFunction
         """
         "*** YOUR CODE HERE ***"
         
         def miniMax(gameState, depth, agent, A, B):
            if agent >= gameState.getNumAgents():
                agent = 0
                depth += 1
            if (depth==self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)
            elif (agent == 0):
                return maximum(gameState, depth, agent, A, B)
            else:
                return minimum(gameState, depth, agent, A, B)
        
         def maximum(gameState, depth, agent, A, B):
            output = ["Yaooza", -float("inf")]
            listofActions = gameState.getLegalActions(agent)
            
            if not listofActions:
                return self.evaluationFunction(gameState)
                
            for action in listofActions:
                current = gameState.generateSuccessor(agent, action)
                cVal = miniMax(current, depth, agent+1, A, B)
                
                if type(cVal) is list:
                    check = cVal[1]
                else:
                    check = cVal
                    
                
                if check > output[1]:
                    output = [action, check]
                if check > B:
                    return [action, check]
                A = max(A, check)
            return output
            
         def minimum(gameState, depth, agent, A, B):
            output = ["Yaooza", float("inf")]
            gmoves = gameState.getLegalActions(agent)
           
            if not gmoves:
                return self.evaluationFunction(gameState)
                
            for action in gmoves:
                current = gameState.generateSuccessor(agent, action)
                cVal = miniMax(current, depth, agent+1, A, B)
                
                if type(cVal) is list:
                    check = cVal[1]
                else:
                    check = cVal
                    
                    
                if check < output[1]:
                    output = [action, check]
                if check < A:
                    return [action, check]
                B = min(B, check)
            return output
             
         outputList = miniMax(gameState, 0, 0, -float("inf"), float("inf"))
         return outputList[0]



         

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def getScore(agent,gameState,depth):
            if agent >= gameState.getNumAgents():
                agent = 0
                depth = depth + 1
            if (depth==self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)
            elif (agent == 0):
                return maximum(depth, agent,gameState)
            else:
                return expectimax(gameState,depth,agent)
        
        def maximum(depth, agent,gameState):
            output = ["Yaooza", -float("inf")]
            pmoves = gameState.getLegalActions(agent)
            
            if len (pmoves) == 0:
                return self.evaluationFunction(gameState)
                
            for action in pmoves:
                current = gameState.generateSuccessor(agent, action)
                value = getScore(agent+1,current,depth)
                if type(value) is list:
                    testVal = value[1]
                else:
                    testVal = value
                if testVal > output[1]:
                    output = [action, testVal]                    
            return output
            
        def expectimax(gameState, depth, agent):
            output = ["Yaooza", 0]
            gmoves = gameState.getLegalActions(agent)
            
            if len(gmoves) == 0:
                return self.evaluationFunction(gameState)
                
            probability = 1.0/len(gmoves)    
                
            for action in gmoves:
                current = gameState.generateSuccessor(agent, action)
                value = getScore( agent+1, current, depth)
                if type(value) is list:
                    val = value[1]
                else:
                    val = value
                output[0] = action
                output[1] += val * probability
            return output
             
        outputList = getScore(0,gameState,0)
        return outputList[0]  

def betterEvaluationFunction(currentGameState):
        """
        Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
        evaluation function (question 5).

        DESCRIPTION: <write something here so we know what you did>
        """
        "*** YOUR CODE HERE ***"
        
        food = [] 
        foodlist = currentGameState.getFood().asList() 
        
        
        current = list(currentGameState.getPacmanPosition()) 
        #storing distances of food in a list
        for f in foodlist:
            foodistance = manhattanDistance(f, current)
            food.append(-2*foodistance)
        
        if  len (food) == 0:
          food.append(0)
        # minimum distance from food subtracted from the last score
        return max(food) + currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction

