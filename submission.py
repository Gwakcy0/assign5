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
  def __init__(self):
    self.lastPositions = []
    self.dc = None

  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()

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

######################################################################################
# Problem 1a: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)
    
      gameState.isWin():
        Returns True if it's a winning state
    
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
    """
    # BEGIN_YOUR_ANSWER
    # if end state, no move
    if gameState.isWin() or gameState.isLose():
      return Directions.STOP

    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions(self.index)
    scores = [self.getQ(gameState, action) for action in legalMoves]
    if self.index == 0:
      bestScore = max(scores)
    else:
      bestScore = min(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
    return legalMoves[chosenIndex]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the minimax Q-Value from the current gameState and given action
      using self.depth and self.evaluationFunction.
      Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves.
    """
    # BEGIN_YOUR_ANSWER
    def v_maxmin(state, player, depth):
      # manage end state utility and evaluation for the end of depth
      if state.isWin() or state.isLose():
        return state.getScore()
      if depth == self.depth:
        return self.evaluationFunction(state)

      # collect legal moves, and list possible states
      legalMoves = state.getLegalActions(player)
      if len(legalMoves) == 0:
        return self.evaluationFunction(state)

      # if pacman turn, choose maximum
      if player == 0:
        return max([v_maxmin(state.generateSuccessor(player, a), player + 1, depth) for a in legalMoves])
      elif player + 1 < state.getNumAgents():
        return min([v_maxmin(state.generateSuccessor(player, a), player + 1, depth) for a in legalMoves])
      else:
        return min([v_maxmin(state.generateSuccessor(player, a), 0, depth + 1) for a in legalMoves])

    numAgents = gameState.getNumAgents()
    nextPlayer = (self.index + 1) % numAgents
    successorState = gameState.generateSuccessor(self.index, action)
    return v_maxmin(successorState, nextPlayer, 0)
    # END_YOUR_ANSWER

######################################################################################
# Problem 2a: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    # if end state, no move
    if gameState.isWin() or gameState.isLose():
      return Directions.STOP

    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions(self.index)
    scores = [self.getQ(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
    return legalMoves[chosenIndex]
    # END_YOUR_ANSWER

  def getQ(self, gameState, action):
    """
      Returns the expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def v_imax(state, player, depth):
      # manage end state utility and evaluation for the end of depth
      if state.isWin() or state.isLose():
        return state.getScore()
      if depth == self.depth:
        return self.evaluationFunction(state)

      # collect legal moves, and list possible states
      legalMoves = state.getLegalActions(player)
      if len(legalMoves) == 0:
        return self.evaluationFunction(state)

      # if pacman turn, choose maximum
      if player == 0:
        return max([v_imax(state.generateSuccessor(player, a), player + 1, depth) for a in legalMoves])
      elif player + 1 < state.getNumAgents():
        return sum([v_imax(state.generateSuccessor(player, a), player + 1, depth) for a in legalMoves]) / len(legalMoves)
      else:
        return sum([v_imax(state.generateSuccessor(player, a), 0, depth + 1) for a in legalMoves]) / len(legalMoves)

    numAgents = gameState.getNumAgents()
    nextPlayer = (self.index + 1) % numAgents
    successorState = gameState.generateSuccessor(self.index, action)
    return v_imax(successorState, nextPlayer, 0)
    # END_YOUR_ANSWER

######################################################################################
# Problem 3a: implementing biased-expectimax

class BiasedExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your biased-expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the biased-expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing stop-biasedly from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    if gameState.isWin() or gameState.isLose():
      return Directions.STOP

    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions(self.index)
    scores = [self.getQ(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
    return legalMoves[chosenIndex]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the biased-expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def v_biased(state, player, depth):
      # manage end state utility and evaluation for the end of depth
      if state.isWin() or state.isLose():
        return state.getScore()
      if depth == self.depth:
        return self.evaluationFunction(state)

      # collect legal moves, and list possible states
      legalMoves = state.getLegalActions(player)
      if len(legalMoves) == 0:
        return self.evaluationFunction(state)

      # choose next move of a ghost
      if util.flipCoin(.5) and Directions.STOP in legalMoves:
        move = Directions.STOP
      else:
        move = random.choice(legalMoves)

      # if pacman turn, choose maximum
      if player == 0:
        return max([v_biased(state.generateSuccessor(player, a), player + 1, depth) for a in legalMoves])
      elif player + 1 < state.getNumAgents():
        v = sum([v_biased(state.generateSuccessor(player, a), player + 1, depth) for a in legalMoves]) / len(legalMoves)
        if Directions.STOP in legalMoves:
          return 0.5 * v_biased(state.generateSuccessor(player, Directions.STOP), player + 1, depth) + 0.5 * v
      else:
        v = sum([v_biased(state.generateSuccessor(player, a), 0, depth + 1) for a in legalMoves]) / len(legalMoves)
        if Directions.STOP in legalMoves:
          return 0.5 * v_biased(state.generateSuccessor(player, Directions.STOP), 0, depth + 1) + 0.5 * v
      return v

    numAgents = gameState.getNumAgents()
    nextPlayer = (self.index + 1) % numAgents
    successorState = gameState.generateSuccessor(self.index, action)
    return v_biased(successorState, nextPlayer, 0)
    # END_YOUR_ANSWER

######################################################################################
# Problem 4a: implementing expectiminimax

class ExpectiminimaxAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent (problem 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction

      The even-numbered ghost should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    if gameState.isWin() or gameState.isLose():
      return Directions.STOP

    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions(self.index)
    scores = [self.getQ(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
    return legalMoves[chosenIndex]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def v_imaxmin(state, player, depth):
      # manage end state utility and evaluation for the end of depth
      if state.isWin() or state.isLose():
        return state.getScore()
      if depth == self.depth:
        return self.evaluationFunction(state)

      # collect legal moves, and list possible states
      legalMoves = state.getLegalActions(player)
      if len(legalMoves) == 0:
        return self.evaluationFunction(state)

      # if pacman turn, choose maximum
      if player == 0:
        return max([v_imaxmin(state.generateSuccessor(player, a), player + 1, depth) for a in legalMoves])
      # if ghost turn, determine next player and depth
      player_next = (player + 1) % gameState.getNumAgents()
      if player_next == 0:
        depth += 1
      # odd: select minimum / even: random selection
      if player % 2 == 1:
        return min([v_imaxmin(state.generateSuccessor(player, a), player_next, depth) for a in legalMoves])
      else:
        return sum([v_imaxmin(state.generateSuccessor(player, a), player_next, depth) for a in legalMoves]) / len(
          legalMoves)

    numAgents = gameState.getNumAgents()
    nextPlayer = (self.index + 1) % numAgents
    successorState = gameState.generateSuccessor(self.index, action)
    return v_imaxmin(successorState, nextPlayer, 0)
    # END_YOUR_ANSWER

######################################################################################
# Problem 5a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent with alpha-beta pruning (problem 5)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_ANSWER
    if gameState.isWin() or gameState.isLose():
      return Directions.STOP

    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions(self.index)
    scores = [self.getQ(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
    return legalMoves[chosenIndex]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def v_alphabeta(state, player, depth, alpha, beta):
      # manage end state utility and evaluation for the end of depth
      if state.isWin() or state.isLose():
        return state.getScore()
      if depth == self.depth:
        return self.evaluationFunction(state)

      # collect legal moves, and list possible states
      legalMoves = state.getLegalActions(player)
      if len(legalMoves) == 0:
        return self.evaluationFunction(state)

      # determine next player and depth
      player_next = (player + 1) % state.getNumAgents()
      if player_next == 0:
        depth += 1

      # if pacman turn, choose maximum
      if player == 0:
        value = -float('inf')
        for a in legalMoves:
          value = max(value, v_alphabeta(state.generateSuccessor(player, a), player_next, depth, alpha, beta))
          alpha = max(value, alpha)
          if beta <= alpha:
            break
        return value
      elif player % 2 == 1:
        value = float('inf')
        for a in legalMoves:
          value = min(value, v_alphabeta(state.generateSuccessor(player, a), player_next, depth, alpha, beta))
          beta = min(value, beta)
          if beta <= alpha:
            break
        return value
      else:
        return sum([v_alphabeta(state.generateSuccessor(player, a), player_next, depth, alpha, beta) for a in legalMoves]) / len(
          legalMoves)

    numAgents = gameState.getNumAgents()
    nextPlayer = (self.index + 1) % numAgents
    successorState = gameState.generateSuccessor(self.index, action)
    return v_alphabeta(successorState, nextPlayer, 0, -float('inf'), +float('inf'))
    # END_YOUR_ANSWER

######################################################################################
# Problem 6a: creating a better evaluation function

class MyAgent(MultiAgentSearchAgent):
  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction
    """

    if gameState.isWin() or gameState.isLose():
      return Directions.STOP

    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions(self.index)
    scores = [self.getQ(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
    return legalMoves[chosenIndex]

  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """

    def v_alphabeta(state, player, depth, alpha, beta):
      # manage end state utility and evaluation for the end of depth
      if state.isWin() or state.isLose() or depth == self.depth:
        return self.evaluationFunction(state)

      # collect legal moves, and list possible states
      legalMoves = state.getLegalActions(player)
      if len(legalMoves) == 0:
        return self.evaluationFunction(state)

      # determine next player and depth
      player_next = (player + 1) % state.getNumAgents()
      if player_next == 0:
        depth += 1

      # if pacman turn, choose maximum
      if player == 0:
        value = -float('inf')
        for a in legalMoves:
          value = max(value, v_alphabeta(state.generateSuccessor(player, a), player_next, depth, alpha, beta))
          alpha = max(value, alpha)
          if beta <= alpha:
            break
        return value
      else:
        pacPos = state.getPacmanPosition()
        oppPos = state.getGhostPosition(player)
        if manhattanDistance(pacPos, oppPos) < 3:
          value = float('inf')
          for a in legalMoves:
            value = min(value, v_alphabeta(state.generateSuccessor(player, a), player_next, depth, alpha, beta))
            beta = min(value, beta)
            if beta <= alpha:
              break
          return value
        else:
          return sum([v_alphabeta(state.generateSuccessor(player, a), player_next, depth, alpha, beta) for a in legalMoves]) / len(
          legalMoves)

    numAgents = gameState.getNumAgents()
    nextPlayer = (self.index + 1) % numAgents
    successorState = gameState.generateSuccessor(self.index, action)
    return v_alphabeta(successorState, nextPlayer, 0, -float('inf'), +float('inf'))


def betterEvaluationFunction(currentGameState):
  """
  Your extreme, unstoppable evaluation function (problem 6).
  """
  # BEGIN_YOUR_ANSWER
  score = currentGameState.getScore()
  # pacPos = currentGameState.getPacmanPosition()
  # food_list = currentGameState.getFood()

  # give additional score for numFood
  # numFood = currentGameState.getNumFood()
  # score += 50 * numFood
  # # # give additional score for progress (percentage of foods eaten)
  # score += numFood * 20

  # ghostStates = currentGameState.getGhostStates()
  # scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
  # print(f'score: {score}, food_remaining: {numFood}')
  return score
  # END_YOUR_ANSWER

def choiceAgent():
  """
    Choose the pacman agent model you want for problem 6.
    You can choose among the agents above or design your own agent model.
    You should return the name of class of pacman agent.
    (e.g. 'MinimaxAgent', 'BiasedExpectimaxAgent', 'MyOwnAgent', ...)
  """
  # BEGIN_YOUR_ANSWER
  return 'MyAgent'
  # END_YOUR_ANSWER

# Abbreviation
better = betterEvaluationFunction
