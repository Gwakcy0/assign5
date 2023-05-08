class MinimaxAgent(Agent):
    """
    A minimax agent chooses an action at each choice point by using
    the minimax algorithm to determine the best move to make.
    """

    def getAction(self, gameState):
        """
        Returns the best action using the minimax algorithm
        """

        # Get the legal actions for the current state
        legalActions = gameState.getLegalActions()

        # Evaluate the minimax value for each legal action
        values = [self.minimax(gameState.generateSuccessor(0, action), 1, 1) for action in legalActions]

        # Find the action with the highest minimax value
        bestValue = max(values)
        bestIndices = [index for index in range(len(values)) if values[index] == bestValue]
        chosenIndex = random.choice(bestIndices)

        # Return the best action
        return legalActions[chosenIndex]

    def minimax(self, gameState, depth, player):
        """
        Returns the minimax value of the current game state
        """

        # Check if we have reached the maximum depth or the game is over
        if depth == self.maxDepth or gameState.isWin() or gameState.isLose():
            return self.getQ(gameState, None)

        # Get the legal actions for the current state
        legalActions = gameState.getLegalActions(player)

        # If there are no legal actions, return the utility value of the current state
        if len(legalActions) == 0:
            return self.getQ(gameState, None)

        # If it's the maximizing player's turn, return the maximum of the minimax values for the child states
        if player == 0:
            return max([self.minimax(gameState.generateSuccessor(player, action), depth, 1-player) for action in legalActions])

        # If it's the minimizing player's turn, return the minimum of the minimax values for the child states
        else:
            return min([self.minimax(gameState.generateSuccessor(player, action), depth+1, 1-player) for action in legalActions])
