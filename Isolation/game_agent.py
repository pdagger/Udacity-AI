"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import numpy as np


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    """
    This function calculates the inverse of the mean distance between the specified player 
    possible positions for the next move and the actual opponent's position + sqrt(5).
    Sqrt(5) is the distance between the opponent's actual position and its next position. 
    The mean of that mean distance is returned.
    """
    
    if len(game.get_legal_moves(player)) is 0:
        if game.is_loser(player):
            return float("-inf")
        if game.is_winner(player):
            return float("inf")
        else:
            return game.utility(player)

    else:
        moves_p_x, moves_p_y = zip(*game.get_legal_moves(player))

    if game.get_player_location(game.get_opponent(player)) is None:
        moves_o_x, moves_o_y = 0, 0
    else:
        moves_o_x, moves_o_y = game.get_player_location(game.get_opponent(player))

    inv_distance = 1 / ((np.sqrt((np.array(moves_p_x) - moves_o_x)**2 \
                + (np.array(moves_p_x) - moves_o_y)**2)) - np.sqrt(5) + 0.1)

    return (np.mean(inv_distance))

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    """
    This function calculates the mean distance between the specified player 
    possible positions in the next move to the center of the board. When  
    (game.height * 2) squares are in filled in the board, this function returns
    the same as custom_score_3.
    """

    if len(game.get_legal_moves(player)) is 0:
        if game.is_loser(player):
            return float("-inf")
        if game.is_winner(player):
            return float("inf")
        else:
            return game.utility(player)

    if len(game.get_blank_spaces()) > game.height * (game.width - 2):
        moves_p_x, moves_p_y = zip(*game.get_legal_moves(player))

        moves_o_x, moves_o_y = int(game.width/2) + 1 , int(game.height/2) + 1

        distance = 1/(np.sqrt((np.array(moves_p_x) - moves_o_x)**2 \
                        + (np.array(moves_p_x) - moves_o_y)**2) -1)

        return (np.mean(distance))

    else:
        #Number of legal moves avalaible to player
        moves_p = len(game.get_legal_moves(player))
        #Number of legal moves avalaible to opponent
        moves_o = len(game.get_legal_moves(game.get_opponent(player)))

        #Active player is cautious
        return float(4 * moves_p - moves_o)



def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    """
    This function returns the difference between the specified player number of
    possible positions multiplied by an arbitrary weight (4) and the possible
    positions of its opponent.
    """

    if len(game.get_legal_moves(player)) is 0:
        if game.is_loser(player):
            return float("-inf")
        if game.is_winner(player):
            return float("inf")
        else:
            return game.utility(player)
    #Number of legal moves avalaible to player
    moves_p = len(game.get_legal_moves(player))
    #Number of legal moves avalaible to opponent
    moves_o = len(game.get_legal_moves(game.get_opponent(player)))

    #Active player is cautious
    return float(4 * moves_p - moves_o)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        if len(game.get_legal_moves()) == 0:
            best_move = (-1, -1)
        else: best_move = random.choice(game.get_legal_moves())

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def terminal_test(self, game, current_depth, depth):
        """
        Returns True if the game is over for the active player or the desired
        search depth is reached and False otherwise.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        if (len(game.get_legal_moves()) is 0) | (current_depth == depth):
            return True
        else:
            return False

    def min_value(self, game, current_depth, depth):
        """
        Returns the value for a win(game.utility(self)) if the game is over, otherwise return
        the minimum value over all legal analyzed child nodes when the search depth is reached.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        elif self.terminal_test(game, current_depth, depth) is True:
            if current_depth == depth:
                return self.score(game, self)
            return game.utility(self)

        #Continue depth search at next level
        current_depth += 1
        v = float('inf')
        for move in game.get_legal_moves():
            v = min(v, self.max_value(game.forecast_move(move), current_depth, depth))

        return v

    def max_value(self, game, current_depth, depth):
        """
        Returns the value for a loss(game.utility(self)) if the game is over, otherwise return
        the maximum value over all legal analyzed child nodes when the search depth is reached.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        elif self.terminal_test(game, current_depth, depth) is True:
            if current_depth == depth:
                return self.score(game, self)
            return game.utility(self)

        #Continue depth search at next level
        current_depth += 1
        v = float('-inf')
        for move in game.get_legal_moves():
            v = max(v, self.min_value(game.forecast_move(move), current_depth, depth))

        return v

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        #Depth search starts at next level
        current_depth = 1

        if (len(game.get_legal_moves()) is 0):
            return (-1, -1)

        return max(game.get_legal_moves(), \
            key=lambda m: self.min_value(game.forecast_move(m), current_depth, depth))


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout.

        if len(game.get_legal_moves()) is 0:
            return (-1, -1)
        else:
            best_move = random.choice(game.get_legal_moves())

        # If player 1 first move, place knight at board's center.
        if (self == game._player_1) & (game.width * game.height == len(game.get_blank_spaces())):
            return (int(game.width/2) +1 , int(game.height/2) + 1)

        # If player 2 first move, place knight at board's center if space is empy, else next to it.
        if (self == game._player_2) & (game.width * game.height -1 == len(game.get_blank_spaces())):
            if (int(game.width/2) + 1, int(game.height/2) + 1) not in game.get_blank_spaces():
                return (int(game.width/2) + 1, int(game.height/2) + 1)
            else: return (int(game.width/2) + 2, int(game.height/2) + 2)
        
        #Iterative deepining search.  
        saved_move = best_move
        try:
            for depth in range(1, 10000):
                if self.time_left() < self.TIMER_THRESHOLD:
                    return saved_move
                best_move = self.alphabeta(game, depth)
                if best_move != (-1, -1):
                    saved_move = best_move
                
        except SearchTimeout:
            pass

        # Return the best move from the last completed search iteration
        return saved_move

    def terminal_test(self, game, current_depth, depth):
        """
        Returns True if the game is over for the active player or the desired
        search depth is reached and False otherwise.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        if (len(game.get_legal_moves()) is 0) | (current_depth == depth):
            return True
        else:
            return False

    def min_value(self, game, current_depth, depth, alpha, beta):
        """
        Returns the value for a win(game.utility(self)) if the game is over, otherwise return
        the minimum value over all legal analyzed child nodes when the search depth is reached.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        elif self.terminal_test(game, current_depth, depth) is True:
            if current_depth == depth:
                return self.score(game, self)
            return game.utility(self)

        #Continue depth search at next level
        current_depth += 1
        v = float('inf')
        for move in game.get_legal_moves():
            v = min(v, self.max_value(game.forecast_move(move), \
                                        current_depth, depth, alpha, beta))

            if v <= alpha:
                return v

            beta = min(beta, v)

        return v

    def max_value(self, game, current_depth, depth, alpha, beta):
        """
        Returns the value for a win(game.utility(self)) if the game is over, otherwise return
        the minimum value over all legal analyzed child nodes when the search depth is reached.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        elif self.terminal_test(game, current_depth, depth) is True:
            if current_depth == depth:
                return self.score(game, self)
            return game.utility(self)

        #Continue depth search at next level
        current_depth += 1
        v = float('-inf')
        for move in game.get_legal_moves():
            v = max(v, self.min_value(game.forecast_move(move), \
                                        current_depth, depth, alpha, beta))

            if v >= beta:
                return v
                
            alpha = max(alpha, v)

        return v

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        #Depth search starts at next level
        current_depth = 1

        if (len(game.get_legal_moves()) is 0):
            return (-1, -1)

        best_value = alpha
        best_move = random.choice(game.get_legal_moves())
        for move in game.get_legal_moves():
            v = self.min_value(game.forecast_move(move), current_depth, depth, alpha, beta)
            if v > alpha:
                alpha = v
                best_move = move

        return best_move





