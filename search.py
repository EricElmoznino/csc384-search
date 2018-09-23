# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import Stack, Queue, PriorityQueue

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

class Path:

    def __init__(self, start_state):
        self.path = []
        self.total_cost = 0.0
        self.path = [[start_state, None]]

    def __copy__(self):
        new = Path(None)
        new.path = [p for p in self.path]
        new.total_cost = self.total_cost
        return new

    def appended(self, state, action, cost):
        new = self.__copy__()
        new.path += [[state, action]]
        new.total_cost += cost
        return new

    def end_state(self):
        return self.path[-1][0]

    def actions(self):
        return [a for _, a in self.path[1:]]

    def cost(self):
        return self.total_cost

    def end_forms_cycle(self):
        end = self.end_state()
        for i in range(len(self.path) - 2, -1, -1):
            if end == self.path[i][0]:
                return True
        return False

class Frontier:

    def __init__(self, problem, type='Stack', heuristic=None, cycle_checking=False, path_checking=False):
        if type == 'Stack':
            self.frontier = Stack()
        elif type == 'Queue':
            self.frontier = Queue()
        elif type == 'PriorityQueue':
            self.frontier = PriorityQueue()
        else:
            util.raiseNotDefined()

        self.problem = problem
        self.heuristic = heuristic

        self.cycle_checking = cycle_checking
        self.path_checking = path_checking
        if cycle_checking:
            self.visited = {}

    def insert(self, path):
        if self.cycle_checking:
            if path.end_state() in self.visited:
                if self.visited[path.end_state()] > self.cost(path):
                    self.visited[path.end_state()] = self.cost(path)
                else:
                    return
            else:
                self.visited[path.end_state()] = self.cost(path)
        elif self.path_checking:
            if path.end_forms_cycle():
                return

        if isinstance(self.frontier, PriorityQueue):
            self.frontier.push(path, self.cost(path))
        else:
            self.frontier.push(path)

    def extract(self):
        expanded = self.frontier.pop()
        return expanded

    def empty(self):
        return self.frontier.isEmpty()

    def cost(self, path):
        cost = path.cost()
        if self.heuristic is not None:
            cost += self.heuristic(path.end_state(), self.problem)
        return cost

    def better_path_visited(self, path):
        if not self.cycle_checking:
            return False
        if path.end_state() not in self.visited:
            return False
        elif self.cost(path) <= self.visited[path.end_state()]:
            return False
        return True

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def search(problem, frontier):
    frontier.insert(Path(start_state=problem.getStartState()))
    while not frontier.empty():
        n = frontier.extract()
        state = n.end_state()
        if not frontier.better_path_visited(n):
            if problem.isGoalState(state):
                return n.actions()
            for succ in problem.getSuccessors(state):
                frontier.insert(n.appended(*succ))
    return []

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    return search(problem, Frontier(problem=problem, type='Stack', path_checking=True))

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    return search(problem, Frontier(problem=problem, type='Queue', cycle_checking=True))

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    return search(problem, Frontier(problem=problem, type='PriorityQueue', cycle_checking=True))

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    return search(problem, Frontier(problem=problem, type='PriorityQueue', cycle_checking=True, heuristic=heuristic))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
