"""
Travelling salesman problem
https://en.wikipedia.org/wiki/Travelling_salesman_problem

Problem Definition:
Given a list of cities and the distances between each pair of cities,
what is the shortest possible route that visits each city exactly once
and returns to the origin city?"

for this project i've chosen to consider a symetric TSP,
where the distance between two cities is the same in each opposite direction.
"""
import numpy as np
from game import Agent
import search
import random
import time


random.seed(100) # necessary to create repeatable results, 100 was chosen arbitrarily


def distancesGenerator(n,d):
    cities = np.zeros((n,n), dtype=int)
    for i in range(n):
        for j in range(1,n-i):
            distance = random.randrange(1,d,1)
            cities[i, i+j] = distance
            cities[i+j, i] = distance # distances bewteen cities is symetric, a->b = b->a
    return cities


class SearchAgent(Agent):
    
    def __init__(self, fn='depthFirstSearch', prob='tsProblem', heuristic='nullHeuristic'):
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        try:
            func.func_code.co_varnames
        except:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function in tsp.py or search.py.')
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            if fn:
                if fn!='depthFirstSearch' and fn!='dfs' and fn!='breatdhFirstSearch' and fn!='bfs' and fn!='uniformCostSearch' and fn!='ucs':
                    self.searchFunction = lambda x: func(x, heuristic=heur)
                else:
                    self.searchFunction = lambda x: func(x)
        else:
            if 'heuristic' not in func.func_code.co_varnames:
                print('[SearchAgent] using function ' + fn)
                self.searchFunction = func
            else:
                if heuristic in globals().keys():
                    heur = globals()[heuristic]
                elif heuristic in dir(search):
                    heur = getattr(search, heuristic)
                else:
                    raise AttributeError(heuristic + ' is not a function in tsp.py or search.py.')
                print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
                self.searchFunction = lambda x: func(x, heuristic=heur)
        
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in tsp.py.')
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)


    def registerInitialState(self, state):
        if self.searchFunction == None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.searchType(state[0],state[1])
        self.actions  = self.searchFunction(problem)
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        print('path found: ',self.actions)
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        pass

class tsProblem(search.SearchProblem):

    def __init__(self,cities,distances):
        self.citiesAmount = cities
        self.distances = distances
        self.start = {}
        self.goal = set(range(cities))
        self.heuristicInfo = {}

        self._expanded = 0


    def getStartState(self):
        return(tuple(self.start),0)

    def isGoalState(self, state):
        return set(state[0])==self.goal

    def getSuccessors(self, state):
        sucessors = []
        visited = list(state[0][:])
        lastState = state[1]
        unvisited = []
        #determine which intermediate cities have not been visited
        for i in range(1,self.citiesAmount):
            if (i not in visited):
                unvisited.append(i)
        # if all in between cities have been visited
        # the path needs to return to the startstate
        if unvisited == []:
            unvisited = [0]
        for action in unvisited:        
            nextState = visited.copy()
            nextState.append(action)
            cost = self.costFn(lastState,action)
            sucessors.append(((tuple(nextState),action), action, cost))
        self._expanded += 1
        return sucessors

    def costFn(self, current, next):
        return self.distances[current, next]

    def getCostOfActions(self, actions):
        if actions == None: return 999999
        cost = 0
        start = actions[-1]
        for i in actions:
            destination = i
            cost += self.costFn(start,destination)
            start = destination
        return cost

def startCityHeuristic(state, problem):
    """
    distance between the current city and the starting city

    This heuristic would prioritize visiting cities that are closer to the starting city,
    which may help to reduce the overall length of the final route.
    """
    current = state[1]
    return problem.costFn(current,0)
    
def closestCityHeuristic(state, problem):
    """
    distance between the current city and the furthest unvisited city
    
    This heuristic would prioritize visiting cities that are farther away from the other unvisited cities,
    which may help to reduce the amount of time spent traversing the same areas of the graph
    """
    current = state[1]
    visited = list(state[0][:])
    unvisited = []
    #determine which intermediate cities have not been visited
    for i in range(1,problem.citiesAmount):
        if (i not in visited):
            unvisited.append(i)
    distances = []
    for city in unvisited:
        distances.append(problem.costFn(current,city))
    if distances == []:
        return 0
    return min(distances)

def furthestCityHeuristic(state, problem):
    """
    distance between the current city and the furthest unvisited city
    
    This heuristic would prioritize visiting cities that are farther away from the other unvisited cities,
    which may help to reduce the amount of time spent traversing the same areas of the graph
    """
    current = state[1]
    visited = list(state[0][:])
    unvisited = []
    #determine which intermediate cities have not been visited
    for i in range(1,problem.citiesAmount):
        if (i not in visited):
            unvisited.append(i)
    distances = [0]
    for city in unvisited:
        distances.append(problem.costFn(current,city))
    return max(distances)

if __name__ == '__main__':
    n=14     # number of cities
    d=100    # max distance bewteen cities
    distances=distancesGenerator(n,d)
    #print(distances)
    startState = (n,distances)
    print()
    agent = SearchAgent(fn='ucs')
    agent.registerInitialState(startState)
    print()
    agent = SearchAgent(fn='aStarSearch', heuristic='startCityHeuristic')
    agent.registerInitialState(startState)
    print()
    agent = SearchAgent(fn='aStarSearch', heuristic='closestCityHeuristic')
    agent.registerInitialState(startState)
    print()
    agent = SearchAgent(fn='aStarSearch', heuristic='furthestCityHeuristic')
    agent.registerInitialState(startState)
    