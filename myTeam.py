# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.captureAgents import CaptureAgent
from contest.game import Directions, Actions
from contest.util import nearestPoint, Queue

#import time

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        self.midWidth = game_state.data.layout.width / 2
        self.height = game_state.data.layout.height
        self.width = game_state.data.layout.width
        self.start_game_state = game_state
        self.closed_in_list = []
        self.position_food_eaten = None
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        #start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        #print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 0}

    def min_distance_to_home(self, game_state, pos):
        """
        Computes the minimum distance to home
        """
        # Home first column
        if self.red:
          x = int(self.midWidth - 1)
        else:
          x = int(self.midWidth + 1)
        positions = []
        # Compute valid positions in first column
        for y in range(self.height):
            if not isWall(game_state, (x,y)):
                positions.append((x,y))
        distance_to_home = 9999
        # Compute min distance
        for position in positions:
            distance =  self.get_maze_distance(position,pos)
            if distance < distance_to_home:
                distance_to_home = distance
        return distance_to_home

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that attacks with different cases.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        twoWays = two_ways(self.start_game_state , self)

        current_capsule_list = self.get_capsules(game_state)
        current_food_list = self.get_food(game_state).as_list()
        next_food_list = self.get_food(successor).as_list()
        next_capsule_list = self.get_capsules(successor)

        current_state =  game_state.get_agent_state(self.index)
        current_pos = current_state.get_position()
        next_state = successor.get_agent_state(self.index)
        next_pos = next_state.get_position()

        # Computes the current minimum distance to the ghosts
        current_min_distance_to_ghost = 0
        current_enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        current_ghosts = [a for a in current_enemies if (not a.is_pacman) and (a.scared_timer == 0) and (a.get_position() is not None)]
        if len(current_ghosts) > 0:
            current_min_distance_to_ghost = min([self.get_maze_distance(current_pos, a.get_position()) for a in current_ghosts])

        # Compute the current distance to pacman enemies we can see
        current_min_distance_to_pacman = 0
        current_invaders = [a for a in current_enemies if a.is_pacman and a.get_position() is not None]
        if len(current_invaders) > 0:
            current_min_distance_to_pacman = min([self.get_maze_distance(current_pos, a.get_position()) for a in current_invaders])

        # Computes the amount of food pacman is currently carrying
        current_carrying_food = current_state.num_carrying

        # Computes whether the current position is closed in
        if current_pos in self.closed_in_list:
            current_not_closed_in = False
        else:
            current_not_closed_in = twoWays.breadthFirstSearch(current_pos)
            if not current_not_closed_in:
                self.closed_in_list.append(current_pos)

        # Computes the current distance to home
        current_distance_to_home = self.min_distance_to_home(game_state, current_pos)

        # Computes whether we are currently on deffense
        if not current_state.is_pacman:
            current_is_defense = True
        else: current_is_defense = False

        # Computes the current distance to the nearest capsule
        if len(current_capsule_list) > 0:
            current_min_distance_to_capsule = min([self.get_maze_distance(current_pos, capsule) for capsule in current_capsule_list])

        # Computes the amount of food that is left for the successor
        features['successor_score'] = -len(next_food_list)  # self.getScore(successor)

        # Computes whether the successor is on defense or offense 
        if next_state.is_pacman:
            features['on_offense'] = 1
        else:
            features['on_defense'] = 1

        # Computes the distance to the nearest food for the successor
        if len(next_food_list) > 0:
            next_min_distance_to_food = min([self.get_maze_distance(next_pos, food) for food in next_food_list])
            features['distance_to_food'] = next_min_distance_to_food

        # Computes the distance to the nearest capsule for the successor
        if len(next_capsule_list) > 0:
            next_min_distance_to_capsule = min([self.get_maze_distance(next_pos, capsule) for capsule in next_capsule_list])
            features['distance_to_capsule'] = next_min_distance_to_capsule

        # Compute the distance to ghost enemies we can see for the successor
        next_min_distance_to_ghost = 0
        next_enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        next_ghosts = [a for a in next_enemies if (not a.is_pacman) and (a.scared_timer == 0) and (a.get_position() is not None)]
        if len(next_ghosts) > 0:
            next_min_distance_to_ghost = min([self.get_maze_distance(next_pos, a.get_position()) for a in next_ghosts])
            features['ghost_distance'] = next_min_distance_to_ghost

        # Compute the distance to pacman enemies we can see for the successor
        next_min_distance_to_pacman = 0
        next_invaders = [a for a in next_enemies if a.is_pacman and a.get_position() is not None]
        if len(next_invaders) > 0:
            next_min_distance_to_pacman = min([self.get_maze_distance(next_pos, a.get_position()) for a in next_invaders])
            features['invader_distance'] = next_min_distance_to_pacman

        # Compute the amount of invaders
        features['num_invaders'] = len(next_invaders)

        # Computes whether the action is STOP
        if action == Directions.STOP:
            features['stop'] = 1

        # Computes whether the action is REVERSE
        rev = Directions.REVERSE[next_state.configuration.direction]
        if action == rev:
            features['reverse'] = 1

        # Computes distance to scared ghosts we can see for the successor
        next_scared_ghosts = [a for a in next_enemies if not a.is_pacman and not a.scared_timer == 0 and a.get_position() is not None]
        if len(next_scared_ghosts) > 0:
            next_min_distance_to_scared_ghost = min([self.get_maze_distance(next_pos, a.get_position()) for a in next_scared_ghosts])
            features['scared_ghost_distance'] = next_min_distance_to_scared_ghost

        # Computes the amount of food pacman is carrying, for the successor
        next_carrying_food = next_state.num_carrying
        features['carrying_food'] = next_carrying_food

        # Compute the amount of food returned, for the successor
        next_returned_food = next_state.num_returned
        features['returned_food'] = next_returned_food

        # Computes whether the successor eats the capsule
        if current_capsule_list > next_capsule_list:
            features['capsule_eaten'] = 1

        # Computes whether the successor is going to be closed in
        if next_pos in self.closed_in_list:
            next_not_closed_in = False
        else:
            next_not_closed_in = twoWays.breadthFirstSearch(next_pos)
            if not next_not_closed_in:
                self.closed_in_list.append(next_pos)
        if  not next_not_closed_in:
            features['closed_in'] = 1

        # Computes the distance to home, for the successor
        next_distance_to_home = self.min_distance_to_home(successor, next_pos)
        features['distance_to_home'] = next_distance_to_home

        """Different cases"""

        # If being close to invader and not scared, eat it.
        if current_min_distance_to_pacman > 0 and current_min_distance_to_pacman < 7 and current_is_defense and current_state.scared_timer == 0:
            features['successor_score'] = 0
            features['on_offense'] = 0
            features['on_defense'] = features['on_defense'] * 10
            features['distance_to_food'] = 0
            features['distance_to_capsule'] = 0
            features['ghost_distance'] = 0
            features['stop']
            features['reverse']
            features['scared_ghost_distance'] = 0
            features['carrying_food']  = 0
            features['returned_food'] = 0
            features['capsule_eaten'] = 0
            features['closed_in'] = 0
            features['distance_to_home'] = 0
            features['invader_distance']
            features['num_invaders']

        # If being followed and there are capsules, go to capsule
        elif current_min_distance_to_ghost > 0 and current_min_distance_to_ghost < 6 and len(current_capsule_list) > 0:
            if current_min_distance_to_capsule < current_min_distance_to_ghost:
                features['ghost_distance'] = 0
            features['successor_score'] = 0
            features['on_offense'] = 0
            features['on_defense'] = 0
            features['distance_to_food'] = 0
            features['distance_to_capsule']
            features['ghost_distance']
            features['stop']
            features['reverse']
            features['scared_ghost_distance'] = 0
            features['carrying_food']  = 0
            features['returned_food']
            features['capsule_eaten']
            features['closed_in']
            features['distance_to_home'] = 0
            features['invader_distance'] = 0
            features['num_invaders'] = 0

        # If closed in, being followed and carrying food => go to non closed in
        elif not current_not_closed_in and current_min_distance_to_ghost < 9 and current_min_distance_to_ghost > 0 and current_carrying_food > 0 :
            features['successor_score'] = 0
            features['on_offense'] = 0
            features['on_defense'] = 0
            features['distance_to_food'] = 0
            features['distance_to_capsule'] = 0
            features['ghost_distance'] = 0
            features['stop'] = 0
            features['reverse'] = 0
            features['scared_ghost_distance'] = 0
            features['carrying_food'] = 0
            features['returned_food']
            features['capsule_eaten']
            features['closed_in']
            features['distance_to_home']
            features['invader_distance'] = 0
            features['num_invaders'] = 0

        # If being followed and carrying food or time is almost up or carrying more than 15 => go to home
        elif (current_min_distance_to_ghost > 0 and current_min_distance_to_ghost < 6  and current_carrying_food > 0) or (len(current_food_list) <= 2) or game_state.data.timeleft < self.min_distance_to_home(game_state, current_pos) +  60 or (current_distance_to_home < 5 and current_carrying_food > 0) or current_carrying_food > 15 :
            features['successor_score'] = 0
            features['on_offense'] = 0
            features['on_defense']
            features['distance_to_food'] = 0
            features['distance_to_capsule'] = 0
            features['ghost_distance'] = next_min_distance_to_ghost * 3
            features['stop']
            features['reverse'] = 0
            features['scared_ghost_distance'] = 0
            features['carrying_food'] = 0
            features['returned_food']
            features['capsule_eaten']
            features['closed_in']
            features['distance_to_home'] = next_distance_to_home / 2
            features['invader_distance'] = 0
            features['num_invaders'] = 0

        # Else eat food
        else:
            features['successor_score']
            features['on_offense'] = 0
            features['on_defense'] = 0
            features['distance_to_food']
            features['distance_to_capsule']
            features['ghost_distance'] = 0
            features['stop']
            features['reverse'] = 0
            features['scared_ghost_distance']
            features['carrying_food'] = 0
            features['returned_food']
            features['capsule_eaten']
            features['closed_in'] = 0
            features['distance_to_home'] = 0
            features['invader_distance'] = 0
            features['num_invaders'] = 0

        return features

    def get_weights(self, game_state, action):

        return {'successor_score': 100, 'distance_to_food': -2, 'distance_to_capsule': -1, 'reverse': -2, 'ghost_distance': 5, 'carrying_food': -2 , 'scared_ghost_distance': -1, 'stop': - 100, 'on_offense': 10, 'returned_food': 10, 'distance_to_home': - 10, 'closed_in': -100, 'capsule_eaten': 9999, 'on_defense': 10,'num_invaders': -1000, 'invader_distance': -10 }

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that attacks with different cases.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        previous_game_state = self.get_previous_observation()
        successor = self.get_successor(game_state, action)
        height_maze = self.height
        width_maze = self.width

        current_state =  game_state.get_agent_state(self.index)
        current_pos = current_state.get_position()
        next_state = successor.get_agent_state(self.index)
        next_pos = next_state.get_position()
        current_food_list = self.get_food_you_are_defending(game_state).as_list()

        # Compute the current distance to pacman invaders we can see
        current_enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        current_total_invaders = [a for a in current_enemies if a.is_pacman]

        # Compute position of last eaten food
        if not previous_game_state is None:
            previous_food = self.get_food_you_are_defending(previous_game_state).as_list()
            if len(previous_food) > len(current_food_list):
                for food in previous_food:
                    if not food in current_food_list:
                        self.position_food_eaten = food

        # Compute current distance to border
        current_distance_border = self.min_distance_to_home(game_state, current_pos)

        # Computes whether the successor is on defense (1) or offense (0)
        features['on_defense'] = 1
        features['do_not_attack'] = 0
        if next_state.is_pacman: 
            features['on_defense'] = 0
            features['do_not_attack'] = 1
            

        # Computes distance to pacman invaders we can see for the successor
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(next_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        # Computes whether the action is STOP
        if action == Directions.STOP: features['stop'] = 1
        # Computes whether the action is REVERSE
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        # Compute distance to last eaten food for the successor
        if not self.position_food_eaten is None:
            features['distance_to_last_food_eaten'] = self.get_maze_distance(next_pos, self.position_food_eaten)

        # Computes the distance to the closest border: top or bottom, and the x middle and y middle
        top_border = height_maze
        bottom_border = 0
        distance_to_closest_border = 0
        distance_to_top_border = abs(current_pos[1] - top_border)
        distance_to_bottom_border = abs(current_pos[1] - bottom_border)
        # Check if current position is closer to the top or bottom border
        if distance_to_top_border < distance_to_bottom_border:
            distance_to_closest_border = distance_to_top_border
        else:
            distance_to_closest_border = distance_to_bottom_border
        features['distance_to_closest_border'] = distance_to_closest_border
        # Compute successor distance to x middle
        x_middle = width_maze / 2
        distance_to_x_middle = abs(next_pos[0] - x_middle)
        features['distance_to_x_middle'] = distance_to_x_middle
        # Compute successor distance to y middle
        y_middle = height_maze / 2
        distance_to_y_middle = abs(current_pos[1] - y_middle)
        features['distance_to_y_middle'] = distance_to_y_middle

        # Compute successor distance to border
        features['distance_to_border'] = self.min_distance_to_home(game_state, next_pos)

        # Compute current distance to x middle
        current_distance_to_x_middle = abs(current_pos[0] - x_middle)

        """Different cases"""

        # If the agent is scared, go to the border.
        if current_state.scared_timer > 0:
            features['on_defense']
            features['do_not_attack']
            features['stop']
            features['distance_to_last_food_eaten'] = 0
            features['distance_to_closest_border'] = 0
            features['distance_to_capsule'] = 0
            features['distance_to_x_middle']
            features['distance_to_y_middle']
            features['reverse']
            features['invader_distance'] = features['invader_distance'] * -1
            features['num_invaders'] = 0
            if current_distance_border < 6:
                features['distance_to_border'] = 0

        #If there is an invader, eat it.
        elif len(current_total_invaders) > 0:
            features['on_defense']
            features['do_not_attack']
            features['stop']
            features['distance_to_last_food_eaten']
            features['distance_to_closest_border'] = 0
            features['distance_to_capsule'] = 0
            features['distance_to_x_middle'] = 0
            features['distance_to_y_middle'] = 0
            features['reverse'] = 0
            features['invader_distance']
            features['num_invaders']
            features['distance_to_border'] = 0

        #If no invader or scared: patrol
        else:
             # If agent is in the middle, patrol vertically
            if current_distance_to_x_middle <= 3:
                features['distance_to_y_middle']
                features['distance_to_x_middle'] = 0
                features['distance_to_closest_border']
            else:
                features['distance_to_x_middle']
                features['distance_to_y_middle'] = 0
                features['distance_to_closest_border'] = 0
            features['stop']
            features['distance_to_last_food_eaten'] = 0
            features['reverse']
            features['invader_distance'] = 0
            features['on_defense']
            features['do_not_attack']
            features['num_invaders'] = 0
            if current_distance_border < 6:
                features['distance_to_border'] = 0

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2, 'distance_to_y_middle': -0.7, 'distance_to_x_middle': -1.2, 'distance_to_closest_border': -0.7, 'distance_to_last_food_eaten': -8 ,  'distance_to_border': -1, 'do_not_attack': -10000 }

class two_ways:
    """
    A class to determine whether a positions has at least two different ways to home.
    """
    def __init__(self, game_state, agent):

        self.agent = agent
        self.game_state = game_state
        self.height = game_state.data.layout.height
        self.width = game_state.data.layout.width
        self.midWidth = game_state.data.layout.width / 2
        self.red = agent.red

    def isGoalState(self, state):
        """
        Checks whether the given state is the goal state, a position in the first column in the home side.
        """
        if self.red:
            i = int(self.midWidth - 1)
        else:
            i = int(self.midWidth + 1)
        return state[0] == i

    def getSuccessors(self, state):
        """
        Determines the valid successors of the given position.
        """
        my_pos = state
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = my_pos
            dx, dy = Actions.direction_to_vector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            next_pos = (nextx, nexty)
            if not isWall(self.game_state, (next_pos)):
                successors.append(next_pos)
        return successors

    def breadthFirstSearch(self, pos):
        """
        Returns a boolean, which is True when the given position has two different ways to home, if not it returns False.
        """
        fringe = Queue()
        fringe.push(pos)
        visited = []
        states = Queue()
        states.push([])
        count = 0
        first_way = []

        while not fringe.isEmpty():
            node = fringe.pop()
            path = states.pop()
            #when the goal state is reached for the first time, the path is saved,
            #if it is reached again, we check if the paths have positions in common.
            if self.isGoalState(node):
                if count == 0:
                    first_way = path
                count += 1
            if count > 1:
                one_way = False
                for elem in first_way:
                    if elem in path:
                        one_way = True
                if not one_way:
                    return True
            if node in visited:
                continue
            else:
                visited.append(node)
            child_nodes = self.getSuccessors(node)
            for child_node in child_nodes:
                pathToNode = path + [child_node]
                if child_node not in visited:
                    fringe.push(child_node)
                    states.push(pathToNode)
        else:
            return False

def isWall(game_state,pos):
    """
    Determines whether the position is a wall
    """
    grid = game_state.data.layout.walls
    return grid[pos[0]][pos[1]]
