import numpy as np
import snake_game
import time


class SnakeTDAgent:

    def __init__(self, game_size=5, method='Q-Learning', space="Coordinates", epsilon=1, epsilon_decay=0.99, epsilon_min=0.1, gamma=1, alpha=0.8, episodes=10000, growing=False):
        self.statespace = space

        if self.statespace == "Coordinates":
            self.q_state_action_values = [
                [[[np.array([0, 0, 0, 0]) for i in range(game_size + 2)] for j in range(game_size + 2)] for k in
                 range(game_size + 2)] for l in range(game_size + 2)]
            # [head pos y][head pos x][fruit pos y][fruit pos x]

            # set all crashes to 0
            for i in range(0, game_size + 2):
                for j in range(0, game_size + 2):
                    for k in range(0, game_size + 2):
                        for l in range(0, game_size + 2):
                            if (i % (game_size + 1)) * (j % (game_size + 1)) == 0:
                                self.q_state_action_values[i][j][k][l] = np.array([0., 0., 0., 0.])

        elif self.statespace == "Distance and Collision":
            # initialize Q
            self.q_state_action_values = [
                [[[[[[np.array([0., 0., 0., 0.]) for dist_y in range(2 * game_size)] for dist_x in
                     range(2 * game_size)] for n in [0, 1]] for e in [0, 1]] for s in [0, 1]] for
                 w in [0, 1]] for current in [0, 1]]
            # [dist y][dist x][obstacle north][east][south][west][current position]

            # set all crashes to 0
            for dist_y in range(2 * game_size):
                for dist_x in range(2 * game_size):
                    for n in [0, 1]:
                        for e in [0, 1]:
                            for s in [0, 1]:
                                for w in [0, 1]:
                                    for current in [0, 1]:
                                        if current == 0:
                                            self.q_state_action_values[dist_y][dist_x][n][e][s][w][current] = np.array(
                                                [0., 0., 0., 0.])

        elif self.statespace == "Indicators":
            # initialize Q
            self.q_state_action_values = [
                [[[[[[np.array([0., 0., 0., 0.]) for y in ['higher', 'same', 'lower']] for x in
                     ['left', 'same', 'right']] for n in [0, 1]] for e in [0, 1]] for s in [0, 1]] for
                 w in [0, 1]] for current in [0, 1]]
            # [dist y][dist x][obstacle north][east][south][west][current position]

            # set all crashes to 0
            for y in [0, 1, 2]:
                for x in [0, 1, 2]:
                    for n in [0, 1]:
                        for e in [0, 1]:
                            for s in [0, 1]:
                                for w in [0, 1]:
                                    for current in [0, 1]:
                                        if current == 1:    # aka there is an obstacle on the head's position = death
                                            self.q_state_action_values[y][x][n][e][s][w][current] = np.array(
                                                [0., 0., 0., 0.])

        self.learningmethod = method
        self.grow = growing
        self.episodes = episodes
        self.game_size = game_size

        self.epsilon, self.epsilon_decay, self.epsilon_min, self.gamma, self.alpha = epsilon, epsilon_decay, epsilon_min, gamma, alpha

    def state_as_positions(self, game):
        return game.snake_position_head[0], game.snake_position_head[1], game.fruit[0], game.fruit[1]

    def state_as_vector(self, game):
        collision = [0, 0, 0, 0]
        if game.snake_position_head[0] == 1 or\
                (lambda x: [x[0]-1, x[1]])(game.snake_position_head) in game.snake_position_body: #north
            collision[0] = 1
        if game.snake_position_head[1] == 1 or\
                (lambda x: [x[0]+1, x[1]])(game.snake_position_head) in game.snake_position_body: #east
            collision[1] = 1
        if game.snake_position_head[0] == game.size or\
                (lambda x: [x[1]-1, x[1]])(game.snake_position_head) in game.snake_position_body: #south
            collision[2] = 1
        if game.snake_position_head[1] == game.size or\
                (lambda x: [x[1] + 1, x[1]])(game.snake_position_head) in game.snake_position_body: #west
            collision[3] = 1

        return (game.snake_position_head[0] - game.fruit[0], game.snake_position_head[1] - game.fruit[1]), collision

    def epsilon_action_choice(self, h_0, h_1, f_0, f_1):
        if self.epsilon > np.random.uniform(0, 1):
            action = max(
                [(self.q_state_action_values[h_0][h_1][f_0][f_1][direction], direction) for direction in
                 range(4)])[1]
        else:
            action = np.random.randint(0, 4)
        return action

    def td_update(self, game, old_score, state, action, next_state, next_action):
        h_0, h_1, f_0, f_1 = state
        h_0_new, h_1_new, f_0_new, f_1_new = next_state
        try:
            reward = game.score - old_score
            target = reward + self.gamma * self.q_state_action_values[h_0_new][h_1_new][f_0_new][f_1_new][
                next_action]
            update = self.alpha * (target - self.q_state_action_values[h_0][h_1][f_0][f_1][action])
            self.q_state_action_values[h_0][h_1][f_0][f_1][action] += update
        finally:
            print(h_0, h_1, f_0, f_1, action)
            print(h_0_new, h_1_new, f_0_new, f_1_new, next_action)

    def sarsa_coord(self):
        for i in range(self.episodes):
            print(i)
            game = snake_game.Game(n=self.game_size, growing=self.grow)

            state = self.state_as_positions(game)
            h_0_new, h_1_new, f_0_new, f_1_new = state

            next_action = self.epsilon_action_choice(h_0, h_1, f_0, f_1)

            while game.alive:
                # choosing action
                h_0, h_1, f_0, f_1 = h_0_new, h_1_new, f_0_new, f_1_new

                old_score = game.score
                action = next_action
                game.direction = action

                game.update_state()

                next_state = self.state_as_positions(game)
                h_0_new, h_1_new, f_0_new, f_1_new = next_state
                next_action = self.epsilon_action_choice(h_0, h_1, f_0, f_1)

                # Updating
                self.td_update(game, old_score, state, action, next_state, next_action)

        epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return

    def q_learn_coord(self):
        for i in range(self.episodes):
            print(i)
            game = snake_game.Game(n=self.game_size, growing=self.grow)

            state = self.state_as_positions(game)
            h_0, h_1, f_0, f_1 = state

            while game.alive:
                # choosing action
                action = self.epsilon_action_choice(h_0, h_1, f_0, f_1)
                game.direction = action
                old_score = game.score

                game.update_state()

                next_state = self.state_as_positions(game)
                h_0_new, h_1_new, f_0_new, f_1_new = next_state

                next_action = max(
                    [(self.q_state_action_values[h_0_new][h_1_new][f_0_new][f_1_new][direction], direction) for
                     direction in range(4)])[1]

                # Updating
                self.td_update(game, old_score, state, action, next_state, next_action)
                # S <- S'
                h_0, h_1, f_0, f_1 = h_0_new, h_1_new, f_0_new, f_1_new

            epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return

    def present_game(self, policy="greedy"):
        game = snake_game.Game(n=self.game_size, growing=self.grow)
        game.print_state()
        while game.alive:
            time.sleep(0.75)
            # choosing action
            h_0 = game.snake_position_head[0]
            h_1 = game.snake_position_head[1]
            f_0 = game.fruit[0]
            f_1 = game.fruit[1]
            if policy == "greedy":
                action = \
                max([(self.q_state_action_values[h_0][h_1][f_0][f_1][direction], direction) for direction in range(4)])[
                    1]

            old_score = game.score

            game.direction = action
            game.update_state()
            game.print_state()
        return



methods = {"Q-Learning", "Sarsa"}
space = {"Coordinates", "Distance and Collision"}

test_agent = SnakeTDAgent(game_size=5, episodes=100000, growing=True)
#print(test_agent.q_state_action_values)
test_agent.learning(epsilon=0.8, gamma=1, alpha=0.8)
test_agent.present_game()
#print(test_agent.q_state_action_values)
