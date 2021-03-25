import numpy as np
import snake_game
import time


class SnakeTDAgent:

    def __init__(self, game_size=5, method='Q-Learning', statespace="Coordinates", epsilon=1, epsilon_decay=0.99,
                 epsilon_min=0.1, gamma=1, alpha=0.8, episodes=10000, growing=False):
        self.statespace = statespace

        self.learningmethod = method
        self.grow = growing
        self.episodes = episodes
        self.game_size = game_size

        self.game = snake_game.Game(n=self.game_size, growing=self.grow)

        self.epsilon, self.epsilon_decay, self.epsilon_min = epsilon, epsilon_decay, epsilon_min
        self.gamma, self.alpha = gamma, alpha

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

            self.state_function = self.game.state_as_positions
            self.epsilon_action_choice = self.epsilon_action_choice_coord
            self.td_update = self.td_update_coord

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

            self.state_function = self.game.state_as_distance_vector
            self.td_update = self.td_update_dist

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

            self.state_function = self.game.state_as_indicators
            self.td_update = self.td_update_indicators

        else:
            print('!!! invalid statespace !!!')
            return

    def epsilon_action_choice_coord(self, state):
        h_0, h_1, f_0, f_1 = state
        state_value = self.q_state_action_values[h_0][h_1][f_0][f_1]

        if self.working_epsilon > np.random.uniform(0, 1):
            action = max([(state_value[direction], direction) for direction in range(4)])[1]
        else:
            action = np.random.randint(0, 4)
        return action

    def td_update_coord(self, old_score, state, action, next_state, next_action):
        h_0, h_1, f_0, f_1 = state
        h_0_new, h_1_new, f_0_new, f_1_new = next_state
        try:
            reward = self.game.score - old_score
            target = reward + self.gamma * self.q_state_action_values[h_0_new][h_1_new][f_0_new][f_1_new][
                next_action].copy()
            update = self.alpha * (target - self.q_state_action_values[h_0][h_1][f_0][f_1][action].copy())
            self.q_state_action_values[h_0][h_1][f_0][f_1][action] += update
        finally:
            print(state, action)
            print(next_state, next_action)

    def epsilon_action_choice_indicators(self, state):
        [dist_y, dist_x], [n, e, s, w, current] = state
        state_value = self.q_state_action_values[dist_y][dist_x][n][e][s][w][current]

        if self.working_epsilon > np.random.uniform(0, 1):
            action = max([(state_value[direction], direction) for direction in range(4)])[1]
        else:
            action = np.random.randint(0, 4)

        return action

    def td_update_indicators(self, old_score, state, action, next_state, next_action):
        [dist_y, dist_x], [n, e, s, w, current] = state
        [dist_y_new, dist_x_new], [n_new, e_new, s_new, w_new, current_new] = next_state
        try:
            reward = self.game.score - old_score
            target = reward + self.gamma * \
                     self.q_state_action_values[dist_y_new][dist_x_new][n_new][e_new][s_new][w_new][current_new][
                         next_action].copy()
            update = self.alpha * (target - self.q_state_action_values[dist_y][dist_x][n][e][s][w][current].copy())
            self.q_state_action_values[dist_y][dist_x][n][e][s][w][current][action] += update
        finally:
            print(state, action)
            print(next_state, next_action)
        return

    def epsilon_action_choice_dist(self, state):
        [indicator_y, indicator_x], [n, e, s, w, current] = state
        state_value = self.q_state_action_values[indicator_y][indicator_x][n][e][s][w][current]

        if self.working_epsilon > np.random.uniform(0, 1):
            action = max([(state_value[direction], direction) for direction in range(4)])[1]
        else:
            action = np.random.randint(0, 4)

        return action

    def sarsa(self):
        self.working_epsilon = self.epsilon
        for i in range(self.episodes):
            self.game.reset()
            print(i)

            state = self.state_function()
            h_0_new, h_1_new, f_0_new, f_1_new = state

            action = self.epsilon_action_choice(state)

            while self.game.alive:
                # choosing action

                old_score = self.game.score
                self.game.direction = action

                self.game.update_state()

                next_state = self.state_function()
                next_action = self.epsilon_action_choice(next_state)

                # Updating
                self.td_update(old_score, state, action, next_state, next_action)

                state, action = next_state, next_action
            self.working_epsilon = max(self.epsilon_min, self.working_epsilon * self.epsilon_decay)

        return

    def q_learn(self):
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
space = {"Coordinates", "Distance and Collision", "Indicators"}

test_agent = SnakeTDAgent(game_size=5, episodes=1, growing=True, epsilon=0.8, gamma=1, alpha=0.8)
#print(test_agent.q_state_action_values)
test_agent.sarsa()
test_agent.present_game()
#print(test_agent.q_state_action_values)
