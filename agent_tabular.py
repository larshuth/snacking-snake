import numpy as np
import snake_game
import time
import matplotlib.pyplot as plt
import seaborn as sns


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
        self.working_epsilon = max(self.epsilon_min, self.epsilon)
        self.gamma, self.alpha = gamma, alpha

        if self.statespace == "Coordinates":
            self.q_state_action_values = [
                [[[np.array([0, 0, 0, 0]) for i in range(self.game_size + 2)] for j in range(self.game_size + 2)] for k
                 in range(self.game_size + 2)] for m in range(self.game_size + 2)]
            # [head pos y][head pos x][fruit pos y][fruit pos x]

            # set all crashes to 0
            for i in range(0, game_size + 2):
                for j in range(0, game_size + 2):
                    for k in range(0, game_size + 2):
                        for m in range(0, game_size + 2):
                            if (i % (game_size + 1)) * (j % (game_size + 1)) == 0:
                                self.q_state_action_values[i][j][k][m] = np.array([0., 0., 0., 0.])

            self.state_function = self.game.state_as_positions
            self.action_choice = self.epsilon_action_choice_coord
            self.learning_update_function = self.td_update_coord

        elif self.statespace == "Distance and Collision":
            # initialize Q
            self.q_state_action_values = [[[[[[[np.array([0., 0., 0., 0.]) for current in [0, 1]] for w in [0, 1]] for s
                                              in [0, 1]] for e in [0, 1]] for n in [0, 1]] for dist_x in
                                           range(2 * self.game_size + 2)] for dist_y in range(2 * self.game_size + 2)]
            # [dist y][dist x][obstacle north][east][south][west][current position]

            print(self.q_state_action_values[0])
            print(self.q_state_action_values[0][1])
            print(self.q_state_action_values[0][2])

            # set all crashes to 0
            for dist_y in range(2 * self.game_size):
                for dist_x in range(2 * self.game_size):
                    for n in [0, 1]:
                        for e in [0, 1]:
                            for s in [0, 1]:
                                for w in [0, 1]:
                                    for current in [0, 1]:
                                        if current == 0:
                                            self.q_state_action_values[dist_y][dist_x][n][e][s][w][current] = np.array(
                                                [0., 0., 0., 0.])

            self.state_function = self.game.state_as_distance_vector
            self.learning_update_function = self.td_update_dist_and_indicators
            self.action_choice = self.epsilon_action_choice_dist_and_indicators

        elif self.statespace == "Indicators":
            # initialize Q
            self.q_state_action_values = [[[[[[[np.array([0., 0., 0., 0.]) for current in [0, 1]] for w in [0, 1]] for s
                                              in [0, 1]] for e in [0, 1]] for n in [0, 1]] for x in
                                           ['left', 'same', 'right']] for y in ['higher', 'same', 'lower']]
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
            self.learning_update_function = self.td_update_dist_and_indicators
            self.action_choice = self.epsilon_action_choice_dist_and_indicators

        else:
            print('!!! invalid statespace !!!')
            return

    def max_action_choice(self, state):
        if self.statespace == 'Coordinates':
            h_0, h_1, f_0, f_1 = state
            state_value = self.q_state_action_values[h_0][h_1][f_0][f_1]
        elif self.statespace == 'Distance and Collision':
            [dist_y, dist_x], [n, e, s, w, current] = state
            state_value = self.q_state_action_values[dist_y][dist_x][n][e][s][w][current]
        elif self.statespace == 'Indicators':
            [indicator_y, indicator_x], [n, e, s, w, current] = state
            state_value = self.q_state_action_values[indicator_y][indicator_x][n][e][s][w][current]
        else:
            print("Not valid")
            return 4

        action = max([(state_value[direction], direction) for direction in range(4)])[1]
        return action

    def epsilon_action_choice(self, state_value):
        if self.working_epsilon < np.random.uniform(0, 1):
            action = max([(state_value[direction], direction) for direction in range(4)])[1]
        else:
            action = np.random.randint(0, 4)
        return action

    def td_update(self, old_score, state_action_value, next_state_action_value):
        reward = self.game.score - old_score
        target = reward + self.gamma * next_state_action_value
        update = self.alpha * (target - state_action_value)
        return update

    def epsilon_action_choice_coord(self, state):
        h_0, h_1, f_0, f_1 = state
        state_value = self.q_state_action_values[h_0][h_1][f_0][f_1]

        action = self.epsilon_action_choice(state_value)
        return action

    def td_update_coord(self, old_score, state, action, next_state, next_action):
        h_0, h_1, f_0, f_1 = state
        h_0_new, h_1_new, f_0_new, f_1_new = next_state

        state_action_value = self.q_state_action_values[h_0][h_1][f_0][f_1][action].copy()
        next_state_action_value = self.q_state_action_values[h_0_new][h_1_new][f_0_new][f_1_new][next_action].copy()

        update = self.td_update(old_score, state_action_value, next_state_action_value)

        self.q_state_action_values[h_0][h_1][f_0][f_1][action] += update

    def epsilon_action_choice_dist_and_indicators(self, state):
        [dist_y, dist_x], [n, e, s, w, current] = state
        state_value = self.q_state_action_values[dist_y][dist_x][n][e][s][w][current]

        action = self.epsilon_action_choice(state_value)
        return action

    def td_update_dist_and_indicators(self, old_score, state, action, next_state, next_action):
        [dist_y, dist_x], [n, e, s, w, current] = state
        [dist_y_new, dist_x_new], [n_new, e_new, s_new, w_new, current_new] = next_state

        next_state_action_value = \
            self.q_state_action_values[dist_y_new][dist_x_new][n_new][e_new][s_new][w_new][current_new][
                next_action].copy()
        state_action_value = self.q_state_action_values[dist_y][dist_x][n][e][s][w][current][action].copy()

        update = self.td_update(old_score, state_action_value, next_state_action_value)
        self.q_state_action_values[dist_y][dist_x][n][e][s][w][current][action] += update
        return

    def sarsa(self):
        self.working_epsilon = self.epsilon
        results = []
        perfect_games = 0
        for i in range(self.episodes):
            self.game.reset()
            print(i)

            state = self.state_function()
            # print(state)

            action = self.action_choice(state)
            # print(action)

            while self.game.alive:
                # choosing action

                old_score = self.game.score
                self.game.direction = action

                self.game.update_state()

                next_state = self.state_function()
                next_action = self.action_choice(next_state)

                # Updating
                self.learning_update_function(old_score, state, action, next_state, next_action)

                state, action = next_state, next_action
            self.working_epsilon = max(self.epsilon_min, self.working_epsilon * self.epsilon_decay)
            results.append(self.game.score)
            if self.game.score == 30:
                perfect_games += 1

        print(perfect_games)

        x = [(i // 1000) for i in range(self.episodes)]
        y = results

        sns.lineplot(x, y)
        plt.show()

        return

    def q_learning(self):
        self.working_epsilon = self.epsilon
        results = []
        perfect_games = 0
        for i in range(self.episodes):
            self.game.reset()
            print(i)

            state = self.state_function()
            # print(state)

            while self.game.alive:
                # choosing action
                action = self.action_choice(state)

                # keep score for reward
                old_score = self.game.score
                # set adn take action
                self.game.direction = action
                self.game.update_state()

                # observe new state
                next_state = self.state_function()

                next_action = self.max_action_choice(next_state)

                # Updating
                self.learning_update_function(old_score, state, action, next_state, next_action)

                state = next_state
            self.working_epsilon = max(self.epsilon_min, self.working_epsilon * self.epsilon_decay)
            results.append(self.game.score)
            if self.game.score == 30:
                perfect_games += 1

        print(perfect_games)

        x = [(i // 1000) for i in range(self.episodes)]
        y = results

        sns.lineplot(x, y)
        plt.show()
        return

    def present_game(self):
        self.game.reset()
        self.working_epsilon = self.epsilon_min
        state = self.state_function()
        action = self.action_choice(state)
        self.game.print_state()
        print(state, action)

        while self.game.alive:
            time.sleep(0.75)
            # choosing action

            self.game.direction = action

            self.game.update_state()

            next_state = self.state_function()
            next_action = self.max_action_choice(next_state)

            state, action = next_state, next_action
            self.game.print_state()
            print(state, action)

        return


methods = {"Q-Learning", "Sarsa"}
space = {"Coordinates", "Distance and Collision", "Indicators"}

test_agent_s = SnakeTDAgent(statespace="Indicators", game_size=5, episodes=100000, growing=True, epsilon=1,
                            epsilon_decay=0.99, epsilon_min=0.1, gamma=1, alpha=0.05)
test_agent_s.sarsa()

test_agent_q_l = SnakeTDAgent(statespace="Indicators", game_size=5, episodes=100000, growing=True, epsilon=1,
                              epsilon_decay=0.99, epsilon_min=0.1, gamma=1, alpha=0.05)
test_agent_q_l.q_learning()

epsilons = [0.1, 0.2]
alphas = []
