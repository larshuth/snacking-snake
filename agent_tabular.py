import numpy as np
import snake_game
import time
import matplotlib.pyplot as plt


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
                [[[np.array([0, 0, 0, 0]) for i in range(self.game_size + 2)] for j in range(self.game_size + 2)] for k in
                 range(self.game_size + 2)] for l in range(self.game_size + 2)]
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
            self.td_update = self.td_update_dist
            self.epsilon_action_choice = self.epsilon_action_choice_dist

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
            self.td_update = self.td_update_indicators
            self.epsilon_action_choice = self.epsilon_action_choice_indicators

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
            #print(self.q_state_action_values[h_0][h_1][f_0][f_1][action])
            reward = self.game.score - old_score
            #print(reward)
            target = reward + self.gamma * self.q_state_action_values[h_0_new][h_1_new][f_0_new][f_1_new][
                next_action].copy()
            #print(target)
            update = self.alpha * (target - self.q_state_action_values[h_0][h_1][f_0][f_1][action].copy())
            #print(update)
            self.q_state_action_values[h_0][h_1][f_0][f_1][action] += update
            #print(self.q_state_action_values[h_0][h_1][f_0][f_1][action])
        except:
            print(state, action)
            print(next_state, next_action)

    def epsilon_action_choice_dist(self, state):
        [dist_y, dist_x], [n, e, s, w, current] = state

        #print(dist_y, dist_x, n, e, s, w, current)

        state_value = self.q_state_action_values[dist_y][dist_x][n][e][s][w][current]

        if self.working_epsilon > np.random.uniform(0, 1):
            action = max([(state_value[direction], direction) for direction in range(4)])[1]
        else:
            action = np.random.randint(0, 4)

        return action

    def td_update_dist(self, old_score, state, action, next_state, next_action):
        [dist_y, dist_x], [n, e, s, w, current] = state
        [dist_y_new, dist_x_new], [n_new, e_new, s_new, w_new, current_new] = next_state
        try:
            reward = self.game.score - old_score
            target = reward + self.gamma * \
                     self.q_state_action_values[dist_y_new][dist_x_new][n_new][e_new][s_new][w_new][current_new][
                         next_action].copy()
            update = self.alpha * (
                        target - self.q_state_action_values[dist_y][dist_x][n][e][s][w][current][next_action].copy())
            self.q_state_action_values[dist_y][dist_x][n][e][s][w][current][action] += update
        except:
            print(state, action)
            print(next_state, next_action)
        return

    def epsilon_action_choice_indicators(self, state):
        [indicator_y, indicator_x], [n, e, s, w, current] = state
        state_value = self.q_state_action_values[indicator_y][indicator_x][n][e][s][w][current]

        if self.working_epsilon > np.random.uniform(0, 1):
            action = max([(state_value[direction], direction) for direction in range(4)])[1]
        else:
            action = np.random.randint(0, 4)

        return action

    def td_update_indicators(self, old_score, state, action, next_state, next_action):
        [indicator_y, indicator_x], [n, e, s, w, current] = state
        [indicator_y_new, indicator_x_new], [n_new, e_new, s_new, w_new, current_new] = next_state
        try:
            reward = self.game.score - old_score
            target = reward + self.gamma * \
                     self.q_state_action_values[indicator_y_new][indicator_x_new][n_new][e_new][s_new][w_new][
                         current_new][next_action].copy()
            update = self.alpha * (
                        target - self.q_state_action_values[indicator_y][indicator_x][n][e][s][w][current][next_action].copy())
            self.q_state_action_values[indicator_y][indicator_x][n][e][s][w][current][action] += update
        except:
            print(state, action)
            print(next_state, next_action)
        return

    def sarsa(self):
        self.working_epsilon = self.epsilon
        results = []
        for i in range(self.episodes):
            self.game.reset()
            print(i)

            state = self.state_function()
            #print(state)

            action = self.epsilon_action_choice(state)
            #print(action)

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
            results.append(self.game.score)

        plt.plot([i * self.episodes // 100 for i in range(int(100))], [
            sum([results[j + i * self.episodes // 100] for j in range(int(self.episodes // 100))]) / (
                        self.episodes // 100) for i in range(100)])
        plt.show()

        #plt.plot(range(self.episodes), results)
        #plt.show()

        return

    def q_learn(self):
        self.working_epsilon = self.epsilon
        results = []
        for i in range(self.episodes):
            self.game.reset()
            print(i)

            state = self.state_function()
            # print(state)

            action = self.epsilon_action_choice(state)
            # print(action)

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
            results.append(self.game.score)

        plt.plot([i * self.episodes // 100 for i in range(int(100))], [
            sum([results[j + i * self.episodes // 100] for j in range(int(self.episodes // 100))]) / (
                    self.episodes // 100) for i in range(100)])
        plt.show()


        for i in range(self.episodes):
            print(i)
            game = snake_game.Game(n=self.game_size, growing=self.grow, reward=1)

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

            if self.statespace == "Coordinates":
                state = game.state_as_positions()
                h_0, h_1, f_0, f_1 = state
                state_value = self.q_state_action_values[h_0][h_1][f_0][f_1]
            elif self.statespace == "Distance and Collision":
                state = game.state_as_distance_vector()
                [indicator_y, indicator_x], [n, e, s, w, current] = state
                state_value = self.q_state_action_values[indicator_y][indicator_x][n][e][s][w][current]

            elif self.statespace == "Indicators":
                state = game.state_as_indicators()
                [dist_y, dist_x], [n, e, s, w, current] = state

                state_value = self.q_state_action_values[dist_y][dist_x][n][e][s][w][current]

            action = max([(state_value[direction], direction) for direction in range(4)])[1]

            old_score = game.score

            game.direction = action
            print(state, action)

            game.update_state()
            game.print_state()

        state = game.state_as_indicators()
        print(state)
        return



methods = {"Q-Learning", "Sarsa"}
space = {"Coordinates", "Distance and Collision", "Indicators"}

test_agent = SnakeTDAgent(statespace="Indicators", game_size=5, episodes=100000, growing=True, epsilon=1, epsilon_decay=0.99, epsilon_min=0.1, gamma=1, alpha=0.05)

#print(test_agent.q_state_action_values)
test_agent.sarsa()
test_agent.present_game()
#print(test_agent.q_state_action_values)
