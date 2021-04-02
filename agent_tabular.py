import numpy as np
import snake_game
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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

    def sarsa(self, averaged_stats=True):
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

        #x = [(i // 200) for i in range(self.episodes)]
        #y = results

        #sns.lineplot(x, y)
        #plt.show()

        if averaged_stats:
            average_over_hundred = [
                sum([results[j + (i * self.episodes // 100)] for j in range(int(self.episodes // 100))]) / (
                        self.episodes // 100) for i in range(100)]
            return average_over_hundred
        else:
            return results

    def q_learning(self, averaged_stats=True):
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

        #x = [(i // 200) for i in range(self.episodes)]
        #y = results

        #sns.lineplot(x, y)
        #plt.show()

        if averaged_stats:
            average_over_hundred = [
                sum([results[j + (i * self.episodes // 100)] for j in range(int(self.episodes // 100))]) / (
                            self.episodes // 100) for i in range(100)]
            return average_over_hundred
        else:
            return results

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

# test_agent_s = SnakeTDAgent(statespace="Indicators", game_size=5, episodes=20000, growing=True, epsilon=1,
#                             epsilon_decay=0.99, epsilon_min=0.1, gamma=1, alpha=0.05)
# test_agent_s.sarsa()

# test_agent_q_l = SnakeTDAgent(statespace="Indicators", game_size=5, episodes=20000, growing=True, epsilon=1,
#                               epsilon_decay=0.99, epsilon_min=0.1, gamma=1, alpha=0.05)
# test_agent_q_l.q_learning()


def compare_parameters(lineplot=False, hist=100):
    ep = 100000

    rough_alphas = [0.1 + (i * 0.2) for i in range(5)]  # 0.005, 0.01, 0.05, 0.1, 0.15]
    rough_gammas = [0.1 + (i * 0.2) for i in range(5)]  # 0.25, 0.275, 0.3, 0.325, 0.35]

    fig, axes = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(16, 8))
    if lineplot:
        fig.suptitle('Testing parameters for the "Distance and Collision" statespace on the Sarsa algorithm.')
    else:
        fig.suptitle('Testing parameters for the "Distance and Collision" statespace on the Sarsa algorithm. Distribution of achieved scores over the last '+str(hist)+' games, averaged over 5 runs.')

    for i in range(len(rough_alphas)):
        test_alpha = rough_alphas[i]
        for j in range(len(rough_gammas)):
            test_gamma = rough_gammas[j]

            if lineplot:
                averages = pd.DataFrame({'Episodes': [i*(ep//100) for i in range(100)]})    #[i for i in range(ep)]})
            else:
                averages = pd.DataFrame({'Episodes': [i for i in range(ep-hist, ep)]})  # [i for i in range(ep)]})

            for run in range(5):
                agent = SnakeTDAgent(statespace="Distance and Collision", game_size=5, episodes=ep, growing=True, epsilon=1,
                                     epsilon_decay=0.999, epsilon_min=0.1, gamma=test_gamma, alpha=test_alpha)
                if lineplot:
                    results = agent.q_learning(averaged_stats=True).copy()
                    averages[str(run)] = results
                    averages[str(run)] = averages[str(run)].rolling(5).mean()
                else:
                    results = agent.q_learning(averaged_stats=False).copy()
                    averages[str(run)] = results[-hist:]
            df = pd.DataFrame(averages)
            print(df)
            if lineplot:
                df = df.melt('Episodes', var_name='runs', value_name='average score')
                g = sns.lineplot(ax=axes[i][j], x='Episodes', y='average score', hue='runs', data=df, legend=False)
            else:
                df = df.melt('Episodes', var_name='runs', value_name='Score')
                g = sns.histplot(ax=axes[i][j], x='Score', stat='probability', data=df, legend=False)

    cols = ['Alpha:  {}'.format(round(alpha, 2)) for alpha in rough_alphas]
    rows = ['Gamma: {}'.format(round(gamma, 2)) for gamma in rough_gammas]

    pad = 5  # in points

    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for ax, row in zip(axes[:, 0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
    plt.show()
    return


def compare_algorithms_and_statespaces():
    #statespace algo alpha gamma
    params = [#["Indicators", "Sarsa", 0.3, 0.3, 0, 0], ["Indicators", "Q", 0.3, 0.1, 0, 1],
              ["Distance and Collision", "Sarsa", 0.7, 0.3, 0, 0], ["Distance and Collision", "Q", 0.1, 0.3, 0, 1]]

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(16, 8))
    fig.suptitle('Testing parameters for the "Indicators" statespace on the Q-Learning algorithm')


    for meters in params:
        if meters[0] == "Distance and Collision":
            ep = 150000
        else:
            ep = 3000
        averages = {'Episodes': [i * (ep // 100) for i in range(100)]}
        for run in range(3):
            agent = SnakeTDAgent(statespace=meters[0], game_size=5, episodes=ep, growing=True, epsilon=1,
                                 epsilon_decay=0.99, epsilon_min=0.1, gamma=meters[3], alpha=meters[2])
            if meters[1] == "Sarsa":
                run = agent.sarsa().copy()
            else:
                run = agent.q_learning().copy()
            averages.update({str(run): run})
        df = pd.DataFrame(averages)
        #print(df)
        df = df.melt('Episodes', var_name='runs', value_name='average reward per '+str(ep/100)+' episodes (games)')
        axes[meters[5]].set_title(meters[1] + "  on  " + meters[0])
        g = sns.lineplot(ax=axes[meters[5]], x='Episodes', y='average reward per '+str(ep/100)+' episodes (games)', hue='runs', data=df, legend=False)

        pad = 5  # in points

    plt.show()


# compare_algorithms_and_statespaces()
compare_parameters(lineplot=False)
