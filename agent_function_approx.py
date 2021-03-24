import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import multiprocessing as mp
import snake_game
import time


class SnakeAgent:

    def __init__(self, game_size=5, growing=False, epsilon=1, epsilon_decay=0.99, epsilon_min=0.1, gamma=1, alpha=0.8,
                 batch=100, hidden_layer_nodes=128, episodes=10000):

        self.action_space = 4
        self.state_space = 7        # [dist y][dist x][obstacle north][east][south][west][current position]
        self.hidden_layer_nodes = hidden_layer_nodes
        self.network_model = self.buid_networm()

        self.game_size = game_size
        self.grow = growing

        self.episodes = episodes
        self.batch_size = batch

        self.epsilon, self.epsilon_decay, self.epsilon_min = epsilon, epsilon_decay, epsilon_min
        self.gamma, self.alpha = gamma, alpha

    def buid_networm(self):     # pythonprogramming.net/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
        model = Sequential()

        model.add(Dense(self.hidden_layer_nodes, input_shape=(self.state_space,), activation='softmax'))
        model.add(Dense(self.action_space), activation='softmax')
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def q_learn(self):
        i = 0
        # for m = 1, ... MAX_STEPS do
        while i < self.episodes:
            i += 1
            # gather
            #pool = mp.Pool(mp.cpu_count())
            #results = pool.map(howmany_within_range_rowonly, [row for row in data])
            if self.epsilon > np.random.uniform(0, 1):
                action = max([(self.network_model.predict(), direction) for direction in
                 range(4)])[1]
            else:
                action = np.random.randint(0, 4)


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
