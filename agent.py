import numpy as np
import snake_game
import time


class SnakeAgent:

    def __init__(self, game_size=5, method='Q-Learning', gamma=1, episodes=10000, growing=False):
        self.q_state_action_values = [
            [[[np.array([0, 0, 0, 0]) for i in range(game_size + 2)] for j in range(game_size + 2)] for k in
             range(game_size + 2)] for l in range(game_size + 2)]

        for i in range(0, game_size + 2):
            for j in range(0, game_size + 2):
                for k in range(0, game_size + 2):
                    for l in range(0, game_size + 2):
                        if (i % (game_size+1)) * (j % (game_size+1)) == 0:
                            self.q_state_action_values[i][j][k][l] = np.array([0., 0., 0., 0.])

        self.lm = method
        self.grow = growing
        self.episodes = episodes
        self.game_size = game_size


    def dqn(self):
        #self.
        return

    def learning(self, epsilon=0.15, gamma=1, alpha=0.8):
        for i in range(self.episodes):
            print(i)
            game = snake_game.Game(n=self.game_size, growing=self.grow)
            while game.alive:
                # choosing action
                h_0 = game.snake_position_head[0]
                h_1 = game.snake_position_head[1]
                f_0 = game.fruit[0]
                f_1 = game.fruit[1]

                if epsilon > np.random.uniform(0, 1):
                    action = max(
                        [(self.q_state_action_values[h_0][h_1][f_0][f_1][direction], direction) for direction in
                         range(4)])[1]
                else:
                    action = np.random.randint(0, 4)

                old_score = game.score

                game.direction = action
                game.update_state()

                h_0_new = game.snake_position_head[0]
                h_1_new = game.snake_position_head[1]
                f_0_new = game.fruit[0]
                f_1_new = game.fruit[1]

                max_action = max(
                    [(self.q_state_action_values[h_0_new][h_1_new][f_0_new][f_1_new][direction], direction) for
                     direction in range(4)])[1]

                try:
                    reward = game.score - old_score
                    target = reward + gamma * self.q_state_action_values[h_0_new][h_1_new][f_0_new][f_1_new][max_action]
                    update = alpha * (target - self.q_state_action_values[h_0][h_1][f_0][f_1][action])
                    self.q_state_action_values[h_0][h_1][f_0][f_1][action] += update
                except:
                    print(h_0, h_1, f_0, f_1, action)
                    print(h_0_new, h_1_new, f_0_new, f_1_new, max_action[1])

        return

    def present_game(self):
        game = snake_game.Game(n=self.game_size, growing=self.grow)
        game.print_state()
        while game.alive:
            time.sleep(0.75)
            # choosing action
            h_0 = game.snake_position_head[0]
            h_1 = game.snake_position_head[1]
            f_0 = game.fruit[0]
            f_1 = game.fruit[1]

            action = \
            max([(self.q_state_action_values[h_0][h_1][f_0][f_1][direction], direction) for direction in range(4)])[1]

            old_score = game.score

            game.direction = action
            game.update_state()
            game.print_state()
        return



test_agent = SnakeAgent(game_size=5, episodes=100000, growing=True)
#print(test_agent.q_state_action_values)
test_agent.learning(epsilon=0.8, gamma=1, alpha=0.8)
test_agent.present_game()
#print(test_agent.q_state_action_values)
