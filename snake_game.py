import random
import math
import numpy as np


class Game:

    def __init__(self, n=5, walls=True, growing=True, reward=1):
        self.size = n

        self.fruit_reward = reward

        self.walls = walls
        self.growing = growing

        self.reset()
        self.fruit = [2, n-1]

        #print("It's alive")
        return

    def reset(self):
        self.snake_position_head = [self.size // 2 + 1, self.size // 2 + 1]
        self.snake_position_body = [[self.size // 2, self.size // 2 + 1]]
        self.snakesize = len(self.snake_position_body.copy()) + 1
        self.direction = 0

        self.fruit = []  # 1, n]
        self.new_fruit()

        self.step_limit = 1000 #3 * self.size * math.log(self.snakesize, 2)
        self.step_countdown = self.step_limit

        self.score = 0

        self.alive = True
        # print("It's alive")
        return

    def new_fruit(self):
        new_fruit = self.snake_position_head
        while new_fruit in self.snake_position_body + [self.snake_position_head]:
            new_fruit = [random.randint(1, self.size), random.randint(1, self.size)]
        self.fruit = new_fruit
    
    def state_as_positions(self):
        return self.snake_position_head[0], self.snake_position_head[1], self.fruit[0], self.fruit[1]

    def collision_indicators(self):
        collision = [0]*5  # np.zeros(5)
        if self.snake_position_head[0] == 1 or self.snake_position_head[0] == 0 or \
                self.snake_position_head[1] == 0 or self.snake_position_head[1] == self.size + 1 or \
                (lambda x: [x[0] - 1, x[1]])(self.snake_position_head) in self.snake_position_body:  # north
            collision[0] = 1
        if self.snake_position_head[1] == self.size or self.snake_position_head[1] == self.size + 1 or \
                self.snake_position_head[0] == 0 or self.snake_position_head[0] == self.size + 1 or \
                (lambda x: [x[0], x[1] + 1])(self.snake_position_head) in self.snake_position_body:  # east
            collision[1] = 1
        if self.snake_position_head[0] == self.size or self.snake_position_head[0] == self.size + 1 or \
                self.snake_position_head[1] == 0 or self.snake_position_head[1] == self.size + 1 or \
            (lambda x: [x[0] + 1, x[1]])(self.snake_position_head) in self.snake_position_body:  # south
            collision[2] = 1
        if self.snake_position_head[1] == 1 or self.snake_position_head[1] == 0 or \
                self.snake_position_head[0] == 0 or self.snake_position_head[0] == self.size + 1 or \
                (lambda x: [x[0], x[1]-1])(self.snake_position_head) in self.snake_position_body:  # west
            collision[3] = 1
        if not self.alive:  # current body position occupied
            collision[4] = 1
        return collision.copy()

    def state_as_distance_vector(self):
        distance_vector = [self.snake_position_head[0] - self.fruit[0] + self.size + 1, self.snake_position_head[1] - self.fruit[1] + self.size + 1]
        collision = self.collision_indicators()
        return distance_vector, collision

    def state_as_indicators(self):
        if self.snake_position_head[0] < self.fruit[0]:
            indicator_y = 0
        elif self.snake_position_head[0] == self.fruit[0]:
            indicator_y = 1
        else:
            indicator_y = 2
        if self.snake_position_head[1] < self.fruit[1]:
            indicator_x = 0
        elif self.snake_position_head[1] == self.fruit[1]:
            indicator_x = 1
        else:
            indicator_x = 2

        collision = self.collision_indicators()
        return [indicator_y, indicator_x], collision

    def update_state(self):
        old_distance = math.sqrt(
            (self.snake_position_head[0] - self.fruit[0]) ** 2 + (self.snake_position_head[1] - self.fruit[0]) ** 2)

        # update position of snake body
        self.snake_position_body.append(self.snake_position_head.copy())
        # update position of snake head
        if self.direction == 0:     #north
            self.snake_position_head[0] -= 1
        elif self.direction == 1:   #east
            self.snake_position_head[1] += 1
        elif self.direction == 2:   #south
            self.snake_position_head[0] += 1
        else:       #west
            self.snake_position_head[1] -= 1

        # bonus reward for getting closer to the fruit and negative reward for getting firther away
        new_distance = math.sqrt(
            (self.snake_position_head[0] - self.fruit[0]) ** 2 + (self.snake_position_head[1] - self.fruit[0]) ** 2)
        # print(new_distance == old_distance)
        #if old_distance > new_distance:
        #    self.score += 0.1
        #else:
        #    self.score += -0.1

        # checking for possible death
        lower_wall = self.size+1
        collision_body = self.snake_position_head in self.snake_position_body
        collision_wall = (lambda x: (x[0] % lower_wall) * (x[1] % lower_wall))(
            self.snake_position_head) == 0 and self.walls
        starvation = self.step_countdown == 0

        if collision_body:
            self.alive = False
            #self.score += -10
            #print("body collision", self.snake_position_head, self.snake_position_body)
        elif collision_wall:
            self.alive = False
            #self.score += -10
            #print("wall collision")
        elif starvation:
            self.alive = False

        self.step_countdown -= 1

        if self.snake_position_head != self.fruit or not self.growing:
            self.snake_position_body.pop(0)

        if self.snake_position_head == self.fruit:
            self.score += self.fruit_reward
            if self.snakesize < 24:
                self.snakesize += 1
            else:
                self.score += 6
                self.alive = False

            self.new_fruit()
            #self.step_countdown = self.step_limit

        return

    def print_state(self):
        # set field
        field = [["#"]+[" " for i in range(self.size)]+["#"] for j in range(self.size)]

        field = [["#"*(self.size + 2)]] + field + [["#"*(self.size + 2)]]

        for x, y in self.snake_position_body:
            field[x][y] = "s"
        try:
            field[self.snake_position_head[0]][self.snake_position_head[1]] = "S"
        except:
            print('Wall collision')

        field[self.fruit[0]][self.fruit[1]] = "o"


        # print state
        field_string = ""

        for row in field:
            for position in row:
                field_string += position
            field_string += "\n"
        print(field_string)
        print("score:", self.score)
        return
