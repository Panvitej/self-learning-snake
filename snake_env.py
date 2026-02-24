import pygame
import random
import numpy as np

BLOCK = 20
WIDTH = 400
HEIGHT = 400
SPEED = 50


class SnakeEnv:

    def __init__(self):
        pygame.init()
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake RL")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = (1, 0)
        self.head = [WIDTH // 2, HEIGHT // 2]
        self.snake = [self.head[:]]
        self.spawn_food()
        self.score = 0
        self.frame_iteration = 0
        return self.get_state()

    def spawn_food(self):
        x = random.randint(0, (WIDTH - BLOCK) // BLOCK) * BLOCK
        y = random.randint(0, (HEIGHT - BLOCK) // BLOCK) * BLOCK
        self.food = [x, y]

    def step(self, action):
        self.frame_iteration += 1
        self._move(action)
        self.snake.insert(0, self.head[:])

        reward = 0
        done = False

        if self.is_collision():
            reward = -10
            done = True
            return self.get_state(), reward, done

        if self.head == self.food:
            reward = 10
            self.score += 1
            self.spawn_food()
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED)

        return self.get_state(), reward, done

    def is_collision(self):
        if (
            self.head[0] < 0 or self.head[0] >= WIDTH or
            self.head[1] < 0 or self.head[1] >= HEIGHT or
            self.head in self.snake[1:]
        ):
            return True
        return False

    def _move(self, action):
        # action: [straight, right, left]
        directions = [(1,0), (0,1), (-1,0), (0,-1)]
        idx = directions.index(self.direction)

        if np.array_equal(action, [1,0,0]):
            new_dir = directions[idx]
        elif np.array_equal(action, [0,1,0]):
            new_dir = directions[(idx + 1) % 4]
        else:
            new_dir = directions[(idx - 1) % 4]

        self.direction = new_dir
        self.head[0] += self.direction[0] * BLOCK
        self.head[1] += self.direction[1] * BLOCK

    def get_state(self):
        head = self.head

        point_l = [head[0] - BLOCK, head[1]]
        point_r = [head[0] + BLOCK, head[1]]
        point_u = [head[0], head[1] - BLOCK]
        point_d = [head[0], head[1] + BLOCK]

        dir_l = self.direction == (-1, 0)
        dir_r = self.direction == (1, 0)
        dir_u = self.direction == (0, -1)
        dir_d = self.direction == (0, 1)

        state = [
            (dir_r and self._collision(point_r)) or
            (dir_l and self._collision(point_l)) or
            (dir_u and self._collision(point_u)) or
            (dir_d and self._collision(point_d)),

            (dir_u and self._collision(point_r)) or
            (dir_d and self._collision(point_l)) or
            (dir_l and self._collision(point_u)) or
            (dir_r and self._collision(point_d)),

            (dir_d and self._collision(point_r)) or
            (dir_u and self._collision(point_l)) or
            (dir_r and self._collision(point_u)) or
            (dir_l and self._collision(point_d)),

            self.food[0] < head[0],
            self.food[0] > head[0],
            self.food[1] < head[1],
            self.food[1] > head[1],
        ]

        return np.array(state, dtype=int)

    def _collision(self, point):
        if (
            point[0] < 0 or point[0] >= WIDTH or
            point[1] < 0 or point[1] >= HEIGHT or
            point in self.snake
        ):
            return True
        return False

    def _update_ui(self):
        self.display.fill((0,0,0))
        for pt in self.snake:
            pygame.draw.rect(self.display, (0,255,0), pygame.Rect(pt[0], pt[1], BLOCK, BLOCK))
        pygame.draw.rect(self.display, (255,0,0), pygame.Rect(self.food[0], self.food[1], BLOCK, BLOCK))
        pygame.display.flip()
