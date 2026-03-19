import pygame
import random
import numpy as np


class SnakeEnv:
    BLOCK = 20
    WIDTH = 400
    HEIGHT = 400
    DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def __init__(self, render=True, speed=50):
        self.render = render
        self.speed = speed

        if render:
            pygame.init()
            self.display = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Snake RL")
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.direction = (1, 0)
        self.head = [self.WIDTH // 2, self.HEIGHT // 2]
        self.snake = [self.head[:]]
        self.score = 0
        self.frame = 0

        self._spawn_food()
        return self._state()

    def _spawn_food(self):
        while True:
            x = random.randrange(0, self.WIDTH, self.BLOCK)
            y = random.randrange(0, self.HEIGHT, self.BLOCK)
            if [x, y] not in self.snake:
                self.food = [x, y]
                break

    def step(self, action):
        self.frame += 1
        self._move(action)
        self.snake.insert(0, self.head[:])

        reward = -0.1  # small penalty (encourages efficiency)
        done = False

        if self._collision() or self.frame > 100 * len(self.snake):
            return self._state(), -10, True

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._spawn_food()
        else:
            self.snake.pop()

        if self.render:
            self._draw()

        return self._state(), reward, done

    def _collision(self, pt=None):
        if pt is None:
            pt = self.head

        return (
            pt[0] < 0 or pt[0] >= self.WIDTH or
            pt[1] < 0 or pt[1] >= self.HEIGHT or
            pt in self.snake[1:]
        )

    def _move(self, action):
        idx = self.DIRECTIONS.index(self.direction)

        if action == 0:
            new_dir = self.DIRECTIONS[idx]
        elif action == 1:
            new_dir = self.DIRECTIONS[(idx + 1) % 4]
        else:
            new_dir = self.DIRECTIONS[(idx - 1) % 4]

        self.direction = new_dir
        self.head[0] += new_dir[0] * self.BLOCK
        self.head[1] += new_dir[1] * self.BLOCK

    def _state(self):
        head = self.head

        def p(x, y): return [head[0] + x, head[1] + y]

        dir_l = self.direction == (-1, 0)
        dir_r = self.direction == (1, 0)
        dir_u = self.direction == (0, -1)
        dir_d = self.direction == (0, 1)

        return np.array([
            (dir_r and self._collision(p(20, 0))) or
            (dir_l and self._collision(p(-20, 0))) or
            (dir_u and self._collision(p(0, -20))) or
            (dir_d and self._collision(p(0, 20))),

            (dir_u and self._collision(p(20, 0))) or
            (dir_d and self._collision(p(-20, 0))) or
            (dir_l and self._collision(p(0, -20))) or
            (dir_r and self._collision(p(0, 20))),

            (dir_d and self._collision(p(20, 0))) or
            (dir_u and self._collision(p(-20, 0))) or
            (dir_r and self._collision(p(0, -20))) or
            (dir_l and self._collision(p(0, 20))),

            self.food[0] < head[0],
            self.food[0] > head[0],
            self.food[1] < head[1],
            self.food[1] > head[1],
        ], dtype=int)

    def _draw(self):
        self.display.fill((0, 0, 0))

        for s in self.snake:
            pygame.draw.rect(self.display, (0, 255, 0), (*s, self.BLOCK, self.BLOCK))

        pygame.draw.rect(self.display, (255, 0, 0), (*self.food, self.BLOCK, self.BLOCK))

        pygame.display.flip()
        self.clock.tick(self.speed)
