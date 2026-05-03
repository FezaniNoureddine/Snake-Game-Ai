import pygame
from gameai import SnakeGameAI
from brain import Linear_QNet, QTrainer
import random
import torch
from collections import deque

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 80  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=100_000)  # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = pygame.Vector2(head.x - 16, head.y)
        point_r = pygame.Vector2(head.x + 16, head.y)
        point_u = pygame.Vector2(head.x, head.y - 16)
        point_d = pygame.Vector2(head.x, head.y + 16)

        dir_l = game.direction == pygame.Vector2(-16, 0)
        dir_r = game.direction == pygame.Vector2(16, 0)
        dir_u = game.direction == pygame.Vector2(0, -16)
        dir_d = game.direction == pygame.Vector2(0, 16)

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.apple.x < head.x,  # food left
            game.apple.x > head.x,  # food right
            game.apple.y < head.y,  # food up
            game.apple.y > head.y   # food down
        ]

        return [int(x) for x in state]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > 1000:
            mini_sample = random.sample(self.memory, 1000)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    save_milestones = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

    while True:
        # Get old state
        state_old = agent.get_state(game)

        # Get move
        final_move = agent.get_action(state_old)

        # Perform move
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short memory (current step)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Store in long-term memory
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()  # Train on a random batch from memory

            # Milestone Saves
            if agent.n_games in save_milestones:
                agent.model.save(f'model_game_{agent.n_games}.pth')

            # Pro-Tip: Save Best Model
            if score > record:
                record = score
                agent.model.save('model_best.pth')

            print(f'Game {agent.n_games} | Score {score} | Record {record}')

if __name__ == "__main__":
    train()