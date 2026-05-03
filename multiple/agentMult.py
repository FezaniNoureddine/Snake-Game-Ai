import random
import torch
import pygame
from collections import deque
from brainMult import Linear_QNet, QTrainer
from gameaiMult import SnakeGameAI

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 80
        self.gamma = 0.9
        self.memory = deque(maxlen=100_000)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)

    def load(self, path='model_best.pth'):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def get_state(self, game, snake_idx):
        snake = game.snakes[snake_idx]
        head = snake[0]
        point_l = pygame.Vector2(head.x - 16, head.y)
        point_r = pygame.Vector2(head.x + 16, head.y)
        point_u = pygame.Vector2(head.x, head.y - 16)
        point_d = pygame.Vector2(head.x, head.y + 16)

        dir_l = game.directions[snake_idx] == pygame.Vector2(-16, 0)
        dir_r = game.directions[snake_idx] == pygame.Vector2(16, 0)
        dir_u = game.directions[snake_idx] == pygame.Vector2(0, -16)
        dir_d = game.directions[snake_idx] == pygame.Vector2(0, 16)

        state = [
            int((dir_r and game.is_collision(point_r, snake_idx)) or
                (dir_l and game.is_collision(point_l, snake_idx)) or
                (dir_u and game.is_collision(point_u, snake_idx)) or
                (dir_d and game.is_collision(point_d, snake_idx))),
            int((dir_u and game.is_collision(point_r, snake_idx)) or
                (dir_d and game.is_collision(point_l, snake_idx)) or
                (dir_l and game.is_collision(point_u, snake_idx)) or
                (dir_r and game.is_collision(point_d, snake_idx))),
            int((dir_d and game.is_collision(point_r, snake_idx)) or
                (dir_u and game.is_collision(point_l, snake_idx)) or
                (dir_r and game.is_collision(point_u, snake_idx)) or
                (dir_l and game.is_collision(point_d, snake_idx))),
            int(dir_l),
            int(dir_r),
            int(dir_u),
            int(dir_d),
            int(game.apple.x < head.x),
            int(game.apple.x > head.x),
            int(game.apple.y < head.y),
            int(game.apple.y > head.y)
        ]

        return state

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

    def get_actions(self, states):
        self.epsilon = max(0, 80 - self.n_games)
        final_moves = []

        for state in states:
            if random.randint(0, 200) < self.epsilon:
                move = random.randint(0, 2)
            else:
                state0 = torch.tensor(state, dtype=torch.float)
                prediction = self.model(state0)
                move = torch.argmax(prediction).item()

            action = [0, 0, 0]
            action[move] = 1
            final_moves.append(action)

        return final_moves


def train():
    record = 0
    agent = Agent()
    game = SnakeGameAI(num_snakes=4)
    save_milestones = [1, 5, 10, 20, 50, 100, 200, 500, 1000]

    while True:
        alive_indices = [i for i, alive in enumerate(game.alive_status) if alive]
        if not alive_indices:
            continue

        state_old = [agent.get_state(game, idx) for idx in alive_indices]
        final_moves = agent.get_actions(state_old)

        actions_full = [None] * game.num_snakes
        for idx, move in zip(alive_indices, final_moves):
            actions_full[idx] = move

        rewards, dones, scores = game.play_step(actions_full)

        next_states = []
        for idx in alive_indices:
            if dones[idx]:
                next_states.append([0] * 11)
            else:
                next_states.append(agent.get_state(game, idx))

        for idx, state, action, reward, next_state, done in zip(alive_indices, state_old, final_moves, [rewards[i] for i in alive_indices], next_states, [dones[i] for i in alive_indices]):
            agent.train_short_memory(state, action, reward, next_state, done)
            agent.remember(state, action, reward, next_state, done)

        if all(not alive for alive in game.alive_status):
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if agent.n_games in save_milestones:
                agent.model.save(f'model_game_{agent.n_games}.pth')

            max_score = max(scores) if scores else 0
            if max_score > record:
                record = max_score
                agent.model.save('model_best.pth')

            print(f'Game {agent.n_games} | Score {max_score} | Record {record}')


if __name__ == "__main__":
    train()