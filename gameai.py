import pygame
import random
import sys

# stuff
GAME_WIDTH = 256
GAME_HEIGHT = 256
TILE_SIZE = 16

# scale factor for the window size, since 256x256 is pretty small
SCALE = 3
WINDOW_WIDTH = GAME_WIDTH * SCALE
WINDOW_HEIGHT = GAME_HEIGHT * SCALE
FPS = 30


class SnakeGameAI:
    def __init__(self, num_snakes=1):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SCALED)
        pygame.display.set_caption("Snake Game AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)
        self.large_font = pygame.font.SysFont(None, 48)
        self.w = self.screen.get_width()
        self.h = self.screen.get_height()
        self.num_snakes = num_snakes
        self.colors = [
            (255, 80, 80),
            (80, 255, 120),
            (80, 180, 255),
            (255, 220, 80),
            (200, 100, 255),
            (80, 255, 255)
        ]
        self.reset()

    def reset(self):
        center_x = (self.w // 2 // TILE_SIZE) * TILE_SIZE
        center_y = (self.h // 2 // TILE_SIZE) * TILE_SIZE

        start_positions = [
            pygame.Vector2(center_x, center_y),
            pygame.Vector2(center_x - TILE_SIZE * 4, center_y),
            pygame.Vector2(center_x + TILE_SIZE * 4, center_y),
            pygame.Vector2(center_x, center_y - TILE_SIZE * 4),
            pygame.Vector2(center_x, center_y + TILE_SIZE * 4)
        ]

        self.snakes = []
        self.directions = []
        self.heads = []
        self.alive_status = []
        self.scores = []
        self.frame_iterations = []

        for i in range(self.num_snakes):
            start = start_positions[i % len(start_positions)].copy()
            self.snakes.append([start])
            self.directions.append(pygame.Vector2(TILE_SIZE, 0))
            self.heads.append(start.copy())
            self.alive_status.append(True)
            self.scores.append(0)
            self.frame_iterations.append(0)

        self.apple = self.spawn_apple()
        self.game_over = False

    def spawn_apple(self):
        taken = [segment for snake in self.snakes for segment in snake]
        while True:
            pos = pygame.Vector2(
                random.randint(1, self.w // TILE_SIZE - 2) * TILE_SIZE,
                random.randint(1, self.h // TILE_SIZE - 2) * TILE_SIZE
            )
            if pos not in taken:
                return pos

    def play_step(self, actions):
        single_mode = (
            isinstance(actions, list) and len(actions) == 3 and
            all(isinstance(x, int) for x in actions) and self.num_snakes == 1
        )

        if single_mode:
            actions = [actions]
        elif len(actions) != self.num_snakes:
            raise ValueError(f"Expected {self.num_snakes} actions, got {len(actions)}")

        rewards = [0] * self.num_snakes
        dones = [False] * self.num_snakes

        for idx, action in enumerate(actions):
            if not self.alive_status[idx]:
                continue

            self.frame_iterations[idx] += 1
            self._move(action, idx)

            if self.is_collision(self.heads[idx], snake_idx=idx) or self.frame_iterations[idx] > 100 * len(self.snakes[idx]):
                self.alive_status[idx] = False
                self.snakes[idx] = []
                self.directions[idx] = pygame.Vector2(0, 0)
                rewards[idx] = -10
                dones[idx] = True
                continue

            self.snakes[idx].insert(0, self.heads[idx])

            if self.heads[idx] == self.apple:
                self.scores[idx] += 1
                rewards[idx] = 10
                self.frame_iterations[idx] = 0
                self._place_food()
            else:
                self.snakes[idx].pop()

        self._update_ui()
        # self.clock.tick(FPS)

        if single_mode:
            return rewards[0], dones[0], self.scores[0]

        return rewards, dones, self.scores

    def is_collision(self, pt=None, snake_idx=0):
        if pt is None:
            pt = self.heads[snake_idx]

        if pt.x > self.w - 2 * TILE_SIZE or pt.x < TILE_SIZE or pt.y > self.h - 2 * TILE_SIZE or pt.y < TILE_SIZE:
            return True

        for idx, snake in enumerate(self.snakes):
            if not snake:
                continue
            if idx == snake_idx:
                if pt in snake[1:]:
                    return True
            else:
                if pt in snake:
                    return True

        return False

    def _update_ui(self):
        self.screen.fill("black")
        pygame.draw.rect(self.screen, "white", self.screen.get_rect(), 16)

        for idx, snake in enumerate(self.snakes):
            if not self.alive_status[idx] or not snake:
                continue
            color = self.colors[idx % len(self.colors)]
            body_surf = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
            body_surf.fill((*color, 180))
            for segment in snake:
                self.screen.blit(body_surf, (segment.x, segment.y))

        pygame.draw.rect(self.screen, "red", (self.apple.x, self.apple.y, TILE_SIZE, TILE_SIZE))

        score_text = self.font.render(
            " ".join([f"S{i}:{self.scores[i]}" for i in range(self.num_snakes)]),
            True, "white"
        )
        self.screen.blit(score_text, (20, 20))

        pygame.display.flip()

    def _move(self, action, idx):
        clock_wise = [
            pygame.Vector2(TILE_SIZE, 0),
            pygame.Vector2(0, TILE_SIZE),
            pygame.Vector2(-TILE_SIZE, 0),
            pygame.Vector2(0, -TILE_SIZE)
        ]
        current_dir = self.directions[idx]
        dir_idx = clock_wise.index(current_dir)

        if action == [1, 0, 0]:
            new_dir = clock_wise[dir_idx]
        elif action == [0, 1, 0]:
            new_dir = clock_wise[(dir_idx + 1) % 4]
        else:
            new_dir = clock_wise[(dir_idx - 1) % 4]

        self.directions[idx] = new_dir
        x = self.snakes[idx][0].x
        y = self.snakes[idx][0].y
        x += new_dir.x
        y += new_dir.y
        self.heads[idx] = pygame.Vector2(x, y)

    def _place_food(self):
        taken = [segment for snake in self.snakes for segment in snake]
        while True:
            x = random.randint(1, (self.w // TILE_SIZE) - 2) * TILE_SIZE
            y = random.randint(1, (self.h // TILE_SIZE) - 2) * TILE_SIZE
            self.apple = pygame.Vector2(x, y)
            if self.apple not in taken:
                return
