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
    def __init__(self, headless=False):
        pygame.init()
        self.headless = headless
        if not headless:
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SCALED)
            pygame.display.set_caption("Snake Game AI")
            self.w = self.screen.get_width()
            self.h = self.screen.get_height()
        else:
            self.screen = None
            self.w = WINDOW_WIDTH
            self.h = WINDOW_HEIGHT
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)
        self.large_font = pygame.font.SysFont(None, 48)
        self.reset()

    def reset(self):
        self.snake = [pygame.Vector2(
            (self.w // 2 // TILE_SIZE) * TILE_SIZE,
            (self.h // 2 // TILE_SIZE) * TILE_SIZE
        )]
        self.direction = pygame.Vector2(TILE_SIZE, 0)
        self.apple = self.spawn_apple()
        self.game_over = False
        self.score = 0
        self.frame_iteration = 0
        self.head = self.snake[0]

    def spawn_apple(self):
        while True:
            pos = pygame.Vector2(
                random.randint(1, self.w // TILE_SIZE - 2) * TILE_SIZE,
                random.randint(1, self.h // TILE_SIZE - 2) * TILE_SIZE
            )
            if pos not in self.snake:
                return pos

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new Food or just move
        if self.head == self.apple:
            self.score += 1
            reward = 10
            self.frame_iteration = 0
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        #self.clock.tick(FPS)
        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary (respecting the 16px white border)
        if pt.x > self.w - 2*TILE_SIZE or pt.x < TILE_SIZE or pt.y > self.h - 2*TILE_SIZE or pt.y < TILE_SIZE:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        if self.headless or self.screen is None:
            return
        self.screen.fill("black")
        pygame.draw.rect(self.screen, "white", self.screen.get_rect(), 16)

        for segment in self.snake:
            pygame.draw.rect(self.screen, "green", (segment.x, segment.y, TILE_SIZE, TILE_SIZE))

        pygame.draw.rect(self.screen, "red", (self.apple.x, self.apple.y, TILE_SIZE, TILE_SIZE))

        score_text = self.font.render(f"Score: {self.score}", True, "white")
        self.screen.blit(score_text, (20, 20))

        pygame.display.flip()

    def render_game(self, surface, x_offset=0, y_offset=0):
        """Render the game board to a given pygame surface at offset (x_offset, y_offset)"""
        # Draw background
        pygame.draw.rect(surface, "black", (x_offset, y_offset, WINDOW_WIDTH, WINDOW_HEIGHT))
        
        # Draw border
        pygame.draw.rect(surface, "white", (x_offset, y_offset, WINDOW_WIDTH, WINDOW_HEIGHT), 16)
        
        # Draw snake
        for segment in self.snake:
            pygame.draw.rect(surface, "green", (x_offset + segment.x, y_offset + segment.y, TILE_SIZE, TILE_SIZE))
        
        # Draw apple
        pygame.draw.rect(surface, "red", (x_offset + self.apple.x, y_offset + self.apple.y, TILE_SIZE, TILE_SIZE))
        
        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, "white")
        surface.blit(score_text, (x_offset + 20, y_offset + 20))

    def _move(self, action):
        # action
        clock_wise = [pygame.Vector2(TILE_SIZE, 0), pygame.Vector2(0, TILE_SIZE), pygame.Vector2(-TILE_SIZE, 0), pygame.Vector2(0, -TILE_SIZE)]
        idx = clock_wise.index(self.direction)

        if action == [1, 0, 0]:
            new_dir = clock_wise[idx]  # no change
        elif action == [0, 1, 0]:
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction = new_dir
        x = self.snake[0].x
        y = self.snake[0].y
        x += self.direction.x
        y += self.direction.y
        self.head = pygame.Vector2(x, y)

    def _place_food(self):
        x = random.randint(1, (self.w // TILE_SIZE) - 2) * TILE_SIZE
        y = random.randint(1, (self.h // TILE_SIZE) - 2) * TILE_SIZE
        self.apple = pygame.Vector2(x, y)
        if self.apple in self.snake:
            self._place_food()