import pygame
import random


#stuff
GAME_WIDTH = 256
GAME_HEIGHT = 256
TILE_SIZE = 16

#scale factor for the window size, since 256x256 is pretty small
SCALE = 3
WINDOW_WIDTH = GAME_WIDTH * SCALE
WINDOW_HEIGHT = GAME_HEIGHT * SCALE

#game initialization
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SCALED)
clock = pygame.time.Clock()

SNAKE = [pygame.Vector2(
    ( screen.get_width() //2 // TILE_SIZE ) * TILE_SIZE,
    ( screen.get_height() // 2 // TILE_SIZE ) * TILE_SIZE
    )]

def apple_spawn():

    pos= pygame.Vector2(
        random.randint(1, screen.get_width() // TILE_SIZE - 2) * TILE_SIZE,
        random.randint(1, screen.get_height() // TILE_SIZE - 2) * TILE_SIZE
    )
    if pos in SNAKE:
        return apple_spawn()
    return pos

MOV_VEC = pygame.Vector2(0,0)
POS_APPLE = apple_spawn()

running = True
game_over = False
move_counter = 0
MOVE_INTERVAL = 5  # Snake moves every 5 frames at 60 FPS (~12 moves/sec)

# main game loop
while running:

    # Handle all events once per frame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_r and game_over:
            MOV_VEC = pygame.Vector2(0,0)
            SNAKE = [pygame.Vector2(
                ( screen.get_width() //2 // TILE_SIZE ) * TILE_SIZE,
                ( screen.get_height() // 2 // TILE_SIZE ) * TILE_SIZE
            )]
            POS_APPLE = apple_spawn()
            move_counter = 0
            game_over = False

    #movement
    if not game_over:
        move_counter += 1

        #keyboard input
        keyboard = pygame.key.get_pressed()
        if keyboard[pygame.K_UP] and MOV_VEC.y != TILE_SIZE: 
            MOV_VEC.y = -TILE_SIZE
            MOV_VEC.x = 0
        elif keyboard[pygame.K_DOWN] and MOV_VEC.y != -TILE_SIZE: 
            MOV_VEC.y = TILE_SIZE
            MOV_VEC.x = 0
        elif keyboard[pygame.K_LEFT] and MOV_VEC.x != TILE_SIZE: 
            MOV_VEC.x = -TILE_SIZE
            MOV_VEC.y = 0
        elif keyboard[pygame.K_RIGHT] and MOV_VEC.x != -TILE_SIZE: 
            MOV_VEC.x = TILE_SIZE
            MOV_VEC.y = 0

        
        
        # ALL GAME LOGIC RUNS HERE AT 5 FPS
        if move_counter >= MOVE_INTERVAL:
            move_counter = 0
            previous_tail = SNAKE[-1].copy()

            if len(SNAKE) > 1:
                for i in range(len(SNAKE) - 1, 0, -1):
                    SNAKE[i] = SNAKE[i - 1].copy()

            SNAKE[0] = SNAKE[0] + MOV_VEC

            # Border collision check
            if SNAKE[0].y <= 0 or SNAKE[0].y >= screen.get_height() - 16 or SNAKE[0].x <= 0 or SNAKE[0].x >= screen.get_width() - 16:
                game_over = True

            # Self collision check
            if SNAKE[0] in SNAKE[1:]:
                game_over = True

            # Apple eating check
            if SNAKE[0] == POS_APPLE:
                POS_APPLE = apple_spawn()
                SNAKE.append(previous_tail)

        #background and border
        screen.fill("black")
        pygame.draw.rect(screen,"white", screen.get_rect(), 16)

        #score
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Score: {len(SNAKE) - 1}", True, "white")
        screen.blit(score_text, (20, 20))

        # Snake
        for segment in SNAKE:
            pygame.draw.rect(screen,"green",
                            (segment.x ,
                            segment.y ,
                            TILE_SIZE, TILE_SIZE 
                            ))

        # Apple
        pygame.draw.rect(screen,"red", (POS_APPLE.x, POS_APPLE.y, TILE_SIZE, TILE_SIZE))



    if game_over:

            # 4. DRAWING (Game Over Screen)
            screen.fill("black")
            font = pygame.font.SysFont(None, 48)
            text = font.render("Game Over! Press R to Restart", True, "white")
            text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
            screen.blit(text, text_rect)



    # flip() the display to put your work on screen
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
