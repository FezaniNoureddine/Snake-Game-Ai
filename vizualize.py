import argparse
import pygame
import torch
import numpy as np
import os
import sys
# Importing your specific file names
from brain import Linear_QNet
from gameai import SnakeGameAI, TILE_SIZE
from agent import Agent

# Exact Window Specifications (1550 x 768)
WIDTH, HEIGHT = 1550, 768
GAME_SIZE = 768
BORDER_WIDTH = 14
FPS = 20 

class BrainVisualizer:
    def __init__(self, model_path='MODELS/model_game_10.pth', max_games=0):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake AI Multiverse - Visualization Mode")
        self.model_path = model_path
        self.max_games = max_games
        self.games_played = 0
        
        # Fonts
        self.font_small = pygame.font.SysFont('Consolas', 12, bold=True)
        self.font_label = pygame.font.SysFont('Arial', 14, bold=True)
        self.font_large = pygame.font.SysFont('Arial', 32, bold=True)
        
        # Initialize Game and Agent
        self.game = SnakeGameAI(headless=True)
        self.agent = Agent()
        
        # Override Agent saving methods to do NOTHING (Safety First)
        self.agent.model.save = lambda x=None: print("Save blocked: Visualization Mode")
        self.agent.remember = lambda *args: None # Don't fill memory
        self.agent.train_short_memory = lambda *args: None # Don't train
        self.agent.train_long_memory = lambda: None # Don't train
        
        # Load the specific model you want to see
        if os.path.exists(model_path):
            self.agent.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.agent.model.eval()
            print(f"Visualization active for: {model_path}")
        else:
            print(f"ERROR: Model {model_path} not found!")
            sys.exit()

    def val_to_color(self, val):
        """Maps neuron value to color: Low=Red, High=Blue[cite: 8]"""
        val = np.clip(val, 0, 1)
        r = int(255 * (1 - val))
        b = int(255 * val)
        return (r, 30, b)

    def draw_brain_panel(self, inputs, hidden, outputs):
        panel_x = GAME_SIZE + BORDER_WIDTH
        
        input_labels = [
            "DANGER_S", "DANGER_R", "DANGER_L",
            "DIR_L", "DIR_R", "DIR_U", "DIR_D",
            "FOOD_L", "FOOD_R", "FOOD_U", "FOOD_D"
        ]

        # Store positions for connection drawing
        input_positions = []
        hidden_circle_positions = []
        output_positions = []

        # Get weights for connection visualization
        with torch.no_grad():
            weights_input_hidden = self.agent.model.linear1.weight.data  # (256, 11)
            weights_hidden_output = self.agent.model.linear2.weight.data  # (3, 256)
            max_weight_ih = max(abs(weights_input_hidden).max().item(), 0.1)
            max_weight_ho = max(abs(weights_hidden_output).max().item(), 0.1)

        # 1. Input Layer (Left) - Circles
        font_input_label = pygame.font.SysFont('Arial', 11, bold=True)
        font_input_val = pygame.font.SysFont('Consolas', 10, bold=True)
        for i, val in enumerate(inputs[0]):
            y_pos = 50 + (i * 50)
            x_pos = panel_x + 80
            input_positions.append((x_pos, y_pos, val))
            
            # Label and value
            lbl = font_input_label.render(input_labels[i], True, (180, 180, 180))
            val_txt = font_input_val.render(str(int(val)), True, (255, 255, 255))
            self.screen.blit(lbl, (panel_x + 15, y_pos - 7))
            self.screen.blit(val_txt, (panel_x + 75, y_pos - 5))

        # 2. Hidden Layer (Middle) - Show top 9 neurons stacked vertically
        h_norm = (hidden - hidden.min()) / (hidden.max() - hidden.min() + 1e-5)
        
        # Get top 9 most active hidden neurons
        top_9_indices = np.argsort(h_norm[0])[-9:][::-1]
        
        # Stack them vertically - single column with better spacing
        hidden_grid_x = panel_x + 330
        hidden_grid_y = 120
        hidden_grid_spacing = 50
        
        for grid_idx, h_idx in enumerate(top_9_indices):
            y_pos = hidden_grid_y + (grid_idx * hidden_grid_spacing)
            x_pos = hidden_grid_x
            val = h_norm[0][h_idx]
            hidden_circle_positions.append((x_pos, y_pos, h_idx, val))

        # 3. Output Layer (Right) - Circles
        decisions = ["LEFT", "CONTINUE", "RIGHT"]
        output_order = [2, 0, 1]  # Map to original output indices
        best_move = np.argmax(outputs[0])
        
        output_x = panel_x + 580
        output_y_positions = [120, 280, 440]
        
        for display_idx, orig_idx in enumerate(output_order):
            y_pos = output_y_positions[display_idx]
            val = outputs[0][orig_idx]
            is_chosen = orig_idx == best_move
            output_positions.append((output_x, y_pos, val, is_chosen, display_idx, orig_idx))

        # 4. Draw connections from INPUT to HIDDEN (FIRST, behind circles)
        for i_x, i_y, i_val in input_positions:
            for h_x, h_y, h_idx, h_val in hidden_circle_positions:
                # Average weights for this hidden neuron across all inputs
                weight = weights_input_hidden[h_idx, :].mean().item()
                weight_norm = abs(weight) / max_weight_ih
                weight_norm = np.clip(weight_norm, 0, 1)
                
                opacity = int(255 * weight_norm)
                if weight > 0:
                    line_color = (100 + 100 * weight_norm, 150, 200)  # Blue-ish
                else:
                    line_color = (200, 100, 100)  # Red-ish
                
                thickness = max(1, int(weight_norm * 2))
                
                if opacity > 30:
                    pygame.draw.line(self.screen, tuple(int(c * (opacity / 255)) for c in line_color), 
                                   (i_x, i_y), (h_x, h_y), thickness)

        # 5. Draw connections from HIDDEN to OUTPUT (FIRST, behind circles)
        for h_x, h_y, h_idx, h_val in hidden_circle_positions:
            for o_x, o_y, o_val, is_chosen, display_idx, orig_idx in output_positions:
                weight = weights_hidden_output[orig_idx, h_idx].item()
                weight_norm = abs(weight) / max_weight_ho
                weight_norm = np.clip(weight_norm, 0, 1)
                
                opacity = int(255 * weight_norm)
                if weight > 0:
                    line_color = (150, 200, 100)  # Green-ish
                else:
                    line_color = (200, 100, 150)  # Magenta-ish
                
                thickness = max(1, int(weight_norm * 2))
                
                if opacity > 30:
                    pygame.draw.line(self.screen, tuple(int(c * (opacity / 255)) for c in line_color), 
                                   (h_x, h_y), (o_x, o_y), thickness)

        # 6. Draw Input neurons (circles) ON TOP of lines
        for i_x, i_y, i_val in input_positions:
            color = self.val_to_color(i_val)
            pygame.draw.circle(self.screen, color, (i_x, i_y), 12)

        # 7. Draw Hidden neurons (circles) ON TOP of lines
        for h_x, h_y, h_idx, h_val in hidden_circle_positions:
            color = self.val_to_color(h_val)
            pygame.draw.circle(self.screen, color, (h_x, h_y), 10)

        # 8. Draw Output neurons (circles) ON TOP of lines - Always filled
        for o_x, o_y, o_val, is_chosen, display_idx, orig_idx in output_positions:
            if is_chosen:
                color = (0, 200, 255)  # Bright cyan when active
                radius = 28
                pygame.draw.circle(self.screen, color, (o_x, o_y), radius)  # Filled
                pygame.draw.circle(self.screen, (255, 255, 255), (o_x, o_y), radius, 2)  # White outline
            else:
                color = (60, 20, 20)  # Dark red when inactive - use as fill
                radius = 28
                pygame.draw.circle(self.screen, color, (o_x, o_y), radius)  # Filled with contour color
                pygame.draw.circle(self.screen, (120, 60, 60), (o_x, o_y), radius, 1)  # Lighter outline
            
            # Centered text
            txt_color = (255, 255, 255)
            font_output = pygame.font.SysFont('Arial', 11, bold=True)
            txt = font_output.render(decisions[display_idx], True, txt_color)
            txt_rect = txt.get_rect(center=(o_x, o_y))
            self.screen.blit(txt, txt_rect)

        # 9. Hidden Layer Visualization (Bottom) - Separate, square grid
        grid_x, grid_y = panel_x + 140, 620
        
        # Show top 64 neurons (most important)
        num_neurons_to_show = min(64, len(h_norm[0]))
        rows, cols = 8, 8
        cell_size = 12
        
        # Draw hidden layer squares
        for i in range(num_neurons_to_show):
            row, col = i // cols, i % cols
            color = self.val_to_color(h_norm[0][i])
            pygame.draw.rect(self.screen, color, (grid_x + (col * cell_size), grid_y + (row * cell_size), cell_size - 1, cell_size - 1))

        # Label for hidden layer
        font_label = pygame.font.SysFont('Arial', 12, bold=True)
        hidden_label = font_label.render("Hidden Layer (64/256)", True, (200, 200, 200))
        self.screen.blit(hidden_label, (grid_x, grid_y - 20))

    def run(self):
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Get state and thinking data without saving anything[cite: 5, 6]
            state = self.agent.get_state(self.game)
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            inputs, hidden, outputs = self.agent.model.get_activations(state_tensor)
            
            # Action
            move = [0, 0, 0]
            move[np.argmax(outputs)] = 1
            
            # Step Game
            _, done, _ = self.game.play_step(move)
            
            # Clear screen with better background color
            self.screen.fill((15, 25, 45))
            
            # Render game board on the left
            self.game.render_game(self.screen, x_offset=0, y_offset=0)
            
            # Render brain panel on the right
            self.draw_brain_panel(inputs, hidden, outputs)
            
            # Draw Divider
            pygame.draw.rect(self.screen, (255, 255, 255), (GAME_SIZE, 0, BORDER_WIDTH, HEIGHT))
            
            status_text = self.font_small.render(
                f"Model: {os.path.basename(self.model_path)}   Games: {self.games_played}", True, (255, 255, 255)
            )
            self.screen.blit(status_text, (20, HEIGHT - 30))
            
            pygame.display.flip()
            
            if done:
                self.games_played += 1
                if self.max_games > 0 and self.games_played >= self.max_games:
                    pygame.quit()
                    return
                self.game.reset()
            clock.tick(FPS)


def find_models(models_dir='MODELS'):
    if not os.path.isdir(models_dir):
        return []
    files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    files.sort(key=lambda x: (len(x), x))
    return [os.path.join(models_dir, f) for f in files]


def choose_model(models_dir='MODELS'):
    models = find_models(models_dir)
    if not models:
        print(f"No .pth models found in {models_dir}")
        sys.exit(1)

    print("Available models:")
    for idx, path in enumerate(models, 1):
        print(f"  {idx}. {os.path.basename(path)}")

    choice = input("Select a model number or filename: ").strip()
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            return models[idx]
    else:
        full_path = os.path.join(models_dir, choice)
        if os.path.exists(full_path):
            return full_path
        for path in models:
            if os.path.basename(path) == choice:
                return path

    print("Invalid selection. Using the first available model.")
    return models[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snake AI brain visualization")
    parser.add_argument("--model", "-m", help="Path or name of model file in MODELS/ to visualize")
    parser.add_argument("--games", "-g", type=int, default=0, help="Auto-close after this many completed games")
    args = parser.parse_args()

    model_path = args.model
    if model_path:
        if not os.path.exists(model_path):
            alt = os.path.join('MODELS', model_path)
            if os.path.exists(alt):
                model_path = alt
            else:
                print(f"Model '{model_path}' not found. Selecting from MODELS/ instead.")
                model_path = choose_model('MODELS')
    else:
        model_path = choose_model('MODELS')

    vis = BrainVisualizer(model_path=model_path, max_games=args.games)
    vis.run()