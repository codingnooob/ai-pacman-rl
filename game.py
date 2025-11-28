import numpy as np
from enum import IntEnum
from PIL import Image

class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class PacmanGame:
    def __init__(self, custom_map_path=None):
        if custom_map_path:
            self._load_map_from_image(custom_map_path)
        else:
            self.width = 28
            self.height = 31
            self._create_classic_maze()
        self.reset()
    
    def _load_map_from_image(self, image_path):
        """
        Load map from image where:
        - Black (0,0,0): Wall
        - White (255,255,255): Empty space (pellet)
        - Yellow (255,255,0): Pacman start
        - Red (255,0,0): Ghost 1 start
        - Pink (255,192,203): Ghost 2 start
        - Cyan (0,255,255): Ghost 3 start
        - Orange (255,165,0): Ghost 4 start
        - Blue (0,0,255): Power pellet
        """
        img = Image.open(image_path).convert('RGB')
        self.width = img.width
        self.height = img.height
        
        pixels = img.load()
        self.walls = set()
        self.power_pellet_positions = set()
        self.pacman_start = [self.height // 2, self.width // 2]
        self.ghost_starts = []
        
        for i in range(self.height):
            for j in range(self.width):
                r, g, b = pixels[j, i]
                
                if (r, g, b) == (0, 0, 0):  # Black - wall
                    self.walls.add((i, j))
                elif (r, g, b) == (255, 255, 0):  # Yellow - Pacman
                    self.pacman_start = [i, j]
                elif (r, g, b) == (255, 0, 0):  # Red - Ghost 1
                    self.ghost_starts.append([i, j])
                elif (r, g, b) == (255, 192, 203):  # Pink - Ghost 2
                    self.ghost_starts.append([i, j])
                elif (r, g, b) == (0, 255, 255):  # Cyan - Ghost 3
                    self.ghost_starts.append([i, j])
                elif (r, g, b) == (255, 165, 0):  # Orange - Ghost 4
                    self.ghost_starts.append([i, j])
                elif (r, g, b) == (0, 0, 255):  # Blue - Power pellet
                    self.power_pellet_positions.add((i, j))
        
        # Ensure we have 4 ghosts
        while len(self.ghost_starts) < 4:
            self.ghost_starts.append([self.height // 2, self.width // 2])
        self.ghost_starts = self.ghost_starts[:4]
        
    def _create_classic_maze(self):
        # Classic Pacman maze pattern
        maze = [
            "############################",
            "#............##............#",
            "#.####.#####.##.#####.####.#",
            "#o####.#####.##.#####.####o#",
            "#.####.#####.##.#####.####.#",
            "#..........................#",
            "#.####.##.########.##.####.#",
            "#.####.##.########.##.####.#",
            "#......##....##....##......#",
            "######.##### ## #####.######",
            "######.##### ## #####.######",
            "######.##          ##.######",
            "######.## ###--### ##.######",
            "######.## #      # ##.######",
            "      .   #      #   .      ",
            "######.## #      # ##.######",
            "######.## ######## ##.######",
            "######.##          ##.######",
            "######.## ######## ##.######",
            "######.## ######## ##.######",
            "#............##............#",
            "#.####.#####.##.#####.####.#",
            "#.####.#####.##.#####.####.#",
            "#o..##.......  .......##..o#",
            "###.##.##.########.##.##.###",
            "###.##.##.########.##.##.###",
            "#......##....##....##......#",
            "#.##########.##.##########.#",
            "#.##########.##.##########.#",
            "#..........................#",
            "############################"
        ]
        
        self.walls = set()
        self.power_pellet_positions = set()
        
        for i, row in enumerate(maze):
            for j, cell in enumerate(row):
                if cell == '#':
                    self.walls.add((i, j))
                elif cell == 'o':
                    self.power_pellet_positions.add((i, j))
    
    def reset(self):
        self.pacman_pos = self.pacman_start.copy() if hasattr(self, 'pacman_start') else [23, 14]
        self.ghosts = [g.copy() for g in self.ghost_starts] if hasattr(self, 'ghost_starts') else [[11, 13], [11, 14], [13, 13], [13, 14]]
        self.ghost_vulnerable = [False] * 4
        self.vulnerable_timer = 0
        self.vulnerable_duration = 180  # ~6 seconds at 30 fps
        self.ghost_eaten_count = 0
        
        # Ghost release mechanism
        self.ghost_released = [True, False, False, False]  # Blinky starts out, others in house
        self.ghost_in_house = [False, True, True, True]
        self.pellets_eaten = 0
        self.game_timer = 0
        
        # Place Blinky outside, others inside ghost house
        center_x, center_y = self.width // 2, self.height // 2
        self.ghosts[0] = [14, center_x]  # Blinky outside (row 14 is clear)
        self.ghosts[1] = [14, center_x - 1]  # Pinky in house
        self.ghosts[2] = [14, center_x]      # Inky in house
        self.ghosts[3] = [14, center_x + 1]  # Clyde in house
        
        # Place pellets
        self.pellets = set()
        self.power_pellets = self.power_pellet_positions.copy()
        
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) not in self.walls and (i, j) not in self.power_pellet_positions:
                    # Skip ghost house area
                    if not (11 <= i <= 16 and 11 <= j <= 16):
                        self.pellets.add((i, j))
        
        self.pellets.discard(tuple(self.pacman_pos))
        for g in self.ghosts:
            self.pellets.discard(tuple(g))
        
        self.score = 0
        self.done = False
        self.steps = 0
        return self.get_state()
    
    def get_state(self):
        return {
            'pacman': tuple(self.pacman_pos),
            'ghosts': [tuple(g) for g in self.ghosts],
            'pellets': self.pellets.copy(),
            'power_pellets': self.power_pellets.copy(),
            'walls': self.walls,
            'ghost_vulnerable': self.ghost_vulnerable.copy(),
            'score': self.score
        }
    
    def step(self, pacman_action, ghost_actions):
        if self.done:
            return self.get_state(), 0, True
        
        self.steps += 1
        self.game_timer += 1
        reward = -0.1  # Small time penalty
        
        # Move Pacman
        new_pos = self._move(self.pacman_pos, pacman_action)
        if self._is_valid(new_pos):
            self.pacman_pos = new_pos
        
        pacman_tuple = tuple(self.pacman_pos)
        
        # Check pellet collection
        if pacman_tuple in self.pellets:
            self.pellets.remove(pacman_tuple)
            self.score += 10
            reward = 10
            self.pellets_eaten += 1
        
        # Check power pellet
        if pacman_tuple in self.power_pellets:
            self.power_pellets.remove(pacman_tuple)
            self.score += 50
            reward = 50
            self.ghost_vulnerable = [True] * 4
            self.vulnerable_timer = self.vulnerable_duration
            self.ghost_eaten_count = 0
            self.pellets_eaten += 1
        
        # Ghost release mechanism
        self._update_ghost_release()
        
        # Update vulnerable timer
        if self.vulnerable_timer > 0:
            self.vulnerable_timer -= 1
            if self.vulnerable_timer == 0:
                self.ghost_vulnerable = [False] * 4
        
        # Move ghosts (only if released and not in house)
        for i, action in enumerate(ghost_actions):
            if not self.ghost_released[i] or self.ghost_in_house[i]:
                continue  # Ghost not released yet or in house
            
            if not self.ghost_vulnerable[i]:
                new_pos = self._move(self.ghosts[i], action)
                if self._is_valid(new_pos):
                    self.ghosts[i] = new_pos
            else:
                # Vulnerable ghosts move slower/randomly
                if self.steps % 2 == 0:
                    new_pos = self._move(self.ghosts[i], action)
                    if self._is_valid(new_pos):
                        self.ghosts[i] = new_pos
        
        # Check collisions
        for i, ghost in enumerate(self.ghosts):
            if tuple(ghost) == pacman_tuple:
                if self.ghost_vulnerable[i]:
                    # Eat ghost
                    ghost_points = 200 * (2 ** self.ghost_eaten_count)
                    self.score += ghost_points
                    reward = ghost_points
                    self.ghost_eaten_count += 1
                    self.ghost_vulnerable[i] = False
                    # Return ghost to house
                    center_x, center_y = self.width // 2, self.height // 2
                    self.ghosts[i] = [center_y, center_x]
                    self.ghost_in_house[i] = True
                    # Ghost will exit immediately (no waiting)
                    self.ghosts[i] = [center_y - 4, center_x]
                    self.ghost_in_house[i] = False
                else:
                    # Game over (only if ghost is released)
                    if self.ghost_released[i]:
                        self.done = True
                        reward = -500
                        return self.get_state(), reward, True
        
        # Win condition
        if len(self.pellets) == 0 and len(self.power_pellets) == 0:
            self.done = True
            reward = 1000
        
        # Timeout
        if self.steps > 5000:
            self.done = True
        
        return self.get_state(), reward, self.done
    
    def _move(self, pos, action):
        new_pos = pos.copy()
        if action == Action.UP:
            new_pos[0] -= 1
        elif action == Action.DOWN:
            new_pos[0] += 1
        elif action == Action.LEFT:
            new_pos[1] -= 1
        elif action == Action.RIGHT:
            new_pos[1] += 1
        
        # Tunnel wrapping
        if new_pos[1] < 0:
            new_pos[1] = self.width - 1
        elif new_pos[1] >= self.width:
            new_pos[1] = 0
            
        return new_pos
    
    def _update_ghost_release(self):
        """Release ghosts based on pellets eaten or timer"""
        center_x, center_y = self.width // 2, self.height // 2
        
        # Pinky: immediate or 5 seconds
        if not self.ghost_released[1] and (self.pellets_eaten >= 0 or self.game_timer >= 150):
            self.ghost_released[1] = True
            self.ghost_in_house[1] = False
            self.ghosts[1] = [center_y - 4, center_x]  # Exit position
        
        # Inky: 30 pellets or 10 seconds
        if not self.ghost_released[2] and (self.pellets_eaten >= 30 or self.game_timer >= 300):
            self.ghost_released[2] = True
            self.ghost_in_house[2] = False
            self.ghosts[2] = [center_y - 4, center_x]  # Exit position
        
        # Clyde: 60 pellets or 15 seconds
        if not self.ghost_released[3] and (self.pellets_eaten >= 60 or self.game_timer >= 450):
            self.ghost_released[3] = True
            self.ghost_in_house[3] = False
            self.ghosts[3] = [center_y - 4, center_x]  # Exit position
    
    def _is_valid(self, pos):
        return tuple(pos) not in self.walls
