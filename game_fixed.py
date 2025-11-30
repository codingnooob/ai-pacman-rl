import numpy as np
from enum import IntEnum
from PIL import Image

class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class PacmanGame:
    def __init__(self, custom_map_path=None, hunger_config=None):
        self.hunger_config = hunger_config or {}
        self.hunger_config_snapshot = dict(self.hunger_config)
        self.initial_pellet_total = 0
        self.initial_power_total = 0
        self.prev_pacman_pos = None
        self.pacman_velocity = (0, 0)
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
        self.last_release_events = []
        
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
        
        self.initial_pellet_total = len(self.pellets)
        self.initial_power_total = len(self.power_pellets)
        self.prev_pacman_pos = tuple(self.pacman_pos)
        self.pacman_velocity = (0, 0)
        
        self.score = 0
        self.done = False
        self.steps = 0
        self.termination_reason = 'RUNNING'
        self.steps_since_progress = 0
        self.score_freeze_steps = 0
        self.hunger_meter = 0.0
        self.unique_tiles = 0
        self._last_score_snapshot = 0
        return self.get_state()
    
    def get_state(self):
        return {
            'pacman': tuple(self.pacman_pos),
            'ghosts': [tuple(g) for g in self.ghosts],
            'pacman_velocity': tuple(self.pacman_velocity),
            'pellets': self.pellets.copy(),
            'power_pellets': self.power_pellets.copy(),
            'walls': self.walls,
            'ghost_vulnerable': self.ghost_vulnerable.copy(),
            'score': self.score,
            'steps': self.steps,
            'dimensions': {'height': self.height, 'width': self.width},
            'initial_counts': {
                'pellets': self.initial_pellet_total,
                'power_pellets': self.initial_power_total
            },
            'termination_reason': getattr(self, 'termination_reason', 'RUNNING'),
            'hunger_stats': {
                'steps_since_progress': getattr(self, 'steps_since_progress', 0),
                'score_freeze_steps': getattr(self, 'score_freeze_steps', 0),
                'hunger_meter': getattr(self, 'hunger_meter', 0.0),
                'unique_tiles': getattr(self, 'unique_tiles', 0)
            },
            'hunger_config': self.hunger_config_snapshot.copy()
        }
    
    def step(self, pacman_action, ghost_actions):
        if self.done:
            return self.get_state(), 0, True
        
        self.steps += 1
        self.game_timer += 1
        
        # FIXED: Increased base reward and added survival penalty
        reward = -1.0  # Stronger time penalty to discourage excessive survival
        progress_made = False
        
        # Move Pacman
        prev_pos = tuple(self.pacman_pos)
        new_pos = self._move(self.pacman_pos, pacman_action)
        if self._is_valid(new_pos):
            self.pacman_pos = new_pos
        
        self.pacman_velocity = (
            self.pacman_pos[0] - prev_pos[0],
            self.pacman_pos[1] - prev_pos[1]
        )
        self.prev_pacman_pos = tuple(self.pacman_pos)
        pacman_tuple = tuple(self.pacman_pos)
        
        # Check pellet collection - STRONGER GOAL INCENTIVES
        if pacman_tuple in self.pellets:
            self.pellets.remove(pacman_tuple)
            self.score += 10
            reward = 50  # Increased from 10 to strongly incentivize goal completion
            self.pellets_eaten += 1
            progress_made = True
        
        # Check power pellet - ENHANCED POWER PELLET REWARD
        if pacman_tuple in self.power_pellets:
            self.power_pellets.remove(pacman_tuple)
            self.score += 50
            reward = 200  # Increased from 50
            self.ghost_vulnerable = [True] * 4
            self.vulnerable_timer = self.vulnerable_duration
            self.ghost_eaten_count = 0
            self.pellets_eaten += 1
            progress_made = True
        
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
                    # Eat ghost - ENHANCED GHOST EATING REWARD
                    ghost_points = 400 * (2 ** self.ghost_eaten_count)  # Increased from 200
                    self.score += ghost_points
                    reward = ghost_points
                    self.ghost_eaten_count += 1
                    self.ghost_vulnerable[i] = False
                    progress_made = True
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
                        self.termination_reason = 'GHOST_COLLISION'
                        reward = -1000  # Increased penalty for losing
                        return self.get_state(), reward, True
        
        # Win condition - ENHANCED WIN REWARD
        if len(self.pellets) == 0 and len(self.power_pellets) == 0:
            self.done = True
            self.termination_reason = 'PACMAN_WIN'
            reward = 2000  # Doubled win reward to strongly incentivize completion
            progress_made = True
            
            # Add efficiency bonus for faster completion
            if self.steps < 1000:
                efficiency_bonus = (1000 - self.steps) * 0.1  # Small bonus for speed
                reward += efficiency_bonus
        
        self._update_internal_progress(progress_made)
        return self.get_state(), reward, self.done

        
    def _update_internal_progress(self, progress_made):
        if progress_made:
            self.steps_since_progress = 0
            self.score_freeze_steps = 0
        else:
            self.steps_since_progress += 1
            self.score_freeze_steps += 1
        self._last_score_snapshot = self.score


    def update_hunger_stats(self, stats):
        self.steps_since_progress = stats.get('steps_since_progress', getattr(self, 'steps_since_progress', 0))
        self.score_freeze_steps = stats.get('score_freeze_steps', getattr(self, 'score_freeze_steps', 0))
        self.hunger_meter = stats.get('hunger_meter', getattr(self, 'hunger_meter', 0.0))
        self.unique_tiles = stats.get('unique_tiles', getattr(self, 'unique_tiles', 0))

    def force_hunger_termination(self, reason='HUNGER'):
        """Forcefully end the episode due to hunger-related violations.

        Mirrors the signature of ``step`` so callers can treat this as a
        terminal transition without duplicating logic elsewhere.
        """
        self.done = True
        self.termination_reason = reason or 'HUNGER'

        reward = None
        if hasattr(self, 'hunger_config'):
            reward = self.hunger_config.get('hunger_termination_reward')
        if reward is None and hasattr(self, 'hunger_config_snapshot'):
            reward = self.hunger_config_snapshot.get('hunger_termination_reward')
        if reward is None:
            reward = -750.0  # Fallback to default trainer penalty

        # ``self.score`` tracks arcade points, so we do not mutate it here to
        # avoid double-counting negative penalties. The trainer already applies
        # the hunger termination reward to the learning signal.
        state = self.get_state()
        return state, reward, self.done
    
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
    
    def _release_ghost_from_house(self, ghost_index, center_x, center_y, reason):
        if self.ghost_released[ghost_index]:
            return False
        self.ghost_released[ghost_index] = True
        self.ghost_in_house[ghost_index] = False
        self.ghosts[ghost_index] = [center_y - 4, center_x]
        self.last_release_events.append({
            'ghost': ghost_index,
            'reason': reason,
            'step': self.game_timer,
        })
        return True

    def _update_ghost_release(self):
        """Release ghosts based on pellets eaten or timer"""
        center_x, center_y = self.width // 2, self.height // 2
        self.last_release_events = []
        
        # Pinky: immediate or 5 seconds (unchanged)
        if not self.ghost_released[1] and (self.pellets_eaten >= 0 or self.game_timer >= 150):
            self._release_ghost_from_house(1, center_x, center_y, 'pinky_threshold')
        
        # Inky: FIXED - 5 pellets or ~1.7 seconds (reduced from 30/300)
        if not self.ghost_released[2] and (self.pellets_eaten >= 5 or self.game_timer >= 50):
            self._release_ghost_from_house(2, center_x, center_y, 'inky_threshold')
        
        # Clyde: FIXED - 10 pellets or ~3.3 seconds (reduced from 60/450)
        if not self.ghost_released[3] and (self.pellets_eaten >= 10 or self.game_timer >= 100):
            self._release_ghost_from_house(3, center_x, center_y, 'clyde_threshold')
        
        if self.game_timer >= 75:
            for idx in range(1, 4):
                self._release_ghost_from_house(idx, center_x, center_y, 'force_timer')
    
    def _is_valid(self, pos):
        return tuple(pos) not in self.walls