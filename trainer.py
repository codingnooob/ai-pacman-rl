from game import PacmanGame
from agent import PacmanAgent, GhostTeam
import numpy as np
from multiprocessing import Pool, cpu_count
import json
import os
from datetime import datetime

class Trainer:
    def __init__(self, custom_map_path=None, n_envs=None):
        # Determine number of parallel environments
        if n_envs is None:
            n_envs = min(cpu_count(), 8)  # Use up to 8 cores
        self.n_envs = n_envs
        
        # Create parallel game instances
        self.games = [PacmanGame(custom_map_path) for _ in range(n_envs)]
        
        # Shared agents across all environments
        self.pacman_agent = PacmanAgent()
        self.ghost_team = GhostTeam()
        
        self.episode = 0
        self.total_rewards = [0] * n_envs
        self.episode_rewards = []
        self.running = False
        
        # Track most recently completed environment
        self.last_completed_env = 0
        
        # Track previous positions for each environment
        self.prev_states = [None] * n_envs
        self.prev_ghost_actions = [None] * n_envs
        self.prev_pacman_pos = [None] * n_envs
        self.prev_ghost_positions = [[None] * 4 for _ in range(n_envs)]
        
        # Enhanced progress tracking
        self.pacman_wins = 0
        self.ghost_wins = 0
        self.episode_lengths = []
        self.pellets_collected = []
        self.best_reward = float('-inf')
        
        # Create checkpoints directory
        os.makedirs('checkpoints', exist_ok=True)
        
    def train_step(self):
        """Execute one step across all parallel environments"""
        done_any = False
        
        for env_id in range(self.n_envs):
            game = self.games[env_id]
            state = game.get_state()
            
            # Get Pacman action
            pacman_state = self.pacman_agent.get_state_repr(state)
            pacman_action = self.pacman_agent.get_action(pacman_state)
            
            # Get coordinated ghost actions
            ghost_actions = self.ghost_team.get_actions(state)
            
            # Execute step
            next_state, reward, done = game.step(pacman_action, ghost_actions)
            
            # Reward shaping
            # Penalize Pacman for oscillating
            if self.prev_pacman_pos[env_id] is not None:
                if tuple(next_state['pacman']) == self.prev_pacman_pos[env_id]:
                    reward -= 0.5  # Penalty for not moving
            
            # Reward ghosts for getting closer to Pacman
            ghost_reward_bonus = 0
            for i, ghost_pos in enumerate(next_state['ghosts']):
                # Penalty for staying in house when released
                center_x, center_y = game.width // 2, game.height // 2
                in_house = (center_y - 2 <= ghost_pos[0] <= center_y + 2 and 
                           center_x - 3 <= ghost_pos[1] <= center_x + 3)
                if in_house and game.ghost_released[i] and not game.ghost_in_house[i]:
                    ghost_reward_bonus -= 1.0  # Strong penalty for lingering in house
                
                # Penalty for not moving (stalling)
                if self.prev_ghost_positions[env_id][i] is not None:
                    if tuple(ghost_pos) == self.prev_ghost_positions[env_id][i]:
                        ghost_reward_bonus -= 0.3  # Penalty for ghost not moving
                    
                    # Reward for getting closer to Pacman (only if not vulnerable)
                    if not next_state['ghost_vulnerable'][i]:
                        prev_dist = abs(state['pacman'][0] - self.prev_ghost_positions[env_id][i][0]) + \
                                   abs(state['pacman'][1] - self.prev_ghost_positions[env_id][i][1])
                        curr_dist = abs(next_state['pacman'][0] - ghost_pos[0]) + \
                                   abs(next_state['pacman'][1] - ghost_pos[1])
                        if curr_dist < prev_dist:
                            ghost_reward_bonus += 0.1  # Small reward for getting closer
            
            self.total_rewards[env_id] += reward
            
            # Update Pacman
            next_pacman_state = self.pacman_agent.get_state_repr(next_state)
            self.pacman_agent.update(pacman_state, pacman_action, reward, next_pacman_state, done)
            
            # Update ghost team
            ghost_reward = -reward if reward != 0 else -0.1
            ghost_reward += ghost_reward_bonus  # Add proximity bonus
            if self.prev_states[env_id] is not None:
                self.ghost_team.update(
                    self.prev_states[env_id], 
                    self.prev_ghost_actions[env_id], 
                    ghost_reward, 
                    state, 
                    done
                )
            
            # Track positions
            self.prev_states[env_id] = state
            self.prev_ghost_actions[env_id] = ghost_actions
            self.prev_pacman_pos[env_id] = tuple(state['pacman'])
            self.prev_ghost_positions[env_id] = [tuple(g) for g in state['ghosts']]
            
            # Handle episode completion
            if done:
                episode_reward = self.total_rewards[env_id]
                self.episode_rewards.append(episode_reward)
                
                # Track this as the most recently completed environment
                self.last_completed_env = env_id
                
                # Track win/loss
                if reward == 1000:  # Pacman won
                    self.pacman_wins += 1
                elif reward == -500:  # Ghost won
                    self.ghost_wins += 1
                
                # Track episode length and pellets
                self.episode_lengths.append(game.steps)
                initial_pellets = len(game.power_pellet_positions) + 244  # Approximate
                remaining_pellets = len(next_state['pellets']) + len(next_state['power_pellets'])
                self.pellets_collected.append(initial_pellets - remaining_pellets)
                
                self.episode += 1
                self.total_rewards[env_id] = 0
                game.reset()
                self.prev_states[env_id] = None
                self.prev_ghost_actions[env_id] = None
                self.prev_pacman_pos[env_id] = None
                self.prev_ghost_positions[env_id] = [None] * 4
                done_any = True
                
                # Save best model
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                    self.save_checkpoint('best')
        
        return done_any
    
    def get_stats(self):
        avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
        avg_length = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0
        avg_pellets = np.mean(self.pellets_collected[-100:]) if self.pellets_collected else 0
        
        total_games = self.pacman_wins + self.ghost_wins
        pacman_win_rate = (self.pacman_wins / total_games * 100) if total_games > 0 else 0
        ghost_win_rate = (self.ghost_wins / total_games * 100) if total_games > 0 else 0
        
        # Get gradient statistics
        pacman_grad_stats = self.pacman_agent.agent.get_grad_stats()
        ghost_grad_stats = self.ghost_team.qmix.get_grad_stats()
        
        return {
            'episode': self.episode,
            'avg_reward': avg_reward,
            'last_reward': self.episode_rewards[-1] if self.episode_rewards else 0,
            'n_envs': self.n_envs,
            'pacman_win_rate': pacman_win_rate,
            'ghost_win_rate': ghost_win_rate,
            'avg_length': avg_length,
            'avg_pellets': avg_pellets,
            'best_reward': self.best_reward,
            'pacman_grad_norm': pacman_grad_stats['mean'],
            'ghost_grad_norm': ghost_grad_stats['mean'],
            'pacman_lr': self.pacman_agent.agent.optimizer.param_groups[0]['lr'],
            'ghost_lr': self.ghost_team.qmix.optimizer.param_groups[0]['lr'],
            'entropy_coef': self.pacman_agent.agent.entropy_coef,
            'ghost_epsilon': self.ghost_team.qmix.epsilon
        }
    
    def save_checkpoint(self, name='checkpoint'):
        """Save model checkpoint"""
        import torch
        checkpoint = {
            'episode': self.episode,
            'pacman_policy': self.pacman_agent.agent.policy.state_dict(),
            'ghost_q_networks': [net.state_dict() for net in self.ghost_team.qmix.q_networks],
            'ghost_mixer': self.ghost_team.qmix.mixer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'pacman_wins': self.pacman_wins,
            'ghost_wins': self.ghost_wins,
            'best_reward': self.best_reward
        }
        torch.save(checkpoint, f'checkpoints/{name}.pt')
        
        # Also save training stats as JSON
        stats = {
            'episode': self.episode,
            'episode_rewards': self.episode_rewards[-1000:],  # Last 1000
            'episode_lengths': self.episode_lengths[-1000:],
            'pellets_collected': self.pellets_collected[-1000:],
            'pacman_wins': self.pacman_wins,
            'ghost_wins': self.ghost_wins,
            'timestamp': datetime.now().isoformat()
        }
        with open(f'checkpoints/{name}_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
    
    def load_checkpoint(self, name='checkpoint'):
        """Load model checkpoint"""
        import torch
        checkpoint = torch.load(f'checkpoints/{name}.pt')
        
        self.episode = checkpoint['episode']
        self.pacman_agent.agent.policy.load_state_dict(checkpoint['pacman_policy'])
        self.pacman_agent.agent.policy_old.load_state_dict(checkpoint['pacman_policy'])
        
        for i, state_dict in enumerate(checkpoint['ghost_q_networks']):
            self.ghost_team.qmix.q_networks[i].load_state_dict(state_dict)
            self.ghost_team.qmix.target_q_networks[i].load_state_dict(state_dict)
        
        self.ghost_team.qmix.mixer.load_state_dict(checkpoint['ghost_mixer'])
        self.ghost_team.qmix.target_mixer.load_state_dict(checkpoint['ghost_mixer'])
        
        self.episode_rewards = checkpoint['episode_rewards']
        self.pacman_wins = checkpoint['pacman_wins']
        self.ghost_wins = checkpoint['ghost_wins']
        self.best_reward = checkpoint['best_reward']

import numpy as np
