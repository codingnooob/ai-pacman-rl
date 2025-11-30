from game import PacmanGame
# Use the fixed game with improved reward structure
import sys
import os
import logging
if os.path.exists('game_fixed.py'):
    from game_fixed import PacmanGame as FixedPacmanGame
    PacmanGame = FixedPacmanGame  # Use fixed version if available
from agent import PacmanAgent, GhostTeam
import numpy as np
from multiprocessing import Pool, cpu_count
import json
import os
import time
from datetime import datetime
from collections import deque
from telemetry import EpisodeTelemetry, TelemetryCollector, TelemetryConfig

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, custom_map_path=None, n_envs=None, telemetry_config=None, telemetry_channel=None, hunger_overrides=None, perf_logging=False):
        # Determine number of parallel environments
        if n_envs is None:
            n_envs = min(cpu_count(), 8)  # Use up to 8 cores
        self.n_envs = n_envs
        
        # Hunger mechanic configuration (overridable via constructor in future tasks)
        self.hunger_config = {
            'hunger_idle_threshold': 18,
            'hunger_decay_rate': 0.12,
            'hunger_decay_growth': 1.015,
            'hunger_termination_limit': -220.0,
            'stagnation_tile_window': 64,
            'stagnation_tile_min': 20,
            'survival_grace_steps': 90,
            'hunger_termination_reward': -1000.0,
            'coverage_bonus_threshold': 0.6,
            'coverage_bonus_scale': 0.8,
            'idle_streak_penalty': -0.25,
            'idle_streak_cap': -2.0,
            'pellet_streak_bonus': 0.6,
        }
        if hunger_overrides:
            for key, value in hunger_overrides.items():
                if value is None:
                    continue
                self.hunger_config[key] = value

        # Ghost reward shaping configuration (centralized for QMIX team reward)
        self.ghost_reward_cfg = {
            'kappa1': 1.0,   # Pellet progress pressure multiplier
            'kappa2': 0.5,   # Idle ratio pressure multiplier
            'kappa3': 0.2,   # Survival time pressure multiplier
            'kappa4': 0.1,   # Proximity drive multiplier
            'kappa5': 0.3,   # Vulnerability tax multiplier
            'win_bonus': 50.0,
            'hunger_fail_penalty': 25.0,
            'distance_norm': None,  # Allow overrides; falls back to maze diagonal
        }
        self.survival_penalty_cfg = {
            'soft_threshold': 9,
            'hard_threshold': 16,
            'soft_penalty': -0.25,
            'hard_penalty': -0.9,
            'cooldown_steps': 6,
        }
        self.reward_clamp = (-1500.0, 2500.0)
        
        # Create parallel game instances with shared hunger config
        self.games = [PacmanGame(custom_map_path, hunger_config=self.hunger_config) for _ in range(n_envs)]
        self.maze_diagonal = self._infer_maze_diagonal()
        
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
        self.survival_cooldowns = [0] * n_envs
        self.pellet_streaks = [0] * n_envs
        self.axis_lock_state = [{'axis': None, 'streak': 0} for _ in range(n_envs)]
        
        # Enhanced progress tracking
        self.pacman_wins = 0
        self.ghost_wins = 0
        self.episode_lengths = []
        self.pellets_collected = []
        self.best_reward = float('-inf')
        
        # Diagnostic logging
        self.movement_penalty_count = 0
        self.ghost_movement_penalty_count = 0
        self.total_reward_components = []
        self.pacman_action_history = []
        self.ghost_action_history = []
        self.epsilon_history = []
        self.reward_breakdown_history = []
        self._perf_logging_default = bool(perf_logging)
        self._perf_logging = self._perf_logging_default
        
        # Hunger tracking scaffolding (per-environment)
        self.prev_pellet_counts = []
        self.initial_pellet_totals = []
        self.progress_trackers = []
        self.hunger_termination_counts = {
            'score_freeze': 0,
            'hunger_meter': 0
        }
        for env_id, game in enumerate(self.games):
            initial_state = game.get_state()
            pellet_count = len(initial_state['pellets']) + len(initial_state['power_pellets'])
            self.prev_pellet_counts.append(pellet_count)
            self.initial_pellet_totals.append(self._resolve_initial_pellet_total(initial_state, pellet_count))
            self.progress_trackers.append(self._create_progress_tracker(initial_state))
        
        # Create checkpoints directory
        os.makedirs('checkpoints', exist_ok=True)

        # Telemetry plumbing
        self.telemetry_config = telemetry_config or TelemetryConfig()
        self.telemetry_collector = TelemetryCollector(
            config=self.telemetry_config,
            channel=telemetry_channel,
        )
        self._telemetry_total_steps = 0
        self._telemetry_last_sample_steps = 0
        self._telemetry_last_sample_time = time.time()
        
    def _create_progress_tracker(self, state):
        tracker = {
            'steps_since_progress': 0,
            'score_freeze_steps': 0,
            'hunger_meter': 0.0,
            'unique_tile_window': deque(maxlen=self.hunger_config['stagnation_tile_window']),
            'last_score': state['score'],
            'last_pellet_count': len(state['pellets']) + len(state['power_pellets']),
            'last_position': tuple(state['pacman']),
            'idle_streak': 0,
            'coverage_ratio': 0.0,
            'survival_penalty_events': 0,
        }
        tracker['unique_tile_window'].append(tuple(state['pacman']))
        return tracker
    
    def _resolve_initial_pellet_total(self, state, fallback_count=None):
        """Derive the per-environment pellet baseline from the state snapshot."""

        counts = state.get('initial_counts') or {}
        pellets = counts.get('pellets')
        power = counts.get('power_pellets')
        if pellets is None:
            pellets = len(state.get('pellets', []))
        if power is None:
            power = len(state.get('power_pellets', []))
        try:
            total = int(pellets) + int(power)
        except (TypeError, ValueError):
            total = 0
        if total <= 0:
            if fallback_count is not None:
                total = int(fallback_count)
            else:
                total = len(state.get('pellets', [])) + len(state.get('power_pellets', []))
        return total

    def _update_initial_pellet_total(self, env_id, state, fallback_count=None):
        """Refresh the cached pellet baseline for an environment after reset."""

        total = self._resolve_initial_pellet_total(state, fallback_count)
        if env_id < len(self.initial_pellet_totals):
            self.initial_pellet_totals[env_id] = total
        else:
            self.initial_pellet_totals.append(total)
        return total

    def _compute_episode_pellets(self, env_id, remaining_pellets):
        """Compute normalized pellets collected for telemetry emission."""

        if env_id < len(self.initial_pellet_totals):
            initial_total = self.initial_pellet_totals[env_id]
        else:
            initial_total = remaining_pellets
        collected = int(initial_total) - int(remaining_pellets)
        return self._normalize_pellet_metric(env_id, collected, initial_total, remaining_pellets)

    def _normalize_pellet_metric(self, env_id, collected, initial_total, remaining_total):
        """Clamp pellet telemetry to non-negative values while logging anomalies."""

        if collected < 0:
            logger.warning(
                "Pellet baseline underflow detected; clamping to zero",
                extra={
                    'env_id': env_id,
                    'episode': self.episode,
                    'initial_total': initial_total,
                    'remaining_total': remaining_total,
                }
            )
            return 0
        return collected
    
    def _reset_progress_tracker(self, env_id, state):
        self.progress_trackers[env_id] = self._create_progress_tracker(state)
        pellet_count = len(state['pellets']) + len(state['power_pellets'])
        self.prev_pellet_counts[env_id] = pellet_count
        self._update_initial_pellet_total(env_id, state, pellet_count)
    
    def _update_progress_tracker(self, env_id, next_state):
        tracker = self.progress_trackers[env_id]
        current_score = next_state['score']
        pellet_count = len(next_state['pellets']) + len(next_state['power_pellets'])
        position = tuple(next_state['pacman'])
        last_position = tracker.get('last_position', position)
        moved = position != last_position
        tracker['unique_tile_window'].append(position)
        window = tracker['unique_tile_window']
        coverage_ratio = len(set(window)) / max(len(window), 1)
        tracker['coverage_ratio'] = coverage_ratio
        progress_event = current_score > tracker['last_score'] or pellet_count < tracker['last_pellet_count']
        if progress_event:
            tracker['steps_since_progress'] = 0
            tracker['score_freeze_steps'] = 0
            tracker['hunger_meter'] = 0.0
        else:
            tracker['steps_since_progress'] += 1
            tracker['score_freeze_steps'] += 1
        if moved:
            tracker['idle_streak'] = 0
        else:
            tracker['idle_streak'] = tracker.get('idle_streak', 0) + 1
        tracker['last_score'] = current_score
        tracker['last_pellet_count'] = pellet_count
        tracker['last_position'] = position
        return tracker, progress_event, moved
    
    def _apply_hunger_penalty(self, tracker, unique_tiles):
        if tracker['steps_since_progress'] <= self.hunger_config['hunger_idle_threshold']:
            return 0.0
        if unique_tiles >= self.hunger_config['stagnation_tile_min']:
            return 0.0
        idle_steps = tracker['steps_since_progress'] - self.hunger_config['hunger_idle_threshold']
        penalty = self.hunger_config['hunger_decay_rate'] * (
            self.hunger_config['hunger_decay_growth'] ** idle_steps
        )
        tracker['hunger_meter'] -= penalty
        return penalty
    
    def _should_end_for_hunger(self, tracker):
        reasons = []
        if tracker['score_freeze_steps'] > self.hunger_config['survival_grace_steps']:
            reasons.append('HUNGER_SCORE_FREEZE')
        if tracker['hunger_meter'] <= self.hunger_config['hunger_termination_limit']:
            reasons.append('HUNGER_METER')
        if not reasons:
            return False, None
        # Prefer hunger meter termination when both triggered to surface critical starvation
        reason = 'HUNGER_METER' if 'HUNGER_METER' in reasons else reasons[0]
        return True, reason
    
    def _sync_game_hunger_stats(self, game, tracker):
        if hasattr(game, 'update_hunger_stats'):
            unique_tiles = len(set(tracker['unique_tile_window']))
            game.update_hunger_stats({
                'steps_since_progress': tracker['steps_since_progress'],
                'score_freeze_steps': tracker['score_freeze_steps'],
                'hunger_meter': tracker['hunger_meter'],
                'unique_tiles': unique_tiles
            })

    def _infer_maze_diagonal(self):
        if not self.games:
            return 30.0
        sample_game = self.games[0]
        width = getattr(sample_game, 'width', None)
        height = getattr(sample_game, 'height', None)
        if width and height:
            return float(np.sqrt(width ** 2 + height ** 2))
        return 30.0

    def set_telemetry_batch_id(self, batch_id=None):
        """Expose batch identifiers to GUI workers without sharing GUI internals."""

        self.telemetry_collector.set_batch_id(batch_id)
        if batch_id:
            self._perf_logging = self._perf_logging_default or batch_id.startswith(('ultra', 'perf', 'headless'))
        else:
            self._perf_logging = self._perf_logging_default

    def _estimate_sim_speed(self):
        now = time.time()
        steps_since = self._telemetry_total_steps - self._telemetry_last_sample_steps
        elapsed = max(now - self._telemetry_last_sample_time, 1e-6)
        speed = steps_since / elapsed if elapsed > 0 else 0.0
        self._telemetry_last_sample_time = now
        self._telemetry_last_sample_steps = self._telemetry_total_steps
        return speed

    def _build_stats_snapshot(self):
        avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
        avg_length = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0.0
        avg_pellets = np.mean(self.pellets_collected[-100:]) if self.pellets_collected else 0.0
        total_games = self.pacman_wins + self.ghost_wins
        pacman_win_rate = (self.pacman_wins / total_games * 100) if total_games > 0 else 0.0
        ghost_win_rate = (self.ghost_wins / total_games * 100) if total_games > 0 else 0.0
        snapshot = {
            'episode': self.episode,
            'avg_reward': avg_reward,
            'last_reward': self.episode_rewards[-1] if self.episode_rewards else 0.0,
            'n_envs': self.n_envs,
            'pacman_win_rate': pacman_win_rate,
            'ghost_win_rate': ghost_win_rate,
            'avg_length': avg_length,
            'avg_pellets': avg_pellets,
            'best_reward': self.best_reward,
            'pacman_lr': self.pacman_agent.agent.optimizer.param_groups[0]['lr'],
            'ghost_lr': self.ghost_team.qmix.optimizer.param_groups[0]['lr'],
            'entropy_coef': self.pacman_agent.agent.entropy_coef,
            'ghost_epsilon': self.ghost_team.qmix.epsilon,
        }
        return snapshot

    def _emit_episode_telemetry(self, episode_reward, episode_length, pellets_collected, hunger_metrics, hunger_reason, action_histogram=None):
        if not self.telemetry_config.enable_collection:
            return
        snapshot = self._build_stats_snapshot()
        custom_metrics = {
            'best_reward': float(self.best_reward),
            'pacman_lr': snapshot['pacman_lr'],
            'ghost_lr': snapshot['ghost_lr'],
            'entropy_coef': snapshot['entropy_coef'],
            'ghost_epsilon': snapshot['ghost_epsilon'],
            'hunger_meter': hunger_metrics.get('hunger_meter', 0.0),
            'hunger_steps_since_progress': hunger_metrics.get('steps_since_progress', 0),
            'hunger_score_freeze_steps': hunger_metrics.get('score_freeze_steps', 0),
            'hunger_unique_tiles': hunger_metrics.get('unique_tiles', 0),
            'coverage_ratio': hunger_metrics.get('coverage_ratio', 0.0),
            'survival_penalty_events': hunger_metrics.get('survival_penalty_events', 0),
            'pellet_streak': hunger_metrics.get('pellet_streak', 0),
            'hunger_termination_reason': hunger_reason or '',
        }
        if action_histogram:
            action_names = ['up', 'down', 'left', 'right']
            for idx, fraction in enumerate(action_histogram):
                custom_metrics[f'action_{action_names[idx]}'] = float(fraction)
        telemetry = EpisodeTelemetry(
            episode_index=int(snapshot['episode']),
            env_count=int(snapshot['n_envs']),
            avg_reward=float(snapshot['avg_reward']),
            last_reward=float(episode_reward),
            pacman_win_rate=float(snapshot['pacman_win_rate']),
            ghost_win_rate=float(snapshot['ghost_win_rate']),
            avg_length=float(snapshot['avg_length']),
            avg_pellets=float(snapshot['avg_pellets']),
            episode_length=int(episode_length),
            pellets_collected=int(pellets_collected),
            sim_speed_fps=self._estimate_sim_speed(),
            custom_metrics=custom_metrics,
        )
        self.telemetry_collector.record_episode(telemetry)
        if self._perf_logging and telemetry.sim_speed_fps is not None:
            print(
                f"[telemetry] Episode {telemetry.episode_index} | speed {telemetry.sim_speed_fps:.1f} steps/sec | "
                f"reward {episode_reward:.1f} | length {episode_length}"
            )
    
    def train_step(self):
        """Execute one step across all parallel environments"""
        done_any = False
        
        for env_id in range(self.n_envs):
            self._telemetry_total_steps += 1
            game = self.games[env_id]
            state = game.get_state()
            
            # Get Pacman action
            pacman_state = self.pacman_agent.get_state_repr(state)
            pacman_action = self.pacman_agent.get_action(pacman_state)
            
            # Get coordinated ghost actions
            ghost_actions = self.ghost_team.get_actions(state)
            
            # Execute step
            next_state, reward, done = game.step(pacman_action, ghost_actions)

            tracker_result = self._update_progress_tracker(env_id, next_state)
            if isinstance(tracker_result, tuple):
                if len(tracker_result) == 3:
                    tracker, progress_event, moved = tracker_result
                elif len(tracker_result) == 2:
                    tracker, progress_event = tracker_result
                    last_pos = tuple(tracker.get('last_position', tuple(next_state['pacman'])))
                    current_pos = tuple(next_state['pacman'])
                    moved = current_pos != last_pos
                    tracker['last_position'] = current_pos
                else:
                    raise ValueError("_update_progress_tracker returned unexpected tuple length")
            else:
                tracker = tracker_result
                progress_event = False
                last_pos = tuple(tracker.get('last_position', tuple(next_state['pacman'])))
                current_pos = tuple(next_state['pacman'])
                moved = current_pos != last_pos
                tracker['last_position'] = current_pos
            
            # Enhanced reward shaping with survival penalty and goal incentives
            base_reward = reward
            movement_penalty = 0.0
            idle_streak_penalty = 0.0
            axis_lock_penalty = 0.0
            ghost_house_penalty = 0.0
            ghost_movement_penalty = 0.0
            ghost_proximity_reward = 0.0
            survival_penalty = 0.0
            goal_achievement_bonus = 0.0
            pellet_streak_bonus = 0.0
            coverage_bonus = 0.0
            
            # Penalize Pacman for not moving (increased penalty) with idle streak escalation
            if self.prev_pacman_pos[env_id] is not None:
                if tuple(next_state['pacman']) == self.prev_pacman_pos[env_id]:
                    movement_penalty = -1.0
                    reward += movement_penalty
                    self.movement_penalty_count += 1
                    streak = tracker.get('idle_streak', 0)
                    if streak > 0:
                        raw_penalty = self.hunger_config['idle_streak_penalty'] * streak
                        idle_streak_penalty = min(raw_penalty, self.hunger_config['idle_streak_cap'])
                        if idle_streak_penalty < 0:
                            reward += idle_streak_penalty
                            movement_penalty += idle_streak_penalty
            
            # Axis-lock penalties discourage single-axis oscillations
            current_axis = 0 if pacman_action in (0, 1) else 1
            axis_state = self.axis_lock_state[env_id]
            if axis_state['axis'] == current_axis:
                axis_state['streak'] += 1
            else:
                axis_state['axis'] = current_axis
                axis_state['streak'] = 1
            if axis_state['streak'] >= 6:
                axis_lock_penalty = -0.1 * (axis_state['streak'] - 5)
                reward += axis_lock_penalty
            
            # Ghost distance driven survival penalty with cooldown management
            ghost_distances = []
            for ghost_pos in next_state['ghosts']:
                dist = abs(next_state['pacman'][0] - ghost_pos[0]) + abs(next_state['pacman'][1] - ghost_pos[1])
                ghost_distances.append(dist)
            min_ghost_distance = min(ghost_distances)
            if self.survival_cooldowns[env_id] > 0:
                self.survival_cooldowns[env_id] -= 1
            else:
                soft = self.survival_penalty_cfg['soft_threshold']
                hard = self.survival_penalty_cfg['hard_threshold']
                if min_ghost_distance > soft:
                    if min_ghost_distance >= hard:
                        survival_penalty = self.survival_penalty_cfg['hard_penalty']
                    else:
                        interp = (min_ghost_distance - soft) / max(hard - soft, 1)
                        delta = self.survival_penalty_cfg['hard_penalty'] - self.survival_penalty_cfg['soft_penalty']
                        survival_penalty = self.survival_penalty_cfg['soft_penalty'] + interp * delta
                    reward += survival_penalty
                    tracker['survival_penalty_events'] = tracker.get('survival_penalty_events', 0) + 1
                    self.survival_cooldowns[env_id] = self.survival_penalty_cfg['cooldown_steps']
                    if (
                        self.telemetry_config.enable_verbose_logging
                        and env_id == 0
                        and self.episode % 20 == 0
                    ):
                        print(f"  SURVIVAL PENALTY: Ghost distance {min_ghost_distance} triggered scaled penalty")
            
            # Pellet pressure and streak shaping
            current_pellets = len(next_state['pellets']) + len(next_state['power_pellets'])
            prev_pellets = self.prev_pellet_counts[env_id]
            pellets_collected = 0
            if prev_pellets is not None and current_pellets < prev_pellets:
                pellets_collected = prev_pellets - current_pellets
                initial_total = max(self.initial_pellet_totals[env_id], 1)
                progress_ratio = 1.0 - (current_pellets / initial_total)
                pellet_multiplier = 2.0 + 2.5 * progress_ratio
                goal_achievement_bonus = pellets_collected * pellet_multiplier
                reward += goal_achievement_bonus
                self.pellet_streaks[env_id] += pellets_collected
            else:
                self.pellet_streaks[env_id] = 0
            if self.pellet_streaks[env_id] > 1:
                pellet_streak_bonus = self.hunger_config['pellet_streak_bonus'] * self.pellet_streaks[env_id]
                reward += pellet_streak_bonus
            
            # Coverage rewards encourage exploring new tiles even before scoring
            coverage_ratio = tracker.get('coverage_ratio', 0.0)
            if moved and not progress_event and coverage_ratio >= self.hunger_config['coverage_bonus_threshold']:
                coverage_bonus = coverage_ratio * self.hunger_config['coverage_bonus_scale']
                reward += coverage_bonus
            
            # Update pellet tracking
            self.prev_pellet_counts[env_id] = current_pellets
            
            # Reward/penalize ghosts for movement diagnostics (legacy tracking only)
            for i, ghost_pos in enumerate(next_state['ghosts']):
                # Penalty for staying in house when released
                center_x, center_y = game.width // 2, game.height // 2
                in_house = (center_y - 2 <= ghost_pos[0] <= center_y + 2 and 
                           center_x - 3 <= ghost_pos[1] <= center_x + 3)
                if in_house and game.ghost_released[i] and not game.ghost_in_house[i]:
                    ghost_house_penalty -= 1.0  # Track lingering events
                
                # Penalty for not moving (stalling)
                if self.prev_ghost_positions[env_id][i] is not None:
                    if tuple(ghost_pos) == self.prev_ghost_positions[env_id][i]:
                        ghost_movement_penalty -= 0.3
                        self.ghost_movement_penalty_count += 1
                    
                    # Reward for getting closer to Pacman (only if not vulnerable)
                    if not next_state['ghost_vulnerable'][i]:
                        prev_dist = abs(state['pacman'][0] - self.prev_ghost_positions[env_id][i][0]) + \
                                   abs(state['pacman'][1] - self.prev_ghost_positions[env_id][i][1])
                        curr_dist = abs(next_state['pacman'][0] - ghost_pos[0]) + \
                                   abs(next_state['pacman'][1] - ghost_pos[1])
                        if curr_dist < prev_dist:
                            ghost_proximity_reward += 0.1
            
            # Hunger progress tracking & penalties
            unique_tiles = len(set(tracker['unique_tile_window']))
            hunger_penalty = self._apply_hunger_penalty(tracker, unique_tiles)
            if hunger_penalty > 0:
                reward -= hunger_penalty
            hunger_reward = 0.0
            hunger_reason = None
            hunger_triggered = False
            if not done:
                hunger_triggered, hunger_reason = self._should_end_for_hunger(tracker)
            if not done and hunger_triggered:
                done = True
                hunger_reward = self.hunger_config['hunger_termination_reward']
                reward += hunger_reward
                if hunger_reason == 'HUNGER_METER':
                    self.hunger_termination_counts['hunger_meter'] += 1
                else:
                    self.hunger_termination_counts['score_freeze'] += 1
                if hasattr(game, 'force_hunger_termination'):
                    game.force_hunger_termination(hunger_reason)
            active_hunger_reason = hunger_reason if hunger_triggered else None
            self._sync_game_hunger_stats(game, tracker)

            # Ghost team reward decomposition (central QMIX signal)
            #   ghost_reward = team_pressure + proximity_drive + vulnerability_tax
            #   team_pressure offsets pellet gains with idle/survival pressure terms
            delta_pellets = max((prev_pellets - current_pellets), 0) if prev_pellets is not None else 0
            idle_threshold = max(1, self.hunger_config['hunger_idle_threshold'])
            pacman_idle_ratio = float(np.clip(tracker['steps_since_progress'] / idle_threshold, 0.0, 2.0))
            survival_grace = max(1, self.hunger_config['survival_grace_steps'])
            survival_time_ratio = float(np.clip(game.steps / survival_grace, 0.0, 2.0))
            distance_norm = self.ghost_reward_cfg.get('distance_norm') or self.maze_diagonal or 30.0
            pacman_position = np.array(next_state['pacman'], dtype=np.float32)
            normalized_distances = []
            for ghost_pos in next_state['ghosts']:
                ghost_vec = np.array(ghost_pos, dtype=np.float32)
                if distance_norm > 0:
                    normalized_distances.append(np.linalg.norm(pacman_position - ghost_vec) / distance_norm)
            if normalized_distances:
                ghost_ring_distance = float(np.clip(np.mean(normalized_distances), 0.0, 1.0))
            else:
                ghost_ring_distance = 1.0
            vulnerability_fraction = float(np.mean(next_state['ghost_vulnerable'])) if next_state['ghost_vulnerable'] else 0.0

            team_pressure = (
                -self.ghost_reward_cfg['kappa1'] * float(delta_pellets)
                + self.ghost_reward_cfg['kappa2'] * pacman_idle_ratio
                + self.ghost_reward_cfg['kappa3'] * survival_time_ratio
            )
            proximity_drive = self.ghost_reward_cfg['kappa4'] * (1.0 - ghost_ring_distance)
            vulnerability_tax = -self.ghost_reward_cfg['kappa5'] * vulnerability_fraction

            ghost_reward = team_pressure + proximity_drive + vulnerability_tax
            ghost_termination_bonus = 0.0
            termination_reason = active_hunger_reason or getattr(game, 'termination_reason', None)
            if done and termination_reason == 'GHOST_COLLISION':
                ghost_reward += self.ghost_reward_cfg['win_bonus']
                ghost_termination_bonus += self.ghost_reward_cfg['win_bonus']
            hunger_failure = False
            if active_hunger_reason:
                hunger_failure = True
            elif isinstance(termination_reason, str) and termination_reason.upper().startswith('HUNGER'):
                hunger_failure = True
            if hunger_failure:
                ghost_reward -= self.ghost_reward_cfg['hunger_fail_penalty']
                ghost_termination_bonus -= self.ghost_reward_cfg['hunger_fail_penalty']

            ghost_total_reward = float(ghost_reward)
            reward = float(np.clip(reward, *self.reward_clamp))
            
            # Log detailed reward breakdown for first environment only (to avoid spam)
            if (
                self.telemetry_config.enable_verbose_logging
                and env_id == 0
                and self.episode % 10 == 0
            ):
                print(f"Episode {self.episode}, Step {game.steps}:")
                print(f"  Base reward: {base_reward:.2f}")
                print(f"  Movement penalty: {movement_penalty:.2f}")
                print(f"  Idle streak penalty: {idle_streak_penalty:.2f}")
                print(f"  Axis lock penalty: {axis_lock_penalty:.2f}")
                print(f"  Survival penalty: {survival_penalty:.2f}")
                print(f"  Goal achievement bonus: {goal_achievement_bonus:.2f}")
                print(f"  Pellet streak bonus: {pellet_streak_bonus:.2f}")
                print(f"  Coverage bonus: {coverage_bonus:.2f}")
                print(f"  Ghost house penalty: {ghost_house_penalty:.2f}")
                print(f"  Ghost movement penalty: {ghost_movement_penalty:.2f}")
                print(f"  Ghost proximity reward (legacy): {ghost_proximity_reward:.2f}")
                print(f"  Ghost ring distance (norm): {ghost_ring_distance:.2f}")
                print(f"  Ghost team pressure: {team_pressure:.2f}")
                print(f"  Ghost proximity drive: {proximity_drive:.2f}")
                print(f"  Ghost vulnerability tax: {vulnerability_tax:.2f}")
                print(f"  Ghost termination bonus: {ghost_termination_bonus:.2f}")
                print(f"  Ghost total reward: {ghost_total_reward:.2f}")
                print(f"  Hunger penalty: {-hunger_penalty:.2f}")
                print(f"  Hunger meter: {tracker['hunger_meter']:.2f} (limit {self.hunger_config['hunger_termination_limit']})")
                print(f"  Steps since progress: {tracker['steps_since_progress']} (score freeze {tracker['score_freeze_steps']})")
                print(f"  Unique tiles (last {self.hunger_config['stagnation_tile_window']}): {unique_tiles}")
                if hunger_triggered:
                    print(f"  Hunger termination reason: {hunger_reason}")
                print(f"  Final reward: {reward:.2f}")
                print(f"  Pacman action: {pacman_action}")
                print(f"  Ghost actions: {ghost_actions}")
                print(f"  Pacman pos: {next_state['pacman']}")
                print(f"  Ghost positions: {next_state['ghosts']}")
                print(f"  Min ghost distance: {min_ghost_distance}")
                print(f"  Pellets remaining: {current_pellets}")
            
            # Store reward breakdown for analysis
            total_penalty_pressure = movement_penalty + survival_penalty + ghost_movement_penalty + ghost_house_penalty + idle_streak_penalty + axis_lock_penalty
            self.reward_breakdown_history.append({
                'episode': self.episode,
                'step': game.steps,
                'base_reward': base_reward,
                'movement_penalty': movement_penalty,
                'idle_streak_penalty': idle_streak_penalty,
                'axis_lock_penalty': axis_lock_penalty,
                'survival_penalty': survival_penalty,
                'goal_achievement_bonus': goal_achievement_bonus,
                'pellet_streak_bonus': pellet_streak_bonus,
                'coverage_bonus': coverage_bonus,
                'ghost_house_penalty': ghost_house_penalty,
                'ghost_movement_penalty': ghost_movement_penalty,
                'ghost_proximity_reward': ghost_proximity_reward,
                'ghost_team_pressure': team_pressure,
                'ghost_proximity_drive': proximity_drive,
                'ghost_vulnerability_tax': vulnerability_tax,
                'ghost_total_reward': ghost_total_reward,
                'ghost_termination_bonus': ghost_termination_bonus,
                'ghost_ring_distance': ghost_ring_distance,
                'ghost_vulnerability_fraction': vulnerability_fraction,
                'pacman_idle_ratio': pacman_idle_ratio,
                'survival_time_ratio': survival_time_ratio,
                'ghost_delta_pellets': float(delta_pellets),
                'hunger_penalty': -hunger_penalty,
                'hunger_meter': tracker['hunger_meter'],
                'steps_since_progress': tracker['steps_since_progress'],
                'score_freeze_steps': tracker['score_freeze_steps'],
                'unique_tiles': unique_tiles,
                'coverage_ratio': coverage_ratio,
                'survival_penalty_events': tracker.get('survival_penalty_events', 0),
                'survival_cooldown': self.survival_cooldowns[env_id],
                'hunger_termination_reward': hunger_reward,
                'termination_reason': active_hunger_reason or getattr(game, 'termination_reason', None),
                'final_reward': reward,
                'pacman_action': pacman_action,
                'ghost_actions': ghost_actions,
                'min_ghost_distance': min_ghost_distance,
                'pellets_remaining': current_pellets,
                'pellet_streak': self.pellet_streaks[env_id],
                'ghost_release_flags': list(getattr(game, 'ghost_released', [])),
                'ghost_release_events': list(getattr(game, 'last_release_events', [])),
                'total_penalty_pressure': total_penalty_pressure,
            })
            
            self.total_rewards[env_id] += reward
            
            # Update Pacman
            next_pacman_state = self.pacman_agent.get_state_repr(next_state)
            self.pacman_agent.update(pacman_state, pacman_action, reward, next_pacman_state, done)
            
            # Update ghost team with decomposed reward signal
            if self.prev_states[env_id] is not None:
                self.ghost_team.update(
                    self.prev_states[env_id], 
                    self.prev_ghost_actions[env_id], 
                    ghost_total_reward, 
                    state, 
                    done
                )
            
            # Track positions and actions for analysis
            self.prev_states[env_id] = state
            self.prev_ghost_actions[env_id] = ghost_actions
            self.prev_pacman_pos[env_id] = tuple(state['pacman'])
            self.prev_ghost_positions[env_id] = [tuple(g) for g in state['ghosts']]
            
            # Store action history for analysis (first environment only)
            if env_id == 0:
                self.pacman_action_history.append(pacman_action)
                self.ghost_action_history.append(ghost_actions.copy())
                self.epsilon_history.append(self.ghost_team.qmix.epsilon)
            
            # Handle episode completion
            if done:
                episode_reward = self.total_rewards[env_id]
                self.episode_rewards.append(episode_reward)
                episode_length = game.steps
                termination_label = active_hunger_reason or getattr(game, 'termination_reason', None)

                # Track this as the most recently completed environment
                self.last_completed_env = env_id
                
                # Track win/loss
                if reward == 1000:  # Pacman won
                    self.pacman_wins += 1
                elif reward == -500:  # Ghost won
                    self.ghost_wins += 1
                
                # Track episode length and pellets
                self.episode_lengths.append(episode_length)
                remaining_pellets = len(next_state['pellets']) + len(next_state['power_pellets'])
                episode_pellets = self._compute_episode_pellets(env_id, remaining_pellets)
                self.pellets_collected.append(episode_pellets)
                hunger_metrics = {
                    'hunger_meter': tracker['hunger_meter'],
                    'steps_since_progress': tracker['steps_since_progress'],
                    'score_freeze_steps': tracker['score_freeze_steps'],
                    'unique_tiles': len(set(tracker['unique_tile_window'])),
                    'coverage_ratio': tracker.get('coverage_ratio', 0.0),
                    'survival_penalty_events': tracker.get('survival_penalty_events', 0),
                    'pellet_streak': self.pellet_streaks[env_id],
                }
                
                self.episode += 1
                self.total_rewards[env_id] = 0
                reset_state = game.reset()
                self._reset_progress_tracker(env_id, reset_state)
                self.prev_states[env_id] = None
                self.prev_ghost_actions[env_id] = None
                self.prev_pacman_pos[env_id] = None
                self.prev_ghost_positions[env_id] = [None] * 4
                done_any = True
                
                # Save best model
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                    self.save_checkpoint('best')
                
                # Periodic diagnostic summary (every 10 episodes)
                if self.episode % 10 == 0 and env_id == 0:
                    self._print_diagnostic_summary()
                
                action_hist = None
                if env_id == 0 and self.pacman_action_history:
                    counts = np.bincount(self.pacman_action_history, minlength=4)
                    total_actions = counts.sum()
                    if total_actions > 0:
                        action_hist = (counts / total_actions).tolist()

                # Reset action history for next episode
                if env_id == 0:
                    self.pacman_action_history.clear()
                    self.ghost_action_history.clear()
                    self.epsilon_history.clear()

                self._emit_episode_telemetry(
                    episode_reward=episode_reward,
                    episode_length=episode_length,
                    pellets_collected=episode_pellets,
                    hunger_metrics=hunger_metrics,
                    hunger_reason=termination_label,
                    action_histogram=action_hist,
                )
        
        return done_any
    
    def _print_diagnostic_summary(self):
        """Print comprehensive diagnostic information"""
        print(f"\n=== DIAGNOSTIC SUMMARY - Episode {self.episode} ===")
        
        # Movement penalty statistics
        recent_penalties = [r for r in self.reward_breakdown_history[-100:] if r['episode'] >= self.episode - 10]
        if recent_penalties:
            movement_penalties = sum(1 for r in recent_penalties if r['movement_penalty'] < 0)
            survival_penalties = sum(1 for r in recent_penalties if r.get('survival_penalty', 0) < 0)
            ghost_movement_penalties = sum(1 for r in recent_penalties if r['ghost_movement_penalty'] < 0)
            ghost_house_penalties = sum(1 for r in recent_penalties if r['ghost_house_penalty'] < 0)
            
            print(f"Movement Penalties (last 100 steps):")
            print(f"  Pacman not moving: {movement_penalties}/100 ({movement_penalties}%)")
            print(f"  Survival behavior: {survival_penalties}/100 ({survival_penalties}%)")
            print(f"  Ghosts not moving: {ghost_movement_penalties}/100 ({ghost_movement_penalties}%)")
            print(f"  Ghosts in house: {ghost_house_penalties}/100 ({ghost_house_penalties}%)")
            release_samples = [r.get('ghost_release_flags') for r in recent_penalties if r.get('ghost_release_flags')]
            if release_samples:
                latest_release = release_samples[-1]
                release_text = ', '.join('Y' if flag else 'N' for flag in latest_release)
                print(f"  Ghost release flags (latest): {release_text}")
        
        # Reward component analysis
        if recent_penalties:
            avg_base_reward = np.mean([r['base_reward'] for r in recent_penalties])
            avg_movement_penalty = np.mean([r['movement_penalty'] for r in recent_penalties])
            avg_survival_penalty = np.mean([r.get('survival_penalty', 0) for r in recent_penalties])
            avg_goal_bonus = np.mean([r.get('goal_achievement_bonus', 0) for r in recent_penalties])
            avg_ghost_penalties = np.mean([r['ghost_movement_penalty'] + r['ghost_house_penalty'] for r in recent_penalties])
            avg_ghost_team_pressure = np.mean([r.get('ghost_team_pressure', 0) for r in recent_penalties])
            avg_ghost_proximity_drive = np.mean([r.get('ghost_proximity_drive', 0) for r in recent_penalties])
            avg_ghost_vulnerability_tax = np.mean([r.get('ghost_vulnerability_tax', 0) for r in recent_penalties])
            avg_ghost_total_reward = np.mean([r.get('ghost_total_reward', 0) for r in recent_penalties])
            avg_ghost_termination_bonus = np.mean([r.get('ghost_termination_bonus', 0) for r in recent_penalties])
            avg_ring_distance = np.mean([r.get('ghost_ring_distance', 0) for r in recent_penalties])
            
            print(f"Average Reward Components (last 100 steps):")
            print(f"  Base reward: {avg_base_reward:.3f}")
            print(f"  Movement penalty: {avg_movement_penalty:.3f}")
            print(f"  Survival penalty: {avg_survival_penalty:.3f}")
            print(f"  Goal achievement bonus: {avg_goal_bonus:.3f}")
            print(f"  Ghost penalties (legacy): {avg_ghost_penalties:.3f}")
            print(f"  Ghost team pressure: {avg_ghost_team_pressure:.3f}")
            print(f"  Ghost proximity drive: {avg_ghost_proximity_drive:.3f}")
            print(f"  Ghost vulnerability tax: {avg_ghost_vulnerability_tax:.3f}")
            print(f"  Ghost termination bonus: {avg_ghost_termination_bonus:.3f}")
            print(f"  Ghost total reward: {avg_ghost_total_reward:.3f}")
            print(f"  Avg ghost ring distance: {avg_ring_distance:.3f}")
            total_penalties = avg_movement_penalty + avg_survival_penalty + avg_ghost_penalties
            print(f"  Total penalty pressure: {total_penalties:.3f}")
            avg_hunger_penalty = np.mean([r.get('hunger_penalty', 0) for r in recent_penalties])
            avg_hunger_meter = np.mean([r.get('hunger_meter', 0) for r in recent_penalties])
            avg_steps_since_progress = np.mean([r.get('steps_since_progress', 0) for r in recent_penalties])
            avg_score_freeze_steps = np.mean([r.get('score_freeze_steps', 0) for r in recent_penalties])
            avg_unique_tiles = np.mean([r.get('unique_tiles', 0) for r in recent_penalties])
            hunger_events = sum(1 for r in recent_penalties if r.get('termination_reason') in ('HUNGER_METER', 'HUNGER_SCORE_FREEZE'))
            print(f"  Hunger penalty: {avg_hunger_penalty:.3f}")
            print(f"  Avg hunger meter: {avg_hunger_meter:.1f} (limit {self.hunger_config['hunger_termination_limit']})")
            print(f"  Avg steps since progress: {avg_steps_since_progress:.1f} (score freeze {avg_score_freeze_steps:.1f})")
            print(f"  Avg unique tiles ({self.hunger_config['stagnation_tile_window']} window): {avg_unique_tiles:.1f}")
            print(f"  Hunger terminations (last 100 steps): {hunger_events}")
            
            # ADDED: Goal-oriented behavior analysis
            avg_ghost_distance = np.mean([r.get('min_ghost_distance', 0) for r in recent_penalties])
            avg_pellets_remaining = np.mean([r.get('pellets_remaining', 0) for r in recent_penalties])
            print(f"Behavioral Analysis:")
            print(f"  Average min ghost distance: {avg_ghost_distance:.1f}")
            print(f"  Average pellets remaining: {avg_pellets_remaining:.0f}")
            
            if avg_ghost_distance > 10:
                print(f"  ⚠️  WARNING: High ghost distance suggests survival-focused behavior!")
            if avg_pellets_remaining > 100:
                print(f"  ⚠️  WARNING: Many pellets remaining suggests poor goal completion!")
        
        # Epsilon analysis
        current_epsilon = self.ghost_team.qmix.epsilon
        print(f"Current ghost epsilon: {current_epsilon:.3f}")
        if current_epsilon < 0.1:
            print("  WARNING: Epsilon is very low - agents may not be exploring enough!")
        
        # Action distribution analysis
        if self.pacman_action_history:
            unique_actions, counts = np.unique(self.pacman_action_history, return_counts=True)
            action_percentages = counts / len(self.pacman_action_history) * 100
            print(f"Pacman action distribution (last episode):")
            action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
            for action, percentage in zip(unique_actions, action_percentages):
                print(f"  {action_names[action]}: {percentage:.1f}%")
            
            # Check for action diversity
            if len(unique_actions) < 3:
                print("  WARNING: Low action diversity - agents may be stuck in local optima!")
        
        # Training statistics
        avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
        avg_length = np.mean(self.episode_lengths[-10:]) if self.episode_lengths else 0
        print(f"Training Progress:")
        print(f"  Average reward (last 10 episodes): {avg_reward:.1f}")
        print(f"  Average episode length (last 10): {avg_length:.0f} steps")
        print(f"  Total movement penalties applied: {self.movement_penalty_count}")
        print(f"  Total ghost movement penalties: {self.ghost_movement_penalty_count}")
        print(f"  Hunger terminations (score freeze / meter): {self.hunger_termination_counts['score_freeze']} / {self.hunger_termination_counts['hunger_meter']}")
        
        print("=" * 50)
    
    def analyze_epsilon_decay(self):
        """Analyze epsilon decay over training episodes"""
        if not self.episode_rewards:
            print("No training data available for epsilon analysis")
            return
        
        print(f"\n=== EPSILON DECAY ANALYSIS ===")
        current_epsilon = self.ghost_team.qmix.epsilon
        initial_epsilon = 1.0
        epsilon_min = 0.05
        
        # Calculate how many episodes to reach current epsilon
        decay_rate = self.ghost_team.qmix.epsilon_decay
        episodes_to_current = 0
        temp_epsilon = initial_epsilon
        
        while temp_epsilon > current_epsilon and episodes_to_current < 10000:
            temp_epsilon *= decay_rate
            episodes_to_current += 1
        
        print(f"Current epsilon: {current_epsilon:.4f}")
        print(f"Initial epsilon: {initial_epsilon}")
        print(f"Minimum epsilon: {epsilon_min}")
        print(f"Decay rate per episode: {decay_rate:.4f}")
        print(f"Episodes to reach current epsilon: ~{episodes_to_current}")
        print(f"Episodes to reach minimum epsilon: ~{-np.log(epsilon_min/initial_epsilon)/np.log(decay_rate):.0f}")
        
        # Analyze recent episodes
        if self.episode > 10:
            recent_episodes = min(50, self.episode)
            print(f"Analysis based on last {recent_episodes} episodes:")
            
            # Estimate current phase
            if current_epsilon > 0.5:
                print("  STATUS: Early exploration phase (high epsilon)")
            elif current_epsilon > 0.2:
                print("  STATUS: Transition phase (moderate epsilon)")
            elif current_epsilon > epsilon_min:
                print("  STATUS: Late exploration phase (low epsilon)")
            else:
                print("  STATUS: Exploitation phase (minimum epsilon)")
                
            # Recommendations
            if current_epsilon < 0.1:
                print("  WARNING: Epsilon may be too low for effective exploration!")
                print("  RECOMMENDATION: Consider increasing minimum epsilon or slowing decay rate")
            elif current_epsilon > 0.7 and self.episode > 100:
                print("  WARNING: Epsilon is still very high after many episodes!")
                print("  RECOMMENDATION: Consider increasing decay rate")
        
        print("=" * 35)
    
    def get_stats(self):
        snapshot = self._build_stats_snapshot()
        # Get gradient statistics
        pacman_grad_stats = self.pacman_agent.agent.get_grad_stats()
        ghost_grad_stats = self.ghost_team.qmix.get_grad_stats()
        snapshot.update({
            'pacman_grad_norm': pacman_grad_stats['mean'],
            'ghost_grad_norm': ghost_grad_stats['mean'],
        })
        return snapshot
    
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
