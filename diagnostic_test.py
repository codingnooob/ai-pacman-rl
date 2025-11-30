#!/usr/bin/env python3
"""
Diagnostic Test for AI Pacman RL Training Progression Issue
Validates root cause hypotheses with actual training data
"""

import numpy as np
from collections import deque
from game import PacmanGame
from agent import PacmanAgent, GhostTeam
import json

from telemetry import EpisodeTelemetry, TelemetryConfig
from telemetry_testing import TelemetrySummaryTracker

HUNGER_CONFIG = {
    'hunger_idle_threshold': 24,
    'hunger_decay_rate': 0.05,
    'hunger_decay_growth': 1.01,
    'hunger_termination_limit': -150.0,
    'stagnation_tile_window': 48,
    'stagnation_tile_min': 16,
    'survival_grace_steps': 120,
}

GHOST_REWARD_CFG = {
    'kappa1': 1.0,
    'kappa2': 0.5,
    'kappa3': 0.2,
    'kappa4': 0.1,
    'kappa5': 0.3,
}


def resolve_initial_pellet_total(state):
    counts = state.get('initial_counts') or {}
    pellets = counts.get('pellets', len(state.get('pellets', [])))
    power = counts.get('power_pellets', len(state.get('power_pellets', [])))
    return int(pellets) + int(power)


def estimate_hunger_penalty(steps_since_progress, unique_tiles):
    if steps_since_progress <= HUNGER_CONFIG['hunger_idle_threshold']:
        return 0.0
    if unique_tiles >= HUNGER_CONFIG['stagnation_tile_min']:
        return 0.0
    idle_steps = steps_since_progress - HUNGER_CONFIG['hunger_idle_threshold']
    penalty = HUNGER_CONFIG['hunger_decay_rate'] * (
        HUNGER_CONFIG['hunger_decay_growth'] ** idle_steps
    )
    return penalty


def compute_ghost_pressure(state, prev_pellets, current_pellets, steps_since_progress, game_steps):
    delta_pellets = max((prev_pellets - current_pellets), 0)
    idle_ratio = float(np.clip(
        steps_since_progress / max(HUNGER_CONFIG['hunger_idle_threshold'], 1), 0.0, 2.0
    ))
    survival_ratio = float(np.clip(
        game_steps / max(HUNGER_CONFIG['survival_grace_steps'], 1), 0.0, 2.0
    ))

    pacman_position = np.array(state['pacman'], dtype=np.float32)
    height = state['dimensions']['height']
    width = state['dimensions']['width']
    distance_norm = float(np.sqrt(height ** 2 + width ** 2))
    normalized_distances = []
    for ghost_pos in state['ghosts']:
        ghost_vector = np.array(ghost_pos, dtype=np.float32)
        normalized_distances.append(
            np.linalg.norm(pacman_position - ghost_vector) / max(distance_norm, 1.0)
        )
    ghost_ring_distance = (
        float(np.clip(np.mean(normalized_distances), 0.0, 1.0))
        if normalized_distances else 1.0
    )
    vulnerability_fraction = float(np.mean(state['ghost_vulnerable'])) if state['ghost_vulnerable'] else 0.0

    team_pressure = (
        -GHOST_REWARD_CFG['kappa1'] * float(delta_pellets)
        + GHOST_REWARD_CFG['kappa2'] * idle_ratio
        + GHOST_REWARD_CFG['kappa3'] * survival_ratio
    )
    proximity_drive = GHOST_REWARD_CFG['kappa4'] * (1.0 - ghost_ring_distance)
    vulnerability_tax = -GHOST_REWARD_CFG['kappa5'] * vulnerability_fraction
    return {
        'ghost_team_pressure': team_pressure,
        'ghost_proximity_drive': proximity_drive,
        'ghost_vulnerability_tax': vulnerability_tax,
    }


def run_quick_diagnostic():
    """Run focused diagnostic to test penalty pressure and hunger enforcement."""
    
    print("=== AI PACMAN RL DIAGNOSTIC TEST ===")
    
    telemetry_tracker = TelemetrySummaryTracker(
        config=TelemetryConfig(
            enable_collection=True,
            enable_dispatcher=True,
            channel_capacity=8,
            drop_oldest=True,
        ),
        flush_every=2,
    )
    
    pacman_agent = PacmanAgent()
    ghost_team = GhostTeam()
    game = PacmanGame()
    
    print(f"Initial epsilon: {ghost_team.qmix.epsilon:.4f}")
    print(f"Initial entropy: {pacman_agent.agent.entropy_coef:.4f}")
    
    total_steps = 0
    total_penalties = 0
    total_moves = 0
    reward_components = []
    ghost_pressure_records = []
    episode_hunger_penalties = []
    hunger_terminations = 0
    
    for episode in range(3):  # 3 short episodes
        state = game.reset()
        episode_moves = 0
        episode_penalties = 0
        episode_hunger_penalty = 0.0
        episode_ghost_metrics = []
        episode_reward_series = []
        initial_pellet_total = resolve_initial_pellet_total(state)
        prev_pellet_count = len(state['pellets']) + len(state['power_pellets'])
        recent_positions = deque(maxlen=HUNGER_CONFIG['stagnation_tile_window'])
        recent_positions.append(state['pacman'])
        
        print(f"\n--- Episode {episode + 1} ---")
        
        for step in range(100):  # Max 100 steps per episode
            pacman_state = pacman_agent.get_state_repr(state)
            pacman_action = pacman_agent.get_action(pacman_state)
            ghost_actions = ghost_team.get_actions(state)
            
            prev_pacman_pos = state['pacman']
            prev_ghost_positions = [list(g) for g in state['ghosts']]
            
            next_state, reward, done = game.step(pacman_action, ghost_actions)
            
            pacman_moved = next_state['pacman'] != prev_pacman_pos
            ghost_moved = sum(1 for i, g in enumerate(next_state['ghosts'])
                              if g != prev_ghost_positions[i])
            if pacman_moved:
                episode_moves += 1
            
            movement_penalty = -0.5 if not pacman_moved else 0
            ghost_movement_penalty = -0.3 * (4 - ghost_moved)
            total_penalty = movement_penalty + ghost_movement_penalty
            final_reward = reward + total_penalty
            
            current_pellet_count = len(next_state['pellets']) + len(next_state['power_pellets'])
            recent_positions.append(next_state['pacman'])
            unique_tiles = len(set(recent_positions))
            steps_since_progress = getattr(game, 'steps_since_progress', 0)
            estimated_penalty = estimate_hunger_penalty(steps_since_progress, unique_tiles)
            ghost_metrics = compute_ghost_pressure(
                next_state,
                prev_pellet_count,
                current_pellet_count,
                steps_since_progress,
                getattr(game, 'steps', 0)
            )
            prev_pellet_count = current_pellet_count
            episode_hunger_penalty += estimated_penalty
            episode_ghost_metrics.append(ghost_metrics)
            ghost_pressure_records.append(ghost_metrics)
            
            reward_breakdown = {
                'base_reward': reward,
                'movement_penalty': movement_penalty,
                'ghost_penalty': ghost_movement_penalty,
                'final_reward': final_reward,
                'pacman_moved': pacman_moved,
                'ghost_moves': ghost_moved,
                'hunger_penalty': -estimated_penalty,
                'hunger_meter': getattr(game, 'hunger_meter', 0.0),
                'steps_since_progress': steps_since_progress,
                'score_freeze_steps': getattr(game, 'score_freeze_steps', 0),
                'unique_tiles_sampled': unique_tiles,
                'termination_reason': next_state.get('termination_reason', 'RUNNING')
            }
            reward_breakdown.update(ghost_metrics)
            reward_components.append(reward_breakdown)
            episode_reward_series.append(final_reward)
            
            total_steps += 1
            episode_penalties += abs(total_penalty)
            
            ghost_reward = -reward if reward != 0 else -0.1
            pacman_agent.update(pacman_state, pacman_action, final_reward,
                                pacman_agent.get_state_repr(next_state), done)
            ghost_team.update(state, ghost_actions, ghost_reward, next_state, done)
            
            state = next_state
            
            if done:
                termination_label = str(
                    next_state.get('termination_reason', getattr(game, 'termination_reason', ''))
                ).upper()
                if termination_label.startswith('HUNGER'):
                    hunger_terminations += 1
                break
        
        total_moves += episode_moves
        episode_hunger_penalties.append(episode_hunger_penalty)
        print(f"Episode moves: {episode_moves}/100 steps")
        print(f"Episode penalties: {episode_penalties:.1f}")
        print(f"Estimated hunger penalty this episode: {-episode_hunger_penalty:.2f}")
        if episode_ghost_metrics:
            avg_team = np.mean([m['ghost_team_pressure'] for m in episode_ghost_metrics])
            avg_proximity = np.mean([m['ghost_proximity_drive'] for m in episode_ghost_metrics])
            avg_vuln = np.mean([m['ghost_vulnerability_tax'] for m in episode_ghost_metrics])
            print("Ghost pressure snapshot:")
            print(f"  Team pressure: {avg_team:.3f}")
            print(f"  Proximity drive: {avg_proximity:.3f}")
            print(f"  Vulnerability tax: {avg_vuln:.3f}")

        episode_steps = step
        avg_reward = float(np.mean(episode_reward_series)) if episode_reward_series else 0.0
        last_reward_value = float(episode_reward_series[-1]) if episode_reward_series else 0.0
        pellets_collected = max(0, initial_pellet_total - pellet_count)
        telemetry_tracker.record(
            EpisodeTelemetry(
                episode_index=episode + 1,
                env_count=1,
                avg_reward=avg_reward,
                last_reward=last_reward_value,
                pacman_win_rate=0.0,
                ghost_win_rate=0.0,
                avg_length=float(episode_steps),
                avg_pellets=float(pellet_count),
                episode_length=int(episode_steps),
                pellets_collected=int(pellets_collected),
                custom_metrics={
                    'movement_rate': episode_moves / max(episode_steps, 1),
                    'hunger_penalty_total': -episode_hunger_penalty,
                    'telemetry_source': 'diagnostic_test',
                },
            )
        )
    
    analysis_status = analyze_results(
        reward_components,
        total_moves,
        total_steps,
        episode_penalties,
        hunger_terminations,
        episode_hunger_penalties,
        ghost_pressure_records
    )

    telemetry_tracker.finalize()
    telemetry_summary = telemetry_tracker.summary()
    avg_cpu_text = (
        f"{telemetry_summary['avg_cpu']:.1f}%"
        if telemetry_summary['avg_cpu'] is not None
        else 'n/a'
    )
    print(
        f"\nTelemetry Summary: produced={telemetry_summary['produced']} | "
        f"dispatched={telemetry_summary['dispatched']} | batches={telemetry_summary['flushes']} "
        f"(capacity {telemetry_summary['channel_capacity']}) | avg CPU={avg_cpu_text}"
    )

    return analysis_status

def analyze_results(reward_components, total_moves, total_steps, total_penalties,
                    hunger_terminations, episode_hunger_penalties, ghost_pressure_records):
    """Analyze diagnostic results"""
    
    print(f"\n=== DIAGNOSTIC ANALYSIS ===")
    print(f"Total steps analyzed: {total_steps}")
    print(f"Total moves made: {total_moves}")
    print(f"Movement rate: {total_moves/total_steps*100:.1f}%")
    
    base_rewards = [r['base_reward'] for r in reward_components]
    movement_penalties = [r['movement_penalty'] for r in reward_components]
    ghost_penalties = [r['ghost_penalty'] for r in reward_components]
    avg_base_reward = np.mean(base_rewards)
    avg_movement_penalty = np.mean(movement_penalties)
    avg_ghost_penalty = np.mean(ghost_penalties)
    avg_total_penalty = avg_movement_penalty + avg_ghost_penalty
    
    print(f"\nReward Component Analysis:")
    print(f"  Average base reward: {avg_base_reward:.3f}")
    print(f"  Average movement penalty: {avg_movement_penalty:.3f}")
    print(f"  Average ghost penalty: {avg_ghost_penalty:.3f}")
    print(f"  Average total penalty: {avg_total_penalty:.3f}")
    
    penalty_frequency = sum(
        1 for r in reward_components if r['movement_penalty'] < 0 or r['ghost_penalty'] < 0
    ) / max(len(reward_components), 1) * 100
    penalty_magnitude = abs(avg_total_penalty)
    penalty_pressure = penalty_frequency/100 * penalty_magnitude
    
    print(f"\nPenalty Pressure Analysis:")
    print(f"  Steps with penalties: {penalty_frequency:.1f}%")
    print(f"  Average penalty magnitude: {penalty_magnitude:.3f}")
    print(f"  Total penalty pressure: {penalty_pressure:.3f}")
    
    avg_hunger_meter = np.mean([r.get('hunger_meter', 0) for r in reward_components])
    avg_steps_since_progress = np.mean([r.get('steps_since_progress', 0) for r in reward_components])
    avg_score_freeze = np.mean([r.get('score_freeze_steps', 0) for r in reward_components])
    avg_hunger_penalty = np.mean([r.get('hunger_penalty', 0) for r in reward_components])
    print(f"\nHunger Diagnostic Snapshot:")
    print(f"  Avg hunger meter: {avg_hunger_meter:.2f}")
    print(f"  Avg steps since progress: {avg_steps_since_progress:.1f}")
    print(f"  Avg score-freeze steps: {avg_score_freeze:.1f}")
    print(f"  Avg per-step hunger penalty: {avg_hunger_penalty:.3f}")
    if episode_hunger_penalties:
        avg_episode_penalty = -np.mean(episode_hunger_penalties)
        print(f"  Avg hunger penalty per episode: {avg_episode_penalty:.3f}")
    print(f"  Hunger terminations observed: {hunger_terminations}")
    
    if ghost_pressure_records:
        avg_team = np.mean([r['ghost_team_pressure'] for r in ghost_pressure_records])
        avg_proximity = np.mean([r['ghost_proximity_drive'] for r in ghost_pressure_records])
        avg_vuln = np.mean([r['ghost_vulnerability_tax'] for r in ghost_pressure_records])
        print(f"\nGhost Pressure Summary:")
        print(f"  Team pressure: {avg_team:.3f}")
        print(f"  Proximity drive: {avg_proximity:.3f}")
        print(f"  Vulnerability tax: {avg_vuln:.3f}")
    
    high_penalty_pressure = penalty_pressure > 0.3
    low_movement_rate = total_moves/total_steps < 0.3
    
    print(f"\n=== HYPOTHESIS VALIDATION ===")
    print(f"High penalty pressure (>0.3): {high_penalty_pressure} ({penalty_pressure:.3f})")
    print(f"Low movement rate (<30%): {low_movement_rate} ({total_moves/total_steps*100:.1f}%)")
    
    if high_penalty_pressure and low_movement_rate:
        print("✓ PRIMARY HYPOTHESIS VALIDATED")
        print("   Excessive penalties are discouraging agent movement")
        return "validated"
    else:
        print("✗ PRIMARY HYPOTHESIS PARTIALLY VALIDATED")
        print("   May need additional investigation")
        return "partial"
    
    results = {
        'penalty_pressure': penalty_pressure,
        'movement_rate': total_moves/total_steps,
        'penalty_frequency': penalty_frequency,
        'avg_penalties': penalty_magnitude,
        'hypothesis_status': 'validated' if high_penalty_pressure and low_movement_rate else 'partial'
    }
    
    with open('diagnostic_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Results saved to diagnostic_results.json")

if __name__ == "__main__":
    run_quick_diagnostic()