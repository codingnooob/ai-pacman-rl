#!/usr/bin/env python3
"""
Simplified Episode Termination Analysis
Quick investigation of 5000+ step episode behavior
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
    return HUNGER_CONFIG['hunger_decay_rate'] * (
        HUNGER_CONFIG['hunger_decay_growth'] ** idle_steps
    )


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
    return team_pressure, proximity_drive, vulnerability_tax


def analyze_episode_termination_patterns():
    """Quick analysis of why episodes don't terminate naturally"""
    
    print("=== EPISODE TERMINATION PATTERN ANALYSIS ===")
    print("Investigating why agents run for extended periods...")
    
    pacman_agent = PacmanAgent()
    ghost_team = GhostTeam()
    
    # Test multiple episodes to see patterns
    episode_results = []
    
    for episode_num in range(5):
        print(f"\n--- Episode {episode_num + 1} ---")
        
        game = PacmanGame()
        state = game.reset()
        initial_pellet_total = resolve_initial_pellet_total(state)
        
        # Track episode progression
        step = 0
        max_steps = 500  # Reasonable limit for analysis
        pellet_count = len(state['pellets']) + len(state['power_pellets'])
        ghost_released = state.get('ghost_released', [True, False, False, False])
        prev_pellet_count = pellet_count
        recent_positions = deque(maxlen=HUNGER_CONFIG['stagnation_tile_window'])
        recent_positions.append(state['pacman'])
        episode_hunger_penalty = 0.0
        episode_ghost_pressures = []
        
        position_history = []
        action_history = []
        hunger_samples = []
        
        while step < max_steps:
            position_history.append(state['pacman'])
            pacman_action = pacman_agent.get_action(pacman_agent.get_state_repr(state))
            action_history.append(pacman_action)
            
            ghost_actions = ghost_team.get_actions(state)
            next_state, reward, done = game.step(pacman_action, ghost_actions)
            steps_since_progress = getattr(game, 'steps_since_progress', 0)
            hunger_samples.append((
                steps_since_progress,
                getattr(game, 'score_freeze_steps', 0),
                getattr(game, 'hunger_meter', 0.0)
            ))
            
            next_pellet_count = len(next_state['pellets']) + len(next_state['power_pellets'])
            if next_pellet_count < pellet_count:
                pellet_count = next_pellet_count
            
            recent_positions.append(next_state['pacman'])
            unique_tiles = len(set(recent_positions))
            estimated_penalty = estimate_hunger_penalty(steps_since_progress, unique_tiles)
            episode_hunger_penalty += estimated_penalty
            team_pressure, proximity_drive, vulnerability_tax = compute_ghost_pressure(
                next_state,
                prev_pellet_count,
                next_pellet_count,
                steps_since_progress,
                getattr(game, 'steps', 0)
            )
            episode_ghost_pressures.append((team_pressure, proximity_drive, vulnerability_tax))
            prev_pellet_count = next_pellet_count
            
            state = next_state
            step += 1
            
            if done:
                break
        
        analysis = analyze_episode_behavior(
            step,
            position_history,
            action_history,
            pellet_count,
            reward if 'reward' in locals() else 0,
            done,
            initial_pellet_total,
        )
        if hunger_samples:
            avg_steps_since = np.mean([sample[0] for sample in hunger_samples])
            avg_score_freeze = np.mean([sample[1] for sample in hunger_samples])
            avg_hunger_meter = np.mean([sample[2] for sample in hunger_samples])
            analysis.update({
                'avg_steps_since_progress': avg_steps_since,
                'avg_score_freeze_steps': avg_score_freeze,
                'avg_hunger_meter': avg_hunger_meter
            })
        if step > 0:
            analysis['avg_hunger_penalty'] = -(episode_hunger_penalty / step)
            analysis['total_hunger_penalty'] = -episode_hunger_penalty
        if episode_ghost_pressures:
            analysis['avg_ghost_team_pressure'] = np.mean([g[0] for g in episode_ghost_pressures])
            analysis['avg_ghost_proximity_drive'] = np.mean([g[1] for g in episode_ghost_pressures])
            analysis['avg_ghost_vulnerability_tax'] = np.mean([g[2] for g in episode_ghost_pressures])
        episode_results.append(analysis)
        
        print(f"  Episode length: {step} steps")
        print(f"  Termination: {'Natural ending' if done else f'Timeout at {max_steps} steps'}")
        print(f"  Pellets remaining: {pellet_count}")
        print(f"  Movement pattern: {analysis['movement_pattern']}")
        print(f"  Evasion score: {analysis['evasion_score']}")
        if 'avg_hunger_penalty' in analysis:
            print(f"  Avg hunger penalty estimate: {analysis['avg_hunger_penalty']:.3f}")
        if 'avg_ghost_team_pressure' in analysis:
            print(f"  Ghost pressure: team={analysis['avg_ghost_team_pressure']:.3f}, "
                  f"prox={analysis['avg_ghost_proximity_drive']:.3f}, "
                  f"vuln={analysis['avg_ghost_vulnerability_tax']:.3f}")
    
    return episode_results

def analyze_episode_behavior(episode_length, positions, actions, pellets_remaining, final_reward, natural_ending, initial_pellets_total):
    """Analyze behavior patterns in a single episode"""
    
    # Movement analysis
    movements = 0
    for i in range(1, len(positions)):
        if positions[i] != positions[i-1]:
            movements += 1
    
    movement_rate = movements / episode_length if episode_length > 0 else 0
    
    # Position diversity
    unique_positions = len(set(positions))
    position_diversity = unique_positions / episode_length if episode_length > 0 else 0
    
    # Action diversity
    unique_actions = len(set(actions))
    action_diversity = unique_actions / 4  # 4 possible actions
    
    # Determine movement pattern
    if movement_rate > 0.6:
        movement_pattern = "High movement"
    elif movement_rate > 0.3:
        movement_pattern = "Moderate movement"
    else:
        movement_pattern = "Low movement"
    
    # Determine evasion score (based on pellet collection efficiency)
    collected = max(initial_pellets_total - pellets_remaining, 0)
    pellet_collection_rate = collected / episode_length if episode_length > 0 else 0
    
    if pellet_collection_rate > 0.1:
        evasion_score = "Goal-oriented"
    elif pellet_collection_rate > 0.01:
        evasion_score = "Mixed behavior"
    else:
        evasion_score = "Survival-focused"
    
    return {
        'episode_length': episode_length,
        'natural_ending': natural_ending,
        'movement_rate': movement_rate,
        'position_diversity': position_diversity,
        'action_diversity': action_diversity,
        'pellets_remaining': pellets_remaining,
        'pellet_collection_rate': pellet_collection_rate,
        'final_reward': final_reward,
        'movement_pattern': movement_pattern,
        'evasion_score': evasion_score,
        'initial_pellets': initial_pellets_total,
    }

def investigate_survival_vs_goal_optimization():
    """Investigate if agents are optimizing survival over goals"""
    
    print(f"\n=== SURVIVAL vs GOAL OPTIMIZATION ANALYSIS ===")
    
    pacman_agent = PacmanAgent()
    ghost_team = GhostTeam()
    game = PacmanGame()
    
    # Run extended episode with detailed tracking
    state = game.reset()
    step = 0
    max_steps = 1000
    
    survival_metrics = []
    goal_metrics = []
    hunger_samples = []
    prev_pellet_count = len(state['pellets']) + len(state['power_pellets'])
    recent_positions = deque(maxlen=HUNGER_CONFIG['stagnation_tile_window'])
    recent_positions.append(state['pacman'])
    episode_hunger_penalty = 0.0
    ghost_pressure_samples = []
    
    while step < max_steps:
        pacman_state = pacman_agent.get_state_repr(state)
        pacman_action = pacman_agent.get_action(pacman_state)
        ghost_actions = ghost_team.get_actions(state)
        
        ghost_distances = []
        for ghost_pos in state['ghosts']:
            dist = abs(state['pacman'][0] - ghost_pos[0]) + abs(state['pacman'][1] - ghost_pos[1])
            ghost_distances.append(dist)
        min_ghost_distance = min(ghost_distances)
        
        all_food = list(state['pellets']) + list(state['power_pellets'])
        if all_food:
            nearest_food = min(all_food, key=lambda p: abs(p[0] - state['pacman'][0]) + abs(p[1] - state['pacman'][1]))
            food_distance = abs(nearest_food[0] - state['pacman'][0]) + abs(nearest_food[1] - state['pacman'][1])
        else:
            food_distance = 0
        
        survival_metrics.append(min_ghost_distance)
        goal_metrics.append(food_distance)
        
        next_state, reward, done = game.step(pacman_action, ghost_actions)
        steps_since_progress = getattr(game, 'steps_since_progress', 0)
        hunger_samples.append((
            steps_since_progress,
            getattr(game, 'score_freeze_steps', 0),
            getattr(game, 'hunger_meter', 0.0)
        ))
        
        recent_positions.append(next_state['pacman'])
        unique_tiles = len(set(recent_positions))
        episode_hunger_penalty += estimate_hunger_penalty(steps_since_progress, unique_tiles)
        next_pellet_count = len(next_state['pellets']) + len(next_state['power_pellets'])
        team_pressure, proximity_drive, vulnerability_tax = compute_ghost_pressure(
            next_state,
            prev_pellet_count,
            next_pellet_count,
            steps_since_progress,
            getattr(game, 'steps', 0)
        )
        ghost_pressure_samples.append((team_pressure, proximity_drive, vulnerability_tax))
        prev_pellet_count = next_pellet_count
        
        state = next_state
        step += 1
        
        if done:
            break
    
    avg_survival_distance = np.mean(survival_metrics)
    avg_goal_distance = np.mean(goal_metrics)
    early_survival = np.mean(survival_metrics[:50]) if len(survival_metrics) >= 50 else np.mean(survival_metrics)
    late_survival = np.mean(survival_metrics[-50:]) if len(survival_metrics) >= 50 else np.mean(survival_metrics)
    survival_trend = late_survival - early_survival
    
    print(f"Survival vs Goal Analysis ({step} steps):")
    print(f"  Average distance from ghosts: {avg_survival_distance:.1f}")
    print(f"  Average distance from food: {avg_goal_distance:.1f}")
    print(f"  Survival distance trend: {survival_trend:+.1f}")
    if step > 0:
        print(f"  Avg hunger penalty per step: {-episode_hunger_penalty / step:.3f}")
    if ghost_pressure_samples:
        avg_team = np.mean([g[0] for g in ghost_pressure_samples])
        avg_proximity = np.mean([g[1] for g in ghost_pressure_samples])
        avg_vuln = np.mean([g[2] for g in ghost_pressure_samples])
        print(f"  Ghost pressure: team={avg_team:.3f}, prox={avg_proximity:.3f}, vuln={avg_vuln:.3f}")
    
    if hunger_samples:
        avg_steps_since_progress = np.mean([sample[0] for sample in hunger_samples])
        avg_score_freeze = np.mean([sample[1] for sample in hunger_samples])
        avg_hunger_meter = np.mean([sample[2] for sample in hunger_samples])
        print(f"  Avg steps since progress: {avg_steps_since_progress:.1f}")
        print(f"  Avg score-freeze steps: {avg_score_freeze:.1f}")
        print(f"  Avg hunger meter: {avg_hunger_meter:.2f}")
    else:
        avg_steps_since_progress = 0
        avg_score_freeze = 0
        avg_hunger_meter = 0
    
    if survival_trend > 1:
        primary_objective = "Survival optimization"
    elif avg_survival_distance > avg_goal_distance * 2:
        primary_objective = "Evasion-focused"
    else:
        primary_objective = "Goal-oriented"
    
    print(f"  Primary objective: {primary_objective}")
    
    result = {
        'episode_length': step,
        'survival_focus': survival_trend > 1,
        'primary_objective': primary_objective,
        'avg_ghost_distance': avg_survival_distance,
        'avg_food_distance': avg_goal_distance,
        'avg_steps_since_progress': avg_steps_since_progress,
        'avg_score_freeze_steps': avg_score_freeze,
        'avg_hunger_meter': avg_hunger_meter,
    }
    if step > 0:
        result['avg_hunger_penalty'] = -(episode_hunger_penalty / step)
    if ghost_pressure_samples:
        result['avg_ghost_team_pressure'] = np.mean([g[0] for g in ghost_pressure_samples])
        result['avg_ghost_proximity_drive'] = np.mean([g[1] for g in ghost_pressure_samples])
        result['avg_ghost_vulnerability_tax'] = np.mean([g[2] for g in ghost_pressure_samples])
    return result

def main():
    """Run simplified episode termination analysis"""
    
    print("AI Pacman RL Episode Termination Analysis")
    print("Quick investigation of extended episode behavior...")
    
    # Analyze episode patterns
    episode_results = analyze_episode_termination_patterns()
    
    # Investigate survival vs goal optimization
    optimization_analysis = investigate_survival_vs_goal_optimization()
    
    telemetry_tracker = TelemetrySummaryTracker(
        config=TelemetryConfig(
            enable_collection=True,
            enable_dispatcher=True,
            channel_capacity=12,
            drop_oldest=True,
        ),
        flush_every=3,
    )
    for idx, result in enumerate(episode_results):
        pellets_remaining = int(result.get('pellets_remaining', 0))
        episode_length = int(result.get('episode_length', 0))
        final_reward = float(result.get('final_reward', 0.0))
        initial_total = int(result.get('initial_pellets', pellets_remaining))
        pellets_collected = max(0, initial_total - pellets_remaining)
        telemetry_tracker.record(
            EpisodeTelemetry(
                episode_index=idx + 1,
                env_count=1,
                avg_reward=final_reward,
                last_reward=final_reward,
                pacman_win_rate=0.0,
                ghost_win_rate=0.0,
                avg_length=float(episode_length),
                avg_pellets=float(pellets_remaining),
                episode_length=episode_length,
                pellets_collected=pellets_collected,
                custom_metrics={
                    'movement_rate': float(result.get('movement_rate', 0.0)),
                    'pellet_collection_rate': float(result.get('pellet_collection_rate', 0.0)),
                    'movement_pattern': result.get('movement_pattern', ''),
                    'evasion_score': result.get('evasion_score', ''),
                    'telemetry_source': 'simple_episode_analysis',
                },
            )
        )
    telemetry_tracker.record(
        EpisodeTelemetry(
            episode_index=len(episode_results) + 1,
            env_count=1,
            avg_reward=float(optimization_analysis.get('avg_ghost_distance', 0.0)),
            last_reward=float(optimization_analysis.get('avg_food_distance', 0.0)),
            pacman_win_rate=0.0,
            ghost_win_rate=0.0,
            avg_length=float(optimization_analysis.get('episode_length', 0)),
            avg_pellets=float(optimization_analysis.get('avg_food_distance', 0.0)),
            episode_length=int(optimization_analysis.get('episode_length', 0)),
            pellets_collected=0,
            custom_metrics={
                'primary_objective': optimization_analysis.get('primary_objective', ''),
                'survival_focus': optimization_analysis.get('survival_focus', False),
                'telemetry_source': 'simple_episode_analysis_summary',
            },
        )
    )
    
    # Summary
    print(f"\n=== ANALYSIS SUMMARY ===")
    
    long_episodes = sum(1 for r in episode_results if r['episode_length'] > 100)
    natural_endings = sum(1 for r in episode_results if r['natural_ending'])
    
    print(f"Episode Pattern Analysis:")
    print(f"  Long episodes (>100 steps): {long_episodes}/{len(episode_results)}")
    print(f"  Natural endings: {natural_endings}/{len(episode_results)}")
    
    low_movement = sum(1 for r in episode_results if r['movement_rate'] < 0.3)
    print(f"  Low movement episodes: {low_movement}/{len(episode_results)}")
    
    goal_oriented = sum(1 for r in episode_results if r['evasion_score'] == "Goal-oriented")
    survival_focused = sum(1 for r in episode_results if r['evasion_score'] == "Survival-focused")
    
    print(f"  Goal-oriented episodes: {goal_oriented}/{len(episode_results)}")
    print(f"  Survival-focused episodes: {survival_focused}/{len(episode_results)}")
    
    hunger_penalty_values = [r['avg_hunger_penalty'] for r in episode_results if 'avg_hunger_penalty' in r]
    if hunger_penalty_values:
        avg_episode_hunger_penalty = np.mean(hunger_penalty_values)
        print(f"  Avg hunger penalty estimate: {avg_episode_hunger_penalty:.3f}")
    ghost_team_values = [r['avg_ghost_team_pressure'] for r in episode_results if 'avg_ghost_team_pressure' in r]
    ghost_prox_values = [r['avg_ghost_proximity_drive'] for r in episode_results if 'avg_ghost_proximity_drive' in r]
    ghost_vuln_values = [r['avg_ghost_vulnerability_tax'] for r in episode_results if 'avg_ghost_vulnerability_tax' in r]
    if ghost_team_values:
        print(f"  Ghost pressure averages: team={np.mean(ghost_team_values):.3f}, "
              f"prox={np.mean(ghost_prox_values):.3f}, vuln={np.mean(ghost_vuln_values):.3f}")
    
    print(f"\nKey Insights:")
    if long_episodes > len(episode_results) / 2:
        print("  PATTERN CONFIRMED: Agents tend to run for extended periods")
    
    if natural_endings < len(episode_results) / 2:
        print("  PATTERN CONFIRMED: Natural episode endings are rare")
    
    if survival_focused > goal_oriented:
        print("  PATTERN CONFIRMED: Agents prioritize survival over goal completion")
        print("  THIS EXPLAINs the 5000+ step behavior!")
    
    print(f"\nPrimary objective in extended test: {optimization_analysis['primary_objective']}")
    
    # Save results
    hunger_summary = {
        'avg_episode_hunger_penalty': float(np.mean(hunger_penalty_values)) if hunger_penalty_values else 0.0,
        'avg_ghost_team_pressure': float(np.mean(ghost_team_values)) if ghost_team_values else 0.0,
        'avg_ghost_proximity_drive': float(np.mean(ghost_prox_values)) if ghost_prox_values else 0.0,
        'avg_ghost_vulnerability_tax': float(np.mean(ghost_vuln_values)) if ghost_vuln_values else 0.0,
    }

    results = {
        'episode_results': episode_results,
        'optimization_analysis': optimization_analysis,
        'hunger_summary': hunger_summary,
        'key_findings': {
            'extended_episodes': long_episodes > len(episode_results) / 2,
            'rare_natural_endings': natural_endings < len(episode_results) / 2,
            'survival_prioritized': survival_focused > goal_oriented
        }
    }
    
    with open('simple_episode_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    telemetry_tracker.finalize()
    telemetry_summary = telemetry_tracker.summary()
    avg_cpu_text = (
        f"{telemetry_summary['avg_cpu']:.1f}%"
        if telemetry_summary['avg_cpu'] is not None
        else 'n/a'
    )
    print(
        f"Telemetry Summary: produced={telemetry_summary['produced']} | "
        f"dispatched={telemetry_summary['dispatched']} | batches={telemetry_summary['flushes']} "
        f"(capacity {telemetry_summary['channel_capacity']}) | avg CPU={avg_cpu_text}"
    )
    print(f"\n📊 Analysis results saved to simple_episode_analysis.json")

if __name__ == "__main__":
    main()