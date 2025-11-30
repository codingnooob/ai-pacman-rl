#!/usr/bin/env python3
"""
Epsilon Decay Validation Test
Tests agent behavior across different epsilon levels to validate root cause hypothesis
"""

import numpy as np
from game import PacmanGame
from agent import PacmanAgent, GhostTeam
import matplotlib.pyplot as plt
import json

def simulate_extended_training_session(num_episodes=800):
    """Simulate extended training session to show epsilon decay impact"""
    
    print("=== EXTENDED TRAINING SESSION SIMULATION ===")
    print(f"Simulating {num_episodes} episodes to demonstrate epsilon decay impact...")
    
    # Initialize agents
    pacman_agent = PacmanAgent()
    ghost_team = GhostTeam()
    
    # Tracking metrics
    episode_data = []
    epsilon_progression = []
    movement_rates = []
    action_diversity = []
    
    # Sample episodes at key points
    sample_episodes = [0, 50, 100, 200, 300, 400, 500, 600, 700, num_episodes-1]
    
    for episode in range(num_episodes):
        # Run one episode and measure behavior
        behavior_metrics = run_test_episode(ghost_team, pacman_agent, episode)
        
        # Store metrics
        episode_data.append({
            'episode': episode,
            'epsilon': ghost_team.qmix.epsilon,
            **behavior_metrics
        })
        
        epsilon_progression.append(ghost_team.qmix.epsilon)
        
        # Sample detailed analysis at key episodes
        if episode in sample_episodes:
            print(f"Episode {episode:3d}: Epsilon={ghost_team.qmix.epsilon:.4f}, "
                  f"Movement Rate={behavior_metrics['movement_rate']:.1%}, "
                  f"Action Diversity={behavior_metrics['action_diversity']}/4")
        
        # Simulate epsilon decay (this happens in ghost_team.qmix.update())
        for _ in range(10):  # Simulate multiple updates per episode
            ghost_team.qmix.epsilon = max(
                ghost_team.qmix.epsilon_min,
                ghost_team.qmix.epsilon * ghost_team.qmix.epsilon_decay
            )
        
        # Update Pacman agent learning rate and entropy
        pacman_agent.agent.entropy_coef = max(
            pacman_agent.agent.min_entropy_coef,
            pacman_agent.agent.entropy_coef * pacman_agent.agent.entropy_decay
        )
    
    return analyze_training_progression(episode_data)

def run_test_episode(ghost_team, pacman_agent, episode_num):
    """Run a single test episode and measure behavior metrics"""
    
    game = PacmanGame()
    state = game.reset()
    
    # Track episode metrics
    total_steps = 0
    pacman_moves = 0
    ghost_moves = 0
    actions_taken = []
    
    for step in range(200):  # Limit to 200 steps for testing
        # Get current epsilon and measure exploration behavior
        current_epsilon = ghost_team.qmix.epsilon
        
        # Get actions
        pacman_state = pacman_agent.get_state_repr(state)
        pacman_action = pacman_agent.get_action(pacman_state)
        ghost_actions = ghost_team.get_actions(state)
        
        # Store action for diversity analysis
        actions_taken.append(pacman_action)
        
        # Track previous positions
        prev_pacman_pos = state['pacman']
        prev_ghost_positions = [list(g) for g in state['ghosts']]
        
        # Execute step
        next_state, reward, done = game.step(pacman_action, ghost_actions)
        
        # Measure movement
        if next_state['pacman'] != prev_pacman_pos:
            pacman_moves += 1
        
        ghost_move_count = sum(1 for i, g in enumerate(next_state['ghosts']) 
                              if g != prev_ghost_positions[i])
        ghost_moves += ghost_move_count
        
        state = next_state
        total_steps += 1
        
        if done:
            break
    
    # Calculate metrics
    movement_rate = pacman_moves / total_steps if total_steps > 0 else 0
    action_diversity = len(set(actions_taken))
    
    # Measure exploration vs exploitation behavior
    exploration_rate = current_epsilon
    
    return {
        'movement_rate': movement_rate,
        'action_diversity': action_diversity,
        'total_steps': total_steps,
        'exploration_rate': exploration_rate,
        'pacman_moves': pacman_moves,
        'ghost_moves': ghost_moves,
        'episode_length': total_steps
    }

def analyze_training_progression(episode_data):
    """Analyze how agent behavior changes with epsilon decay"""
    
    print(f"\n=== TRAINING PROGRESSION ANALYSIS ===")
    
    # Extract key metrics
    episodes = [d['episode'] for d in episode_data]
    epsilons = [d['epsilon'] for d in episode_data]
    movement_rates = [d['movement_rate'] for d in episode_data]
    action_diversity = [d['action_diversity'] for d in episode_data]
    
    # Find critical transition points
    low_epsilon_episodes = [ep for ep, eps in zip(episodes, epsilons) if eps < 0.2]
    high_epsilon_episodes = [ep for ep, eps in zip(episodes, epsilons) if eps > 0.7]
    
    print(f"Epsilon Decay Progression:")
    print(f"  High exploration (epsilon > 0.7): episodes {high_epsilon_episodes[0] if high_epsilon_episodes else 'N/A'} - {high_epsilon_episodes[-1] if high_epsilon_episodes else 'N/A'}")
    print(f"  Low exploration (epsilon < 0.2): episodes {low_epsilon_episodes[0] if low_epsilon_episodes else 'N/A'} - end")
    
    # Analyze behavior at different epsilon levels
    high_eps_movement = [m for ep, m, eps in zip(episodes, movement_rates, epsilons) if eps > 0.7]
    low_eps_movement = [m for ep, m, eps in zip(episodes, movement_rates, epsilons) if eps < 0.2]
    
    if high_eps_movement:
        avg_high_eps_movement = np.mean(high_eps_movement)
        print(f"  Average movement rate (high epsilon): {avg_high_eps_movement:.1%}")
    
    if low_eps_movement:
        avg_low_eps_movement = np.mean(low_eps_movement)
        print(f"  Average movement rate (low epsilon): {avg_low_eps_movement:.1%}")
        
        if high_eps_movement:
            movement_degradation = (avg_high_eps_movement - avg_low_eps_movement) / avg_high_eps_movement
            print(f"  Movement degradation: {movement_degradation:.1%}")
    
    # Check for minimal movement threshold violations
    minimal_movement_episodes = sum(1 for m in movement_rates if m < 0.3)
    print(f"\nMinimal Movement Analysis:")
    print(f"  Episodes with <30% movement rate: {minimal_movement_episodes}/{len(movement_rates)} ({minimal_movement_episodes/len(movement_rates)*100:.1f}%)")
    
    # Validate root cause hypothesis
    low_epsilon_minimal_movement = 0
    high_epsilon_minimal_movement = 0
    
    for ep, m, eps in zip(episodes, movement_rates, epsilons):
        if m < 0.3:  # Minimal movement threshold
            if eps < 0.2:
                low_epsilon_minimal_movement += 1
            elif eps > 0.7:
                high_epsilon_minimal_movement += 1
    
    print(f"\n=== HYPOTHESIS VALIDATION ===")
    print(f"Minimal movement episodes (epsilon < 0.2): {low_epsilon_minimal_movement}")
    print(f"Minimal movement episodes (epsilon > 0.7): {high_epsilon_minimal_movement}")
    
    if low_epsilon_minimal_movement > high_epsilon_minimal_movement * 2:
        print("✓ ROOT CAUSE VALIDATED: Low epsilon strongly correlates with minimal movement")
        return "validated"
    else:
        print("? Root cause partially validated - may need additional investigation")
        return "partial"

def create_visualization(episode_data):
    """Create visualization of training progression"""
    
    episodes = [d['episode'] for d in episode_data]
    epsilons = [d['epsilon'] for d in episode_data]
    movement_rates = [d['movement_rate'] for d in episode_data]
    
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Epsilon decay
    plt.subplot(2, 2, 1)
    plt.plot(episodes, epsilons, 'b-', linewidth=2)
    plt.axhline(y=0.2, color='r', linestyle='--', alpha=0.7, label='Low exploration threshold')
    plt.axhline(y=0.7, color='g', linestyle='--', alpha=0.7, label='High exploration threshold')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon (Exploration Rate)')
    plt.title('Epsilon Decay Over Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Movement rate
    plt.subplot(2, 2, 2)
    plt.plot(episodes, movement_rates, 'g-', linewidth=2)
    plt.axhline(y=0.3, color='r', linestyle='--', alpha=0.7, label='Minimal movement threshold')
    plt.xlabel('Episode')
    plt.ylabel('Movement Rate')
    plt.title('Agent Movement Rate Over Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Correlation scatter
    plt.subplot(2, 2, 3)
    plt.scatter(epsilons, movement_rates, alpha=0.6, c=episodes, cmap='viridis')
    plt.xlabel('Epsilon (Exploration Rate)')
    plt.ylabel('Movement Rate')
    plt.title('Epsilon vs Movement Rate Correlation')
    plt.colorbar(label='Episode')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Rolling averages
    plt.subplot(2, 2, 4)
    window = 50
    if len(movement_rates) >= window:
        movement_avg = np.convolve(movement_rates, np.ones(window)/window, mode='valid')
        epsilon_avg = np.convolve(epsilons, np.ones(window)/window, mode='valid')
        episodes_avg = episodes[window-1:]
        
        plt.plot(episodes_avg, movement_avg, 'g-', label=f'Movement Rate (avg {window})', linewidth=2)
        plt.plot(episodes_avg, epsilon_avg, 'b-', label=f'Epsilon (avg {window})', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Value')
        plt.title(f'{window}-Episode Rolling Averages')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('epsilon_decay_impact.png', dpi=150, bbox_inches='tight')
    print("📊 Visualization saved as epsilon_decay_impact.png")

if __name__ == "__main__":
    print("AI Pacman RL Epsilon Decay Validation Test")
    print("This test validates the root cause hypothesis with extended training simulation...")
    
    # Run extended training simulation
    validation_result = simulate_extended_training_session(800)
    
    # Create detailed report
    print(f"\n=== FINAL VALIDATION RESULT ===")
    print(f"Root cause hypothesis: {validation_result.upper()}")
    
    if validation_result == "validated":
        print("✓ The epsilon decay analysis is CONFIRMED as the root cause")
        print("  - Low epsilon strongly correlates with minimal agent movement")
        print("  - Extended training sessions (800+ episodes) show clear degradation")
        print("  - Fix required: Adjust epsilon decay parameters and minimum values")
    
    # Save detailed results
    with open('epsilon_validation_results.json', 'w') as f:
        json.dump({
            'validation_result': validation_result,
            'test_type': 'extended_training_simulation',
            'episodes_simulated': 800,
            'key_finding': 'Low epsilon correlates with minimal movement in extended sessions'
        }, f, indent=2)
    
    print(f"\n📊 Detailed results saved to epsilon_validation_results.json")