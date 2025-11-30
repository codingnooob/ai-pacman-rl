#!/usr/bin/env python3
"""
Epsilon Decay Analysis - Investigate exploration rate over training
"""

from agent import GhostTeam
import math

def analyze_epsilon_decay():
    """Analyze epsilon decay patterns and impact on exploration"""
    
    print("=== EPSILON DECAY ANALYSIS ===")
    
    # Test epsilon decay parameters from agent.py
    initial_epsilon = 1.0
    epsilon_decay = 0.995  # From QMIXAgent.__init__
    epsilon_min = 0.05
    
    print(f"Initial epsilon: {initial_epsilon}")
    print(f"Decay rate: {epsilon_decay}")
    print(f"Minimum epsilon: {epsilon_min}")
    
    # Calculate decay progression
    episodes = []
    epsilons = []
    
    epsilon = initial_epsilon
    episode = 0
    
    while epsilon > epsilon_min and episode < 2000:
        episodes.append(episode)
        epsilons.append(epsilon)
        epsilon *= epsilon_decay
        episode += 1
    
    print(f"\nEpsilon Decay Timeline:")
    key_episodes = [0, 50, 100, 200, 500, 1000, 1500, len(episodes)-1]
    for ep in key_episodes:
        if ep < len(episodes):
            print(f"  Episode {ep:4d}: epsilon = {epsilons[ep]:.4f}")
    
    # Analyze exploration phases
    print(f"\nExploration Phase Analysis:")
    high_exploration = sum(1 for e in epsilons if e > 0.7)
    moderate_exploration = sum(1 for e in epsilons if 0.2 <= e <= 0.7)
    low_exploration = sum(1 for e in epsilons if epsilon_min < e < 0.2)
    
    total_episodes = len(epsilons)
    print(f"  High exploration (epsilon > 0.7): {high_exploration/total_episodes*100:.1f}% ({high_exploration} episodes)")
    print(f"  Moderate exploration (0.2-0.7): {moderate_exploration/total_episodes*100:.1f}% ({moderate_exploration} episodes)")
    print(f"  Low exploration (<0.2): {low_exploration/total_episodes*100:.1f}% ({low_exploration} episodes)")
    
    # Check if low exploration starts too early
    low_exp_start = next((i for i, e in enumerate(epsilons) if e < 0.2), None)
    if low_exp_start:
        print(f"\n⚠️  Low exploration begins at episode {low_exp_start}")
        print(f"   This may explain movement degradation in long training sessions!")
        
        if low_exp_start < 200:
            print(f"   RECOMMENDATION: Increase minimum epsilon or slow decay rate")
    
    # Calculate episodes to reach various thresholds
    thresholds = [0.5, 0.3, 0.1, 0.05]
    print(f"\nEpisodes to reach exploration thresholds:")
    for threshold in thresholds:
        episodes_to_threshold = -math.log(threshold/initial_epsilon) / math.log(epsilon_decay)
        print(f"  Epsilon {threshold}: {episodes_to_threshold:.0f} episodes")
    
    return {
        'low_exploration_start': low_exp_start,
        'total_exploration_episodes': total_episodes,
        'final_epsilon': epsilons[-1] if epsilons else epsilon_min
    }

def test_training_vs_inference_behavior():
    """Test differences between training and inference behavior"""
    
    print(f"\n=== TRAINING vs INFERENCE BEHAVIOR ===")
    
    # Initialize fresh ghost team
    ghost_team = GhostTeam()
    
    print(f"Initial state:")
    print(f"  Epsilon: {ghost_team.qmix.epsilon:.4f}")
    print(f"  Training mode: Active")
    
    # Simulate some training updates to change epsilon
    for i in range(50):
        # Simulate epsilon decay (this happens in ghost_team.qmix.update())
        ghost_team.qmix.epsilon = max(ghost_team.qmix.epsilon_min, 
                                     ghost_team.qmix.epsilon * ghost_team.qmix.epsilon_decay)
    
    print(f"\nAfter 50 training updates:")
    print(f"  Epsilon: {ghost_team.qmix.epsilon:.4f}")
    
    # Test action selection with current epsilon
    import random
    dummy_state = {
        'pacman': (10, 10),
        'ghosts': [(10, 10), (10, 11), (11, 10), (11, 11)],
        'ghost_vulnerable': [False] * 4,
        'pellets': set(),
        'power_pellets': set(),
        'walls': set(),
        'dimensions': {'height': 31, 'width': 28},
        'initial_counts': {'pellets': 1, 'power_pellets': 1},
        'hunger_stats': {
            'steps_since_progress': 0,
            'score_freeze_steps': 0,
            'hunger_meter': 0.0,
            'unique_tiles': 1
        },
        'hunger_config': {
            'hunger_idle_threshold': 24,
            'survival_grace_steps': 120,
            'stagnation_tile_window': 48,
            'hunger_termination_limit': -150.0
        },
        'pacman_velocity': (0, 0)
    }
    
    exploration_actions = 0
    exploitation_actions = 0
    
    for _ in range(100):
        actions = ghost_team.get_actions(dummy_state)
        if random.random() < ghost_team.qmix.epsilon:
            exploration_actions += 1
        else:
            exploitation_actions += 1
    
    print(f"\nAction selection test (100 actions):")
    print(f"  Exploration actions: {exploration_actions}")
    print(f"  Exploitation actions: {exploitation_actions}")
    print(f"  Actual exploration rate: {exploration_actions/100*100:.1f}%")
    
    return ghost_team.qmix.epsilon

if __name__ == "__main__":
    epsilon_analysis = analyze_epsilon_decay()
    final_epsilon = test_training_vs_inference_behavior()
    
    print(f"\n=== SUMMARY ===")
    print(f"Key findings:")
    print(f"- Epsilon decays to minimum in ~{epsilon_analysis['total_exploration_episodes']} episodes")
    print(f"- Low exploration starts at episode {epsilon_analysis['low_exploration_start']}")
    print(f"- This may cause movement degradation in long training sessions")