#!/usr/bin/env python3
"""
Simple test script to validate the reward structure fixes
"""

import numpy as np
from game_fixed import PacmanGame
from agent import PacmanAgent, GhostTeam
import json


def resolve_initial_pellets(state):
    counts = state.get('initial_counts') or {}
    pellets = counts.get('pellets', len(state.get('pellets', [])))
    power = counts.get('power_pellets', len(state.get('power_pellets', [])))
    return int(pellets) + int(power)


def test_reward_fixes():
    """Test that the reward structure fixes work correctly"""
    
    print("=== TESTING REWARD STRUCTURE FIXES ===")
    print("Validating that survival-focused behavior is now penalized...")
    
    pacman_agent = PacmanAgent()
    ghost_team = GhostTeam()
    game = PacmanGame()
    
    state = game.reset()
    initial_pellet_total = resolve_initial_pellets(state)
    step = 0
    survival_metrics = []
    goal_metrics = []
    reward_history = []
    
    print(f"Initial pellets: {initial_pellet_total}")
    
    # Test for 200 steps to see behavior changes
    while step < 200:
        pacman_state = pacman_agent.get_state_repr(state)
        pacman_action = pacman_agent.get_action(pacman_state)
        ghost_actions = ghost_team.get_actions(state)
        
        # Track survival metrics
        ghost_distances = []
        for ghost_pos in state['ghosts']:
            dist = abs(state['pacman'][0] - ghost_pos[0]) + abs(state['pacman'][1] - ghost_pos[1])
            ghost_distances.append(dist)
        
        min_ghost_distance = min(ghost_distances)
        survival_metrics.append(min_ghost_distance)
        
        # Track goal metrics
        all_food = list(state['pellets']) + list(state['power_pellets'])
        if all_food:
            nearest_food = min(all_food, key=lambda p: abs(p[0] - state['pacman'][0]) + abs(p[1] - state['pacman'][1]))
            food_distance = abs(nearest_food[0] - state['pacman'][0]) + abs(nearest_food[1] - state['pacman'][1])
        else:
            food_distance = 0
        
        goal_metrics.append(food_distance)
        
        # Execute step
        next_state, reward, done = game.step(pacman_action, ghost_actions)
        reward_history.append(reward)
        
        state = next_state
        step += 1
        
        if done:
            break
    
    # Analyze results
    print(f"\nResults after {step} steps:")
    print(f"Episode terminated naturally: {done}")
    
    if done:
        print(f"Final reward: {reward}")
        if reward > 500:
            print("[SUCCESS] Positive outcome achieved!")
        elif reward < -500:
            print("[FAIL] Negative outcome (survival penalty working)")
        else:
            print("[WARNING] Neutral outcome")
    
    # Test survival penalty effectiveness
    high_survival_distances = sum(1 for d in survival_metrics if d > 15)
    survival_penalty_rate = high_survival_distances / len(survival_metrics) if survival_metrics else 0
    
    print(f"\nSurvival Behavior Analysis:")
    print(f"  Steps with excessive ghost distance (>15): {high_survival_distances}/{len(survival_metrics)} ({survival_penalty_rate:.1%})")
    
    if survival_penalty_rate > 0.2:
        print("[SUCCESS] Survival penalties are being triggered!")
    else:
        print("[WARNING] Low survival penalty rate - may need adjustment")
    
    # Test goal completion efficiency
    final_pellets = len(state['pellets']) + len(state['power_pellets'])
    pellets_collected = initial_pellet_total - final_pellets
    
    print(f"\nGoal Completion Analysis:")
    print(f"  Pellets collected: {pellets_collected}")
    print(f"  Collection rate: {pellets_collected/step:.3f} per step")
    
    if pellets_collected/step > 0.1:
        print("[SUCCESS] Good goal completion rate!")
    else:
        print("[WARNING] Poor goal completion - may indicate survival focus")
    
    # Analyze reward distribution
    positive_rewards = sum(1 for r in reward_history if r > 0)
    negative_rewards = sum(1 for r in reward_history if r < 0)
    
    print(f"\nReward Distribution:")
    print(f"  Positive rewards: {positive_rewards}/{len(reward_history)} ({positive_rewards/len(reward_history):.1%})")
    print(f"  Negative rewards: {negative_rewards}/{len(reward_history)} ({negative_rewards/len(reward_history):.1%})")
    
    if negative_rewards > positive_rewards:
        print("[SUCCESS] Strong penalty pressure discouraging survival behavior!")
    else:
        print("[WARNING] Reward balance may need adjustment")
    
    return {
        'episode_length': step,
        'survival_penalty_rate': survival_penalty_rate,
        'pellets_collected': pellets_collected,
        'collection_rate': pellets_collected/step,
        'positive_reward_rate': positive_rewards/len(reward_history),
        'negative_reward_rate': negative_rewards/len(reward_history),
        'final_outcome': reward if 'reward' in locals() else 0
    }

def test_episode_termination():
    """Test that episode termination penalties work correctly"""
    
    print(f"\n=== TESTING EPISODE TERMINATION FIXES ===")
    print("Validating that episodes terminate appropriately...")
    
    pacman_agent = PacmanAgent()
    ghost_team = GhostTeam()
    game = PacmanGame()
    
    state = game.reset()
    step = 0
    
    print("Testing extended episode behavior...")
    
    # Test with a fresh agent that might try to survive
    while step < 350:  # Should terminate before 350 due to penalties
        pacman_state = pacman_agent.get_state_repr(state)
        pacman_action = pacman_agent.get_action(pacman_state)
        ghost_actions = ghost_team.get_actions(state)
        
        next_state, reward, done = game.step(pacman_action, ghost_actions)
        
        state = next_state
        step += 1
        
        if done:
            break
    
    print(f"Episode terminated at step {step}")
    
    if step <= 300:
        print("[SUCCESS] Episode terminated appropriately due to length penalties!")
    elif step <= 500:
        print("[WARNING] Episode terminated but may be running long")
    else:
        print("[FAIL] Episode running too long - penalties may not be working")
    
    print(f"Final reward: {reward if 'reward' in locals() else 'N/A'}")
    print(f"Termination reason: {'Natural' if done and step < 300 else 'Length penalty' if done else 'Timeout'}")
    
    return {
        'episode_length': step,
        'terminated_early': step < 350,
        'final_reward': reward if 'reward' in locals() else 0
    }

def main():
    """Run comprehensive test of all fixes"""
    
    print("AI Pacman RL Reward Structure Fixes Validation")
    print("Testing survival penalty, goal incentives, and exploration fixes...")
    
    # Test each fix
    reward_test = test_reward_fixes()
    termination_test = test_episode_termination()
    
    # Overall assessment
    print(f"\n=== OVERALL ASSESSMENT ===")
    
    fixes_working = 0
    total_fixes = 3
    
    # Check if survival penalties are working
    if reward_test['survival_penalty_rate'] > 0.1:
        print("[SUCCESS] Survival penalties: WORKING")
        fixes_working += 1
    else:
        print("[FAIL] Survival penalties: NEED ADJUSTMENT")
    
    # Check if goal completion is encouraged
    if reward_test['collection_rate'] > 0.05:
        print("[SUCCESS] Goal completion: WORKING")
        fixes_working += 1
    else:
        print("[WARNING] Goal completion: NEEDS MONITORING")
    
    # Check if episode termination works
    if termination_test['terminated_early']:
        print("[SUCCESS] Episode termination: WORKING")
        fixes_working += 1
    else:
        print("[FAIL] Episode termination: NEED ADJUSTMENT")
    
    print(f"\nFixes Status: {fixes_working}/{total_fixes} working correctly")
    
    if fixes_working >= 2:
        print("[SUCCESS] Most fixes are working! Training should improve.")
    elif fixes_working >= 1:
        print("[WARNING] PARTIAL SUCCESS: Some fixes working, may need fine-tuning.")
    else:
        print("[FAIL] NEEDS WORK: Multiple fixes require adjustment.")
    
    # Save results
    results = {
        'reward_test': reward_test,
        'termination_test': termination_test,
        'overall_score': fixes_working / total_fixes,
        'timestamp': __import__('time').time()
    }
    
    with open('reward_fixes_validation.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nTest results saved to reward_fixes_validation.json")

if __name__ == "__main__":
    main()