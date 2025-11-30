#!/usr/bin/env python3
"""
Episode Termination Investigation Tool
Analyzes why agents run for 5000+ steps instead of reaching natural game endings
"""

import numpy as np
from game import PacmanGame
from agent import PacmanAgent, GhostTeam
import json
import time

class EpisodeTerminationInvestigator:
    """Investigates episode termination behavior and evasive strategies"""
    
    def __init__(self):
        self.pacman_agent = PacmanAgent()
        self.ghost_team = GhostTeam()
    
    def investigate_natural_episode_endings(self):
        """Investigate what triggers natural episode endings vs timeouts"""
        
        print("=== NATURAL EPISODE ENDING INVESTIGATION ===")
        print("Analyzing different episode termination scenarios...")
        
        scenarios = [
            ("Pacman Eaten", self.simulate_pacman_death_scenario),
            ("All Pellets Collected", self.simulate_pellet_collection_scenario),
            ("Extended Survival", self.simulate_extended_survival_scenario),
            ("Ghost Strategy", self.simulate_ghost_evasion_scenario)
        ]
        
        results = {}
        
        for scenario_name, scenario_func in scenarios:
            print(f"\n--- {scenario_name} Scenario ---")
            result = scenario_func()
            results[scenario_name] = result
            
            print(f"  Result: {result['outcome']}")
            print(f"  Episode length: {result['episode_length']} steps")
            print(f"  Steps to termination: {result['termination_step']}")
            print(f"  Natural ending: {result['natural_ending']}")
        
        return results
    
    def simulate_pacman_death_scenario(self):
        """Test scenario where Pacman gets caught"""
        
        game = PacmanGame()
        state = game.reset()
        
        # Force Pacman into dangerous positions
        for step in range(100):
            # Get actions but force Pacman toward ghosts
            pacman_state = self.pacman_agent.get_state_repr(state)
            pacman_action = self.pacman_agent.get_action(pacman_state)
            ghost_actions = self.ghost_team.get_actions(state)
            
            # Execute step
            next_state, reward, done = game.step(pacman_action, ghost_actions)
            
            # Force position near ghost if not moving
            if step > 10 and next_state['pacman'] == state['pacman']:
                # Try different action
                forced_actions = [0, 1, 2, 3]
                for forced_action in forced_actions:
                    test_state, test_reward, test_done = game.step(forced_action, ghost_actions)
                    if test_done:
                        return {
                            'outcome': 'Forced death successful',
                            'episode_length': step + 1,
                            'termination_step': step + 1,
                            'natural_ending': False,
                            'reward': test_reward
                        }
            
            state = next_state
            if done:
                break
        
        return {
            'outcome': 'Unable to force death - agents avoiding capture',
            'episode_length': step + 1,
            'termination_step': step + 1,
            'natural_ending': done,
            'reward': reward if 'reward' in locals() else 0
        }
    
    def simulate_pellet_collection_scenario(self):
        """Test scenario focused on pellet collection"""
        
        game = PacmanGame()
        state = game.reset()
        
        total_pellets = len(state['pellets']) + len(state['power_pellets'])
        collected_pellets = 0
        
        for step in range(1000):  # Extended search
            # Get actions focused on pellet collection
            pacman_state = self.pacman_agent.get_state_repr(state)
            pacman_action = self.pacman_agent.get_action(pacman_state)
            ghost_actions = self.ghost_team.get_actions(state)
            
            # Execute step
            next_state, reward, done = game.step(pacman_action, ghost_actions)
            
            # Count collected pellets
            next_total_pellets = len(next_state['pellets']) + len(next_state['power_pellets'])
            if next_total_pellets < total_pellets:
                collected_pellets += (total_pellets - next_total_pellets)
                total_pellets = next_total_pellets
            
            state = next_state
            if done:
                break
        
        return {
            'outcome': 'Pellet collection scenario',
            'episode_length': step + 1,
            'termination_step': step + 1,
            'natural_ending': done,
            'pellets_collected': collected_pellets,
            'pellets_remaining': total_pellets
        }
    
    def simulate_extended_survival_scenario(self):
        """Test extended survival scenario (simulate your 5000+ step issue)"""
        
        game = PacmanGame()
        state = game.reset()
        
        survival_steps = 0
        max_test_steps = 1000  # Test up to 1000 steps (less than your 5000)
        
        position_history = []
        action_history = []
        
        for step in range(max_test_steps):
            # Store position for analysis
            position_history.append(state['pacman'])
            action_history.append(self.pacman_agent.get_action(self.pacman_agent.get_state_repr(state)))
            
            # Get actions
            pacman_state = self.pacman_agent.get_state_repr(state)
            pacman_action = self.pacman_agent.get_action(pacman_state)
            ghost_actions = self.ghost_team.get_actions(state)
            
            # Execute step
            next_state, reward, done = game.step(pacman_action, ghost_actions)
            
            state = next_state
            survival_steps += 1
            
            if done:
                break
        
        # Analyze behavior patterns
        movement_analysis = self.analyze_movement_patterns(position_history, action_history)
        evasion_analysis = self.analyze_evasion_behavior(position_history, state['ghosts'])
        
        return {
            'outcome': f'Survival test: {survival_steps} steps',
            'episode_length': survival_steps,
            'termination_step': survival_steps,
            'natural_ending': done if 'done' in locals() else False,
            'max_steps_reached': survival_steps >= max_test_steps,
            'movement_analysis': movement_analysis,
            'evasion_analysis': evasion_analysis,
            'final_state': {
                'pacman_pos': state['pacman'],
                'ghost_positions': state['ghosts'],
                'pellets_remaining': len(state['pellets']) + len(state['power_pellets'])
            }
        }
    
    def simulate_ghost_evasion_scenario(self):
        """Test ghost evasion strategies"""
        
        game = PacmanGame()
        state = game.reset()
        
        evasion_steps = 0
        ghost_distances = []
        
        for step in range(500):
            # Track distances to ghosts
            pacman_pos = state['pacman']
            ghost_positions = state['ghosts']
            
            distances = [abs(pacman_pos[0] - ghost_pos[0]) + abs(pacman_pos[1] - ghost_pos[1]) 
                        for ghost_pos in ghost_positions]
            ghost_distances.append(min(distances))
            
            # Get actions
            pacman_state = self.pacman_agent.get_state_repr(state)
            pacman_action = self.pacman_agent.get_action(pacman_state)
            ghost_actions = self.ghost_team.get_actions(state)
            
            # Execute step
            next_state, reward, done = game.step(pacman_action, ghost_actions)
            
            state = next_state
            evasion_steps += 1
            
            if done:
                break
        
        # Analyze evasion effectiveness
        avg_distance = np.mean(ghost_distances)
        distance_trend = self.analyze_distance_trend(ghost_distances)
        
        return {
            'outcome': f'Ghost evasion test: {evasion_steps} steps',
            'episode_length': evasion_steps,
            'termination_step': evasion_steps,
            'natural_ending': done if 'done' in locals() else False,
            'avg_ghost_distance': avg_distance,
            'distance_trend': distance_trend,
            'evasion_effectiveness': 'High' if avg_distance > 5 else 'Low'
        }
    
    def analyze_movement_patterns(self, positions, actions):
        """Analyze movement patterns to detect looping/evasive behavior"""
        
        unique_positions = len(set(positions))
        total_positions = len(positions)
        position_reuse = (total_positions - unique_positions) / total_positions
        
        # Detect potential loops
        loop_indicators = []
        for i in range(len(positions) - 10):
            recent_positions = positions[i:i+10]
            if len(set(recent_positions)) < 3:  # Low position diversity
                loop_indicators.append(i)
        
        return {
            'position_diversity': unique_positions / total_positions,
            'position_reuse_rate': position_reuse,
            'potential_loops': len(loop_indicators),
            'loop_locations': loop_indicators[:5]  # First 5 loop indicators
        }
    
    def analyze_evasion_behavior(self, pacman_positions, ghost_positions):
        """Analyze how effectively Pacman evades ghosts"""
        
        if len(pacman_positions) < 2:
            return {'evasion_score': 0, 'strategy': 'Insufficient data'}
        
        # Calculate average distance over time
        distances = []
        for i in range(1, len(pacman_positions)):
            pacman_pos = pacman_positions[i]
            prev_ghost_positions = ghost_positions[i-1] if i > 0 else ghost_positions
            
            min_dist = min(abs(pacman_pos[0] - g[0]) + abs(pacman_pos[1] - g[1]) 
                          for g in prev_ghost_positions)
            distances.append(min_dist)
        
        avg_distance = np.mean(distances)
        distance_trend = np.polyfit(range(len(distances)), distances, 1)[0]  # Linear trend
        
        # Determine strategy
        if avg_distance > 6 and distance_trend > 0:
            strategy = 'Effective evasion'
        elif avg_distance < 3:
            strategy = 'Poor evasion'
        else:
            strategy = 'Balanced behavior'
        
        return {
            'evasion_score': avg_distance,
            'distance_trend': distance_trend,
            'strategy': strategy
        }
    
    def analyze_distance_trend(self, distances):
        """Analyze if Pacman is getting better or worse at avoiding ghosts"""
        
        if len(distances) < 10:
            return 'Insufficient data'
        
        # Calculate trend
        trend = np.polyfit(range(len(distances)), distances, 1)[0]
        
        if trend > 0.1:
            return 'Improving evasion'
        elif trend < -0.1:
            return 'Worsening evasion'
        else:
            return 'Stable evasion'
    
    def investigate_reward_structure_impact(self):
        """Analyze how reward structure affects episode termination"""
        
        print(f"\n=== REWARD STRUCTURE IMPACT ANALYSIS ===")
        
        # Test different reward scenarios
        reward_scenarios = [
            ("High survival reward", self.test_survival_reward_scenario),
            ("High goal reward", self.test_goal_reward_scenario),
            ("Balanced rewards", self.test_balanced_reward_scenario)
        ]
        
        results = {}
        
        for scenario_name, scenario_func in reward_scenarios:
            print(f"\n--- {scenario_name} ---")
            result = scenario_func()
            results[scenario_name] = result
            
            print(f"  Episode length: {result['episode_length']}")
            print(f"  Termination reason: {result['termination_reason']}")
            print(f"  Goal completion: {result.get('goal_completed', 'Unknown')}")
        
        return results
    
    def test_survival_reward_scenario(self):
        """Test with survival-biased rewards"""
        
        game = PacmanGame()
        state = game.reset()
        
        for step in range(500):
            pacman_state = self.pacman_agent.get_state_repr(state)
            pacman_action = self.pacman_agent.get_action(pacman_state)
            ghost_actions = self.ghost_team.get_actions(state)
            
            next_state, reward, done = game.step(pacman_action, ghost_actions)
            
            # Simulate high survival reward bias
            if not done:
                reward = max(reward, 0.1)  # Bias toward survival
            
            state = next_state
            if done:
                break
        
        return {
            'episode_length': step + 1,
            'termination_reason': 'Natural ending' if done else 'Step limit',
            'goal_completed': done and reward > 0
        }
    
    def test_goal_reward_scenario(self):
        """Test with goal-biased rewards"""
        
        game = PacmanGame()
        state = game.reset()
        
        for step in range(500):
            pacman_state = self.pacman_agent.get_state_repr(state)
            pacman_action = self.pacman_agent.get_action(pacman_state)
            ghost_actions = self.ghost_team.get_actions(state)
            
            next_state, reward, done = game.step(pacman_action, ghost_actions)
            
            # Simulate high goal reward bias
            if reward > 0:
                reward = reward * 2  # Double positive rewards
            
            state = next_state
            if done:
                break
        
        return {
            'episode_length': step + 1,
            'termination_reason': 'Natural ending' if done else 'Step limit',
            'goal_completed': done and reward > 0
        }
    
    def test_balanced_reward_scenario(self):
        """Test with balanced rewards (current implementation)"""
        
        game = PacmanGame()
        state = game.reset()
        
        for step in range(500):
            pacman_state = self.pacman_agent.get_state_repr(state)
            pacman_action = self.pacman_agent.get_action(pacman_state)
            ghost_actions = self.ghost_team.get_actions(state)
            
            next_state, reward, done = game.step(pacman_action, ghost_actions)
            
            state = next_state
            if done:
                break
        
        return {
            'episode_length': step + 1,
            'termination_reason': 'Natural ending' if done else 'Step limit',
            'goal_completed': done and reward > 0
        }

def main():
    """Run comprehensive episode termination investigation"""
    
    print("AI Pacman RL Episode Termination Investigation")
    print("Investigating why agents run for 5000+ steps instead of reaching natural endings...")
    
    investigator = EpisodeTerminationInvestigator()
    
    # Run all investigations
    termination_scenarios = investigator.investigate_natural_episode_endings()
    reward_structure_impact = investigator.investigate_reward_structure_impact()
    
    # Summary analysis
    print(f"\n=== INVESTIGATION SUMMARY ===")
    
    print("Episode Termination Patterns:")
    for scenario, result in termination_scenarios.items():
        print(f"  {scenario}: {result['episode_length']} steps ({'Natural' if result['natural_ending'] else 'Forced'})")
    
    print(f"\nReward Structure Impact:")
    for scenario, result in reward_structure_impact.items():
        print(f"  {scenario}: {result['episode_length']} steps")
    
    # Identify key findings
    long_episodes = [r for r in termination_scenarios.values() if r['episode_length'] > 100]
    natural_endings = [r for r in termination_scenarios.values() if r['natural_ending']]
    
    print(f"\nKey Findings:")
    print(f"  Long episodes (>100 steps): {len(long_episodes)}/{len(termination_scenarios)}")
    print(f"  Natural endings: {len(natural_endings)}/{len(termination_scenarios)}")
    
    if len(long_episodes) > len(termination_scenarios) / 2:
        print("  PATTERN DETECTED: Agents tend to run for extended periods")
    
    if len(natural_endings) < len(termination_scenarios) / 2:
        print("  PATTERN DETECTED: Natural episode endings are rare")
    
    # Save results
    results = {
        'termination_scenarios': termination_scenarios,
        'reward_structure_impact': reward_structure_impact,
        'analysis_timestamp': time.time()
    }
    
    with open('episode_termination_investigation.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Investigation results saved to episode_termination_investigation.json")

if __name__ == "__main__":
    main()