"""
CRITICAL DEBUG INVESTIGATION: Ghost Agent Paralysis & Epsilon Convergence
Comprehensive diagnostic script to identify root causes of ghost movement issues
"""

import sys
import os
sys.path.append('.')

from game_fixed import PacmanGame
from agent import PacmanAgent, GhostTeam
import numpy as np
import time
from collections import defaultdict

class GhostParalysisDiagnostic:
    def __init__(self):
        self.game = PacmanGame()
        self.pacman_agent = PacmanAgent()
        self.ghost_team = GhostTeam()
        
        # Diagnostic tracking
        self.step_data = []
        self.ghost_movement_stats = defaultdict(list)
        self.epsilon_values = []
        self.house_constraints = defaultdict(list)
        self.release_constraints = defaultdict(list)
        self.vulnerable_states = defaultdict(list)
        
    def run_comprehensive_diagnostic(self):
        """Run comprehensive diagnostic for ghost agent behavior"""
        print("=== GHOST AGENT PARALYSIS DIAGNOSTIC ===")
        print("Investigating ghost movement, epsilon decay, and constraint violations")
        print()
        
        # Reset game
        state = self.game.reset()
        
        print("INITIAL STATE ANALYSIS:")
        print(f"Pacman position: {state['pacman']}")
        print(f"Ghost positions: {state['ghosts']}")
        print(f"Ghost vulnerable: {state['ghost_vulnerable']}")
        print(f"Initial epsilon: {self.ghost_team.qmix.epsilon:.4f}")
        print()
        
        # Track ghost constraints over time
        for step in range(100):  # Extended diagnostic run
            self.game.steps = step  # Manually set steps for testing
            
            # Get ghost constraints
            ghost_released = self.game.ghost_released
            ghost_in_house = self.game.ghost_in_house
            ghost_vulnerable = state['ghost_vulnerable']
            
            # Get actions
            ghost_actions = self.ghost_team.get_actions(state)
            pacman_action = self.pacman_agent.get_action(self.pacman_agent.get_state_repr(state))
            
            # Store constraint data
            for i in range(4):
                self.release_constraints[i].append(ghost_released[i])
                self.house_constraints[i].append(ghost_in_house[i])
                self.vulnerable_states[i].append(ghost_vulnerable[i])
            
            # Store epsilon
            self.epsilon_values.append(self.ghost_team.qmix.epsilon)
            
            # Execute step
            prev_positions = [list(g) for g in state['ghosts']]
            next_state, reward, done = self.game.step(pacman_action, ghost_actions)
            
            # Analyze ghost movement
            for i in range(4):
                prev_pos = prev_positions[i]
                curr_pos = list(next_state['ghosts'][i])
                moved = prev_pos != curr_pos
                
                # Check if movement was allowed
                movement_allowed = ghost_released[i] and not ghost_in_house[i]
                
                self.ghost_movement_stats[i].append({
                    'step': step,
                    'action': ghost_actions[i],
                    'prev_pos': prev_pos,
                    'curr_pos': curr_pos,
                    'moved': moved,
                    'movement_allowed': movement_allowed,
                    'vulnerable': ghost_vulnerable[i],
                    'epsilon': self.ghost_team.qmix.epsilon,
                    'reward': reward
                })
            
            state = next_state
            
            if done:
                break
        
        # Run analysis
        self.analyze_results()
        
    def analyze_results(self):
        """Analyze the collected diagnostic data"""
        print("=== DIAGNOSTIC RESULTS ANALYSIS ===\n")
        
        # 1. Ghost Movement Analysis
        print("1. GHOST MOVEMENT ANALYSIS:")
        for i in range(4):
            ghost_data = self.ghost_movement_stats[i]
            if not ghost_data:
                continue
                
            total_steps = len(ghost_data)
            actual_moves = sum(1 for d in ghost_data if d['moved'])
            allowed_moves = sum(1 for d in ghost_data if d['movement_allowed'])
            constrained_steps = sum(1 for d in ghost_data if not d['movement_allowed'])
            vulnerable_steps = sum(1 for d in ghost_data if d['vulnerable'])
            
            print(f"  Ghost {i}:")
            print(f"    Total steps: {total_steps}")
            print(f"    Actual moves: {actual_moves} ({actual_moves/total_steps*100:.1f}%)")
            print(f"    Movement allowed: {allowed_moves} ({allowed_moves/total_steps*100:.1f}%)")
            print(f"    Constrained (house/not released): {constrained_steps} ({constrained_steps/total_steps*100:.1f}%)")
            print(f"    Vulnerable steps: {vulnerable_steps} ({vulnerable_steps/total_steps*100:.1f}%)")
            
            # Check for paralysis
            if actual_moves == 0 and allowed_moves > 10:
                print(f"    *** CRITICAL: Ghost {i} exhibits paralysis despite movement allowance!")
            
        print()
        
        # 2. Epsilon Decay Analysis
        print("2. EPSILON DECAY ANALYSIS:")
        initial_epsilon = self.epsilon_values[0] if self.epsilon_values else 1.0
        final_epsilon = self.epsilon_values[-1] if self.epsilon_values else 0.0
        print(f"  Initial epsilon: {initial_epsilon:.4f}")
        print(f"  Final epsilon: {final_epsilon:.4f}")
        print(f"  Decay over {len(self.epsilon_values)} steps: {((final_epsilon - initial_epsilon) / initial_epsilon * 100):.1f}%")
        
        # Calculate decay rate
        if len(self.epsilon_values) > 1 and self.epsilon_values[0] > 0:
            decay_rate = (self.epsilon_values[-1] / self.epsilon_values[0]) ** (1.0 / len(self.epsilon_values))
            print(f"  Effective decay rate: {decay_rate:.6f}")
            
            # Project when epsilon will reach minimum
            epsilon_min = self.ghost_team.qmix.epsilon_min
            if final_epsilon > epsilon_min and decay_rate < 1.0:
                steps_to_min = np.log(epsilon_min / initial_epsilon) / np.log(decay_rate)
                print(f"  Projected steps to epsilon_min: {steps_to_min:.0f}")
        
        # Check for low epsilon periods
        low_epsilon_steps = sum(1 for eps in self.epsilon_values if eps < 0.1)
        if low_epsilon_steps > 0:
            print(f"  WARNING: {low_epsilon_steps} steps with epsilon < 0.1 (exploration limited)")
        
        print()
        
        # 3. Constraint Analysis
        print("3. CONSTRAINT ANALYSIS:")
        for i in range(4):
            release_data = self.release_constraints[i]
            house_data = self.house_constraints[i]
            
            if not release_data:
                continue
                
            released_steps = sum(release_data)
            house_steps = sum(house_data)
            total_steps = len(release_data)
            
            print(f"  Ghost {i}:")
            print(f"    Released: {released_steps}/{total_steps} steps ({released_steps/total_steps*100:.1f}%)")
            print(f"    In house: {house_steps}/{total_steps} steps ({house_steps/total_steps*100:.1f}%)")
            
            # Check for prolonged constraints
            if house_steps > total_steps * 0.5:
                print(f"    *** CRITICAL: Ghost {i} spends >50% time in house!")
        
        print()
        
        # 4. Action Selection Analysis
        print("4. ACTION SELECTION ANALYSIS:")
        for i in range(4):
            ghost_data = self.ghost_movement_stats[i]
            if not ghost_data:
                continue
                
            actions = [d['action'] for d in ghost_data]
            unique_actions = set(actions)
            action_counts = np.bincount(actions, minlength=4)
            
            print(f"  Ghost {i}:")
            print(f"    Actions used: {len(unique_actions)}/4")
            print(f"    Action distribution: {action_counts}")
            print(f"    Epsilon-based exploration: {sum(1 for d in ghost_data if d['epsilon'] > 0.1)} steps")
            
            # Check for action diversity
            if len(unique_actions) < 3:
                print(f"    WARNING: Low action diversity - may indicate local optima")
        
        print()
        
        # 5. Root Cause Identification
        print("5. ROOT CAUSE ANALYSIS:")
        
        # Check for house constraints as primary cause
        total_house_steps = sum(sum(self.house_constraints[i]) for i in range(4))
        total_steps = len(self.epsilon_values) * 4
        house_percentage = (total_house_steps / total_steps * 100) if total_steps > 0 else 0
        
        if house_percentage > 30:
            print(f"  *** PRIMARY SUSPECT: Ghost house constraints ({house_percentage:.1f}% of ghost-steps constrained)")
            print(f"     -> Ghosts are stuck in house and cannot move regardless of actions")
        
        # Check for epsilon convergence
        final_eps = self.epsilon_values[-1] if self.epsilon_values else 1.0
        if final_eps <= self.ghost_team.qmix.epsilon_min + 0.01:
            print(f"  *** SECONDARY SUSPECT: Epsilon convergence (final epsilon: {final_eps:.3f})")
            print(f"     -> Agents have stopped exploring and rely on learned policies")
        
        # Check for reward penalties causing defensive behavior
        rewards = [d['reward'] for d in self.ghost_movement_stats[0]]  # Check first ghost
        if rewards and min(rewards) < -10:
            print(f"  *** TERTIARY SUSPECT: Excessive penalties (min reward: {min(rewards):.1f})")
            print(f"     -> Agents may be learning to minimize movement to avoid penalties")
        
        print("\n" + "="*60)
        
    def run_epsilon_stress_test(self):
        """Test epsilon decay under different scenarios"""
        print("=== EPSILON STRESS TEST ===")
        
        # Test current decay rate
        print("Testing current epsilon decay parameters:")
        print(f"  Decay rate: {self.ghost_team.qmix.epsilon_decay}")
        print(f"  Minimum epsilon: {self.ghost_team.qmix.epsilon_min}")
        print()
        
        # Simulate training progression
        epsilon = 1.0
        decay_rate = self.ghost_team.qmix.epsilon_decay
        epsilon_min = self.ghost_team.qmix.epsilon_min
        
        epochs_to_test = [1, 10, 50, 100, 200, 500, 1000, 2000, 5000]
        
        print("Epsilon progression over training epochs:")
        for epoch in epochs_to_test:
            current_epsilon = max(epsilon_min, epsilon * (decay_rate ** epoch))
            exploration_level = "HIGH" if current_epsilon > 0.5 else "MODERATE" if current_epsilon > 0.2 else "LOW" if current_epsilon > epsilon_min else "ZERO"
            print(f"  Epoch {epoch:4d}: ε = {current_epsilon:.4f} ({exploration_level})")
        
        print()
        
        # Calculate when exploration becomes problematic
        low_exploration_threshold = 0.1
        if decay_rate < 1.0:
            epochs_to_low_exploration = np.log(low_exploration_threshold) / np.log(decay_rate)
            print(f"  WARNING: Low exploration (ε < {low_exploration_threshold}) starts at epoch {epochs_to_low_exploration:.0f}")
            
            epochs_to_minimum = np.log(epsilon_min) / np.log(decay_rate)
            print(f"  CRITICAL: Zero exploration (ε = {epsilon_min}) reached at epoch {epochs_to_minimum:.0f}")

if __name__ == "__main__":
    diagnostic = GhostParalysisDiagnostic()
    
    print("Starting comprehensive ghost agent paralysis investigation...")
    print("This will analyze movement constraints, epsilon decay, and behavioral patterns.\n")
    
    # Run main diagnostic
    diagnostic.run_comprehensive_diagnostic()
    
    print("\n" + "="*80 + "\n")
    
    # Run epsilon stress test
    diagnostic.run_epsilon_stress_test()
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE - Review findings above for root cause identification")