#!/usr/bin/env python3
"""
CRITICAL EPSILON DECAY VALIDATION TEST
=====================================

This test validates that the epsilon decay schedule fix is working correctly.
Before the fix: epsilon was decaying every step, causing premature convergence.
After the fix: epsilon should decay only at episode boundaries (when done=True).

Expected behavior:
- Epsilon should remain constant throughout an episode
- Epsilon should decay only when an episode ends (done=True)
- Exploration boost should still work every 1000 episodes
- No premature convergence to minimum values in early episodes
"""

import numpy as np
import sys
import os
sys.path.append('.')

from agent import QMIXAgent, GHOST_STATE_DIM, GLOBAL_STATE_DIM
from game import PacmanGame

class EpsilonDecayValidator:
    def __init__(self):
        self.qmix = QMIXAgent(n_agents=4, state_dim=GHOST_STATE_DIM, global_state_dim=GLOBAL_STATE_DIM, 
                             epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.1)
        self.episode_count = 0
        self.step_count = 0
        self.test_results = []
        
    def simulate_episode(self, steps_in_episode=50):
        """Simulate a complete episode with multiple steps"""
        print(f"\n=== SIMULATING EPISODE {self.episode_count + 1} ===")
        
        initial_epsilon = self.qmix.epsilon
        print(f"Starting epsilon: {initial_epsilon:.4f}")
        
        # Fill buffer with enough experiences to trigger updates
        experiences_needed = self.qmix.batch_size
        experiences_added = 0
        
        # Simulate multiple steps within the episode
        for step in range(steps_in_episode):
            self.step_count += 1
            
            # Simulate some training experience (non-terminal steps)
            states = [[0.1] * GHOST_STATE_DIM for _ in range(4)]
            actions = [0, 1, 2, 3]
            reward = -0.1
            next_states = [[0.2] * GHOST_STATE_DIM for _ in range(4)]
            done = False  # Not done yet
            
            global_state = [0.1] * GLOBAL_STATE_DIM
            next_global_state = [0.1] * GLOBAL_STATE_DIM
            
            # Store transition
            self.qmix.store_transition(states, actions, reward, next_states, done, 
                                     global_state, next_global_state)
            experiences_added += 1
            
            # Update QMIX periodically to test intermediate updates
            if experiences_added >= self.qmix.batch_size and experiences_added % 10 == 0:
                # Update QMIX (should NOT decay epsilon for non-terminal steps)
                old_epsilon = self.qmix.epsilon
                self.qmix.update(done=done)
                
                # Check epsilon hasn't changed during episode
                current_epsilon = self.qmix.epsilon
                if current_epsilon != old_epsilon:
                    print(f"ERROR: Epsilon changed during episode! Expected {old_epsilon:.4f}, got {current_epsilon:.4f}")
                    self.test_results.append("FAIL: Epsilon changed during episode")
                    return False
                
        print(f"Epsilon after {steps_in_episode} steps: {self.qmix.epsilon:.4f}")
        
        # Now end the episode (terminal step)
        print("Ending episode...")
        
        states = [[0.1] * GHOST_STATE_DIM for _ in range(4)]
        actions = [0, 1, 2, 3]
        reward = -10.0
        next_states = [[0.2] * GHOST_STATE_DIM for _ in range(4)]
        done = True  # Episode ends here
        
        global_state = [0.1] * GLOBAL_STATE_DIM
        next_global_state = [0.1] * GLOBAL_STATE_DIM
        
        # Store transition
        self.qmix.store_transition(states, actions, reward, next_states, done, 
                                 global_state, next_global_state)
        
        # Update QMIX (should decay epsilon now)
        old_epsilon = self.qmix.epsilon
        self.qmix.update(done=done)
        
        final_epsilon = self.qmix.epsilon
        expected_decayed_epsilon = max(self.qmix.epsilon_min, initial_epsilon * self.qmix.epsilon_decay)
        
        print(f"Expected epsilon after episode: {expected_decayed_epsilon:.4f}")
        print(f"Actual epsilon after episode: {final_epsilon:.4f}")
        print(f"Epsilon before update: {old_epsilon:.4f}")
        
        if abs(final_epsilon - expected_decayed_epsilon) > 1e-6:
            print(f"ERROR: Epsilon decay incorrect!")
            self.test_results.append("FAIL: Incorrect epsilon decay at episode end")
            return False
        else:
            print("SUCCESS: Epsilon decayed correctly at episode boundary")
            self.test_results.append("PASS: Epsilon decayed correctly at episode boundary")
            
        self.episode_count += 1
        return True
    
    def test_exploration_boost(self):
        """Test that exploration boost still works every 1000 episodes"""
        print(f"\n=== TESTING EXPLORATION BOOST ===")
        
        # Fill buffer first to ensure updates work
        for _ in range(self.qmix.batch_size):
            states = [[0.1] * GHOST_STATE_DIM for _ in range(4)]
            actions = [0, 1, 2, 3]
            reward = -10.0
            next_states = [[0.2] * GHOST_STATE_DIM for _ in range(4)]
            done = False
            
            self.qmix.store_transition(states, actions, reward, next_states, done, 
                                     [0.1] * GLOBAL_STATE_DIM, [0.1] * GLOBAL_STATE_DIM)
        
        # Set episode count to 999
        self.qmix.episode_count = 999
        self.qmix.epsilon = 0.15  # Low epsilon
        
        # Add some experience and trigger update with done=True
        states = [[0.1] * GHOST_STATE_DIM for _ in range(4)]
        actions = [0, 1, 2, 3]
        reward = -10.0
        next_states = [[0.2] * GHOST_STATE_DIM for _ in range(4)]
        done = True
        
        self.qmix.store_transition(states, actions, reward, next_states, done, 
                                 [0.1] * GLOBAL_STATE_DIM, [0.1] * GLOBAL_STATE_DIM)
        self.qmix.update(done=done)
        
        # Check if exploration boost was applied
        expected_epsilon = min(0.3, 0.15 + 0.1)  # min(0.3, 0.25) = 0.25
        
        print(f"Expected epsilon after boost: {expected_epsilon:.4f}")
        print(f"Actual epsilon after boost: {self.qmix.epsilon:.4f}")
        
        if abs(self.qmix.epsilon - expected_epsilon) < 1e-6:
            print("SUCCESS: Exploration boost working correctly")
            self.test_results.append("PASS: Exploration boost working correctly")
            return True
        else:
            print("ERROR: Exploration boost not working!")
            self.test_results.append("FAIL: Exploration boost not working")
            return False
    
    def run_comprehensive_test(self):
        """Run comprehensive validation test"""
        print("CRITICAL EPSILON DECAY VALIDATION TEST")
        print("=" * 50)
        
        # Test 1: Multiple episodes with steps in between
        print(f"\nRunning {5} test episodes...")
        for i in range(5):
            success = self.simulate_episode(steps_in_episode=20 + i * 10)
            if not success:
                print(f"Test failed at episode {i+1}")
                return False
        
        # Test 2: Exploration boost
        boost_success = self.test_exploration_boost()
        
        # Test 3: Minimum epsilon floor
        print(f"\n=== TESTING MINIMUM EPSILON FLOOR ===")
        self.qmix.episode_count = 1000  # Set high episode count
        
        # Fill buffer first
        for _ in range(self.qmix.batch_size):
            states = [[0.1] * GHOST_STATE_DIM for _ in range(4)]
            actions = [0, 1, 2, 3]
            reward = -10.0
            next_states = [[0.2] * GHOST_STATE_DIM for _ in range(4)]
            done = False
            
            self.qmix.store_transition(states, actions, reward, next_states, done, 
                                     [0.1] * GLOBAL_STATE_DIM, [0.1] * GLOBAL_STATE_DIM)
        
        # Set epsilon very low
        self.qmix.epsilon = 0.15
        
        # Run many episodes to test floor
        for _ in range(50):
            states = [[0.1] * GHOST_STATE_DIM for _ in range(4)]
            actions = [0, 1, 2, 3]
            reward = -10.0
            next_states = [[0.2] * GHOST_STATE_DIM for _ in range(4)]
            done = True
            
            self.qmix.store_transition(states, actions, reward, next_states, done, 
                                     [0.1] * GLOBAL_STATE_DIM, [0.1] * GLOBAL_STATE_DIM)
            self.qmix.update(done=done)
        
        final_epsilon = self.qmix.epsilon
        print(f"Final epsilon after many episodes: {final_epsilon:.4f}")
        print(f"Expected minimum epsilon: {self.qmix.epsilon_min}")
        
        if final_epsilon >= self.qmix.epsilon_min - 1e-6:
            print("SUCCESS: Minimum epsilon floor working correctly")
            self.test_results.append("PASS: Minimum epsilon floor working correctly")
            floor_success = True
        else:
            print("ERROR: Minimum epsilon floor not working!")
            self.test_results.append("FAIL: Minimum epsilon floor not working")
            floor_success = False
        
        # Summary
        print(f"\n=== TEST RESULTS SUMMARY ===")
        for result in self.test_results:
            print(f"  {result}")
        
        all_passed = all("PASS" in result for result in self.test_results)
        
        if all_passed:
            print("\n🎉 ALL TESTS PASSED! Epsilon decay fix is working correctly!")
            print("✅ Epsilon now decays at episode boundaries, not step boundaries")
            print("✅ No premature convergence to minimum values")
            print("✅ Exploration boost mechanism preserved")
            print("✅ Minimum epsilon floor working correctly")
        else:
            print("\n❌ SOME TESTS FAILED! There may be issues with the epsilon decay fix.")
        
        return all_passed

if __name__ == "__main__":
    validator = EpsilonDecayValidator()
    success = validator.run_comprehensive_test()
    
    if success:
        print("\n" + "="*50)
        print("🎯 EPSILON DECAY FIX VALIDATION: SUCCESS")
        print("The critical learning dynamics issue has been resolved!")
        print("Agents will now maintain proper exploration throughout training.")
        print("="*50)
        sys.exit(0)
    else:
        print("\n" + "="*50)
        print("❌ EPSILON DECAY FIX VALIDATION: FAILED")
        print("The fix needs additional attention.")
        print("="*50)
        sys.exit(1)