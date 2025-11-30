#!/usr/bin/env python3
"""
Simple debug test for epsilon decay fix
"""

import sys
sys.path.append('.')

from agent import QMIXAgent, GHOST_STATE_DIM, GLOBAL_STATE_DIM

def test_epsilon_decay_simple():
    print("=== SIMPLE EPSILON DECAY DEBUG TEST ===")
    
    # Create QMIX agent with specific parameters
    qmix = QMIXAgent(n_agents=4, state_dim=GHOST_STATE_DIM, global_state_dim=GLOBAL_STATE_DIM, 
                     epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.1)
    
    print(f"Initial epsilon: {qmix.epsilon}")
    print(f"Epsilon decay rate: {qmix.epsilon_decay}")
    print(f"Minimum epsilon: {qmix.epsilon_min}")
    print(f"Batch size: {qmix.batch_size}")
    print(f"Current buffer size: {len(qmix.memory)}")
    
    # Fill buffer with enough experiences
    print(f"\nFilling buffer with {qmix.batch_size} experiences...")
    for i in range(qmix.batch_size):
        states = [[0.1] * GHOST_STATE_DIM for _ in range(4)]
        actions = [0, 1, 2, 3]
        reward = -1.0
        next_states = [[0.2] * GHOST_STATE_DIM for _ in range(4)]
        done = (i == qmix.batch_size - 1)  # Last experience ends episode
        
        global_state = [0.1] * GLOBAL_STATE_DIM
        next_global_state = [0.1] * GLOBAL_STATE_DIM
        
        qmix.store_transition(states, actions, reward, next_states, done, 
                            global_state, next_global_state)
        
        print(f"  Added experience {i+1}/{qmix.batch_size}, done={done}")
    
    print(f"\nBuffer size after filling: {len(qmix.memory)}")
    
    # Test update with done=False (should NOT decay epsilon)
    print(f"\n--- Testing update with done=False ---")
    print(f"Epsilon before update: {qmix.epsilon}")
    
    qmix.update(done=False)
    
    print(f"Epsilon after update (done=False): {qmix.epsilon}")
    
    # Reset buffer and test update with done=True (should decay epsilon)
    print(f"\n--- Testing update with done=True ---")
    print(f"Epsilon before update: {qmix.epsilon}")
    expected_after_decay = max(qmix.epsilon_min, qmix.epsilon * qmix.epsilon_decay)
    print(f"Expected epsilon after decay: {expected_after_decay}")
    
    qmix.update(done=True)
    
    print(f"Epsilon after update (done=True): {qmix.epsilon}")
    
    # Check if decay worked
    if abs(qmix.epsilon - expected_after_decay) < 1e-6:
        print("SUCCESS: Epsilon decay worked correctly!")
        return True
    else:
        print(f"FAILURE: Epsilon decay failed! Expected {expected_after_decay}, got {qmix.epsilon}")
        return False

if __name__ == "__main__":
    success = test_epsilon_decay_simple()
    if success:
        print("\nEpsilon decay fix is working!")
    else:
        print("\nEpsilon decay fix has issues!")