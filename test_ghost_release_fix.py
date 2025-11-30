"""
VALIDATION TEST: Ghost Release Fix
Confirms all ghosts now participate in training within reasonable timeframes
"""

import sys
import os
sys.path.append('.')

from game_fixed import PacmanGame
from agent import PacmanAgent, GhostTeam
import numpy as np

def test_ghost_release_fix():
    """Test that the ghost release fix ensures all ghosts participate in training"""
    print("=== GHOST RELEASE FIX VALIDATION ===\n")
    
    game = PacmanGame()
    state = game.reset()
    
    print("Testing fixed ghost release thresholds...")
    print("Expected: All ghosts released within 100 steps")
    print()
    
    # Track release progression
    release_timeline = []
    
    for step in range(150):  # Extended test to ensure all ghosts release
        # Check pre-step state
        pre_state = {
            'released': game.ghost_released.copy(),
            'in_house': game.ghost_in_house.copy(),
            'pellets_eaten': game.pellets_eaten,
            'game_timer': game.game_timer
        }
        
        # Update ghost release
        game._update_ghost_release()
        
        # Check for changes
        if pre_state['released'] != game.ghost_released:
            for i in range(4):
                if not pre_state['released'][i] and game.ghost_released[i]:
                    release_timeline.append({
                        'step': step,
                        'ghost_id': i,
                        'pellets_eaten': game.pellets_eaten,
                        'game_timer': game.game_timer
                    })
                    ghost_name = ['Blinky', 'Pinky', 'Inky', 'Clyde'][i]
                    print(f"  *** Ghost {i} ({ghost_name}) RELEASED at step {step}")
        
        # Progress updates
        if step % 25 == 0 or step < 10:
            released_count = sum(game.ghost_released)
            print(f"Step {step:3d}: {released_count}/4 ghosts released, Pellets: {game.pellets_eaten}, Timer: {game.game_timer}")
        
        game.game_timer += 1
        game.steps = step
        
        # Early termination if all ghosts released
        if all(game.ghost_released):
            print(f"\n*** SUCCESS: All ghosts released by step {step}!")
            break
    
    print(f"\nRELEASE TIMELINE SUMMARY:")
    if release_timeline:
        for event in release_timeline:
            ghost_name = ['Blinky', 'Pinky', 'Inky', 'Clyde'][event['ghost_id']]
            print(f"  Ghost {event['ghost_id']} ({ghost_name}): Step {event['step']} (pellets: {event['pellets_eaten']}, timer: {event['game_timer']})")
    else:
        print("  No release events detected")
    
    # Validate fix effectiveness
    final_released = game.ghost_released
    max_release_step = max([event['step'] for event in release_timeline]) if release_timeline else 0
    
    print(f"\nFIX VALIDATION RESULTS:")
    print(f"  All ghosts released: {all(final_released)}")
    print(f"  Max release time: {max_release_step} steps")
    print(f"  Fix success criteria:")
    print(f"    OK Ghost 0 (Blinky): Released immediately = {final_released[0]}")
    print(f"    OK Ghost 1 (Pinky): Released quickly = {final_released[1]}")
    print(f"    OK Ghost 2 (Inky): Released within 100 steps = {final_released[2] and max_release_step <= 100}")
    print(f"    OK Ghost 3 (Clyde): Released within 100 steps = {final_released[3] and max_release_step <= 100}")
    
    # Success判定
    success = all(final_released) and max_release_step <= 100
    print(f"\nOVERALL RESULT: {'SUCCESS' if success else 'FAILURE'}")
    
    if success:
        print("*** FIX CONFIRMED: All ghosts now participate in training within reasonable timeframes!")
    else:
        print("*** FIX FAILED: Some ghosts still take too long to release")
    
    return success

def test_ghost_movement_after_fix():
    """Test that ghosts actually move after being released"""
    print("\n=== GHOST MOVEMENT TEST AFTER FIX ===\n")
    
    game = PacmanGame()
    state = game.reset()
    
    # Force all ghosts released immediately for movement test
    center_x, center_y = game.width // 2, game.height // 2
    for i in range(4):
        game.ghost_released[i] = True
        game.ghost_in_house[i] = False
        game.ghosts[i] = [center_y - 4, center_x]
    
    print("All ghosts released, testing movement...")
    
    ghost_team = GhostTeam()
    
    # Test movement for several steps
    moved_counts = [0] * 4
    total_steps = 20
    
    for step in range(total_steps):
        prev_positions = [list(g) for g in state['ghosts']]
        ghost_actions = ghost_team.get_actions(state)
        
        # Simulate game step (Pacman stays put)
        next_state, reward, done = game.step(0, ghost_actions)
        
        # Count moved ghosts
        for i in range(4):
            if list(next_state['ghosts'][i]) != prev_positions[i]:
                moved_counts[i] += 1
        
        if step < 5:  # Show first few steps
            print(f"Step {step}: Actions={ghost_actions}, Moved={[i for i in range(4) if list(next_state['ghosts'][i]) != prev_positions[i]]}")
        
        state = next_state
    
    print(f"\nMOVEMENT ANALYSIS (over {total_steps} steps):")
    for i in range(4):
        movement_rate = moved_counts[i] / total_steps * 100
        ghost_name = ['Blinky', 'Pinky', 'Inky', 'Clyde'][i]
        print(f"  Ghost {i} ({ghost_name}): {moved_counts[i]}/{total_steps} moves ({movement_rate:.1f}%)")
        
        if movement_rate < 20:
            print(f"    WARNING: Low movement rate detected")
        elif movement_rate > 80:
            print(f"    INFO: High movement rate (normal for random exploration)")
        else:
            print(f"    OK: Moderate movement rate")
    
    # Overall movement success
    avg_movement_rate = sum(moved_counts) / (total_steps * 4) * 100
    print(f"\nAVERAGE MOVEMENT RATE: {avg_movement_rate:.1f}%")
    
    movement_success = avg_movement_rate > 30  # At least 30% movement indicates healthy behavior
    print(f"MOVEMENT RESULT: {'SUCCESS' if movement_success else 'POTENTIAL ISSUE'}")
    
    return movement_success

if __name__ == "__main__":
    print("GHOST RELEASE FIX VALIDATION TEST")
    print("=" * 50)
    
    # Test 1: Ghost release timing
    release_success = test_ghost_release_fix()
    
    print("\n" + "=" * 70 + "\n")
    
    # Test 2: Ghost movement after release
    movement_success = test_ghost_movement_after_fix()
    
    print("\n" + "=" * 80)
    print("FINAL VALIDATION SUMMARY:")
    print(f"  Ghost release fix: {'SUCCESS' if release_success else 'FAILED'}")
    print(f"  Ghost movement test: {'SUCCESS' if movement_success else 'FAILED'}")
    
    overall_success = release_success and movement_success
    print(f"\nOVERALL FIX VALIDATION: {'COMPLETE SUCCESS' if overall_success else 'NEEDS ATTENTION'}")
    
    if overall_success:
        print("\n*** RECOMMENDATION: Ghost release fix successfully resolves paralysis issue!")
        print("*** Training should now include all 4 ghosts from early episodes")
    else:
        print("\n*** WARNING: Fix may need adjustment or additional changes required")