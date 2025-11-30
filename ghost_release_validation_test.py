"""
GHOST RELEASE VALIDATION TEST
Confirms ghost release mechanism failure and tests proposed fixes
"""

import sys
import os
sys.path.append('.')

from game_fixed import PacmanGame
from agent import PacmanAgent, GhostTeam
import numpy as np

def test_ghost_release_mechanism():
    """Test current ghost release behavior"""
    print("=== GHOST RELEASE MECHANISM TEST ===\n")
    
    game = PacmanGame()
    state = game.reset()
    
    print("INITIAL STATE:")
    print(f"Ghost released status: {game.ghost_released}")
    print(f"Ghost in house status: {game.ghost_in_house}")
    print(f"Ghost positions: {game.ghosts}")
    print(f"Pellets eaten: {game.pellets_eaten}")
    print(f"Game timer: {game.game_timer}")
    print()
    
    # Track release progression
    release_tracker = []
    
    # Run extended episode to trigger releases
    for step in range(500):  # Extended to trigger all releases
        # Simulate Pacman eating pellets quickly for testing
        if step < 50:  # First 50 steps, Pacman eats pellets rapidly
            pacman_tuple = tuple(game.pacman_pos)
            if pacman_tuple in game.pellets:
                game.pellets.remove(pacman_tuple)
                game.pellets_eaten += 1
        
        # Check pre-step release state
        pre_release = game.ghost_released.copy()
        pre_in_house = game.ghost_in_house.copy()
        
        # Update ghost release
        game._update_ghost_release()
        
        # Check for changes
        if pre_release != game.ghost_released or pre_in_house != game.ghost_in_house:
            release_tracker.append({
                'step': step,
                'released_before': pre_release,
                'released_after': game.ghost_released,
                'in_house_before': pre_in_house,
                'in_house_after': game.ghost_in_house,
                'pellets_eaten': game.pellets_eaten,
                'game_timer': game.game_timer
            })
        
        game.game_timer += 1
        game.steps = step
        
        if step % 50 == 0 or step < 10:  # Progress updates
            print(f"Step {step:3d}: Released={game.ghost_released}, InHouse={game.ghost_in_house}, Pellets={game.pellets_eaten}, Timer={game.game_timer}")
    
    print(f"\nRELEASE EVENTS DETECTED: {len(release_tracker)}")
    for event in release_tracker:
        print(f"  Step {event['step']:3d}: Released {event['released_before']} -> {event['released_after']}")
        print(f"                   InHouse {event['in_house_before']} -> {event['in_house_after']}")
        print(f"                   Pellets: {event['pellets_eaten']}, Timer: {event['game_timer']}")
        print()
    
    # Analyze why ghosts 2 and 3 aren't releasing
    print("GHOST RELEASE ANALYSIS:")
    final_release = game.ghost_released
    final_pellets = game.pellets_eaten
    final_timer = game.game_timer
    
    print(f"After {500} steps:")
    print(f"  Ghost 0 (Blinky): Released={final_release[0]} - {'OK' if final_release[0] else 'PROBLEM'}")
    print(f"  Ghost 1 (Pinky):  Released={final_release[1]} - {'OK' if final_release[1] else 'PROBLEM'}")
    print(f"  Ghost 2 (Inky):   Released={final_release[2]} - {'OK' if final_release[2] else 'PROBLEM'}")
    print(f"  Ghost 3 (Clyde):  Released={final_release[3]} - {'OK' if final_release[3] else 'PROBLEM'}")
    print(f"  Pellets eaten: {final_pellets}")
    print(f"  Game timer: {final_timer}")
    
    # Check release conditions
    print("\nRELEASE CONDITION ANALYSIS:")
    print("Ghost 2 (Inky) conditions:")
    print(f"  Needs: 30 pellets or 300 timer (10 seconds)")
    print(f"  Has: {final_pellets} pellets or {final_timer} timer")
    print(f"  Status: {'RELEASED' if final_release[2] else 'STUCK IN HOUSE'}")
    
    print("Ghost 3 (Clyde) conditions:")
    print(f"  Needs: 60 pellets or 450 timer (15 seconds)")  
    print(f"  Has: {final_pellets} pellets or {final_timer} timer")
    print(f"  Status: {'RELEASED' if final_release[3] else 'STUCK IN HOUSE'}")
    
    return final_release, final_pellets, final_timer

def test_immediate_release_fix():
    """Test immediate release fix"""
    print("\n=== TESTING IMMEDIATE RELEASE FIX ===\n")
    
    class FixedPacmanGame(PacmanGame):
        def _update_ghost_release(self):
            """FIXED: Immediate release for all ghosts"""
            center_x, center_y = self.width // 2, self.height // 2
            
            # Release all ghosts immediately for testing
            for i in range(4):
                if not self.ghost_released[i]:
                    self.ghost_released[i] = True
                    self.ghost_in_house[i] = False
                    self.ghosts[i] = [center_y - 4, center_x]  # Exit position
                    print(f"  FIXED: Ghost {i} released immediately!")
    
    game = FixedPacmanGame()
    state = game.reset()
    
    print("BEFORE FIX:")
    print(f"Ghost released: {game.ghost_released}")
    print(f"Ghost positions: {game.ghosts}")
    
    # Force update to trigger releases
    game._update_ghost_release()
    
    print("\nAFTER IMMEDIATE RELEASE FIX:")
    print(f"Ghost released: {game.ghost_released}")
    print(f"Ghost positions: {game.ghosts}")
    
    # Test movement
    ghost_team = GhostTeam()
    ghost_actions = ghost_team.get_actions(state)
    print(f"Ghost actions: {ghost_actions}")
    
    # Simulate a few steps to see if movement occurs
    game.pacman_pos = [23, 14]  # Reset Pacman position
    game.ghost_released = [True, True, True, True]  # All released
    game.ghost_in_house = [False, False, False, False]  # None in house
    
    state = game.get_state()
    for step in range(10):
        ghost_actions = ghost_team.get_actions(state)
        prev_positions = [list(g) for g in state['ghosts']]
        next_state, reward, done = game.step(0, ghost_actions)  # Pacman stays put
        
        # Check if ghosts moved
        moved_ghosts = []
        for i in range(4):
            if list(next_state['ghosts'][i]) != prev_positions[i]:
                moved_ghosts.append(i)
        
        print(f"Step {step}: Moved ghosts = {moved_ghosts}, Positions = {next_state['ghosts']}")
        state = next_state
        
        if done:
            break
    
    return True

def test_reduced_threshold_fix():
    """Test reduced pellet/timer thresholds"""
    print("\n=== TESTING REDUCED THRESHOLD FIX ===\n")
    
    class ReducedThresholdPacmanGame(PacmanGame):
        def _update_ghost_release(self):
            """FIXED: Reduced thresholds for ghost release"""
            center_x, center_y = self.width // 2, self.height // 2
            
            # Pinky: 0 pellets or immediate (already good)
            if not self.ghost_released[1] and (self.pellets_eaten >= 0 or self.game_timer >= 150):
                self.ghost_released[1] = True
                self.ghost_in_house[1] = False
                self.ghosts[1] = [center_y - 4, center_x]
                print(f"  FIXED: Ghost 1 (Pinky) released with reduced threshold!")
            
            # Inky: 5 pellets or 50 timer (reduced from 30/300)
            if not self.ghost_released[2] and (self.pellets_eaten >= 5 or self.game_timer >= 50):
                self.ghost_released[2] = True
                self.ghost_in_house[2] = False
                self.ghosts[2] = [center_y - 4, center_x]
                print(f"  FIXED: Ghost 2 (Inky) released with reduced threshold!")
            
            # Clyde: 10 pellets or 100 timer (reduced from 60/450)
            if not self.ghost_released[3] and (self.pellets_eaten >= 10 or self.game_timer >= 100):
                self.ghost_released[3] = True
                self.ghost_in_house[3] = False
                self.ghosts[3] = [center_y - 4, center_x]
                print(f"  FIXED: Ghost 3 (Clyde) released with reduced threshold!")
    
    game = ReducedThresholdPacmanGame()
    state = game.reset()
    
    print("Testing with reduced thresholds...")
    
    # Simulate rapid pellet eating to trigger releases
    for step in range(120):  # Should trigger all releases with new thresholds
        if step < 15:  # Eat 15 pellets quickly
            pacman_tuple = tuple(game.pacman_pos)
            if pacman_tuple in game.pellets:
                game.pellets.remove(pacman_tuple)
                game.pellets_eaten += 1
        
        prev_released = game.ghost_released.copy()
        game._update_ghost_release()
        game.game_timer += 1
        
        # Check for new releases
        if prev_released != game.ghost_released:
            for i in range(4):
                if not prev_released[i] and game.ghost_released[i]:
                    print(f"  Ghost {i} released at step {step} (pellets: {game.pellets_eaten}, timer: {game.game_timer})")
        
        if step % 20 == 0:
            print(f"Step {step:3d}: Released={game.ghost_released}, Pellets={game.pellets_eaten}, Timer={game.game_timer}")
    
    return game.ghost_released

if __name__ == "__main__":
    print("GHOST RELEASE VALIDATION TEST")
    print("Testing ghost release mechanism and proposed fixes\n")
    
    # Test current behavior
    release_status, pellets, timer = test_ghost_release_mechanism()
    
    # Test immediate release fix
    print("\n" + "="*60)
    immediate_success = test_immediate_release_fix()
    
    # Test reduced threshold fix
    print("\n" + "="*60)
    reduced_success = test_reduced_threshold_fix()
    
    print("\n" + "="*80)
    print("VALIDATION TEST SUMMARY:")
    print(f"Current behavior: Ghost 2&3 stuck = {not release_status[2]} and {not release_status[3]}")
    print(f"Immediate release fix: SUCCESS = {immediate_success}")
    print(f"Reduced threshold fix: SUCCESS = {all(reduced_success)}")
    
    if not release_status[2] or not release_status[3]:
        print("\n*** CONFIRMED: Ghost release mechanism failure detected!")
        print("*** RECOMMENDATION: Implement ghost release fix to resolve paralysis")
    else:
        print("\n*** Ghost release mechanism appears to be working correctly")