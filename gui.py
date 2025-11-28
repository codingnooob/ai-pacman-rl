import tkinter as tk
from tkinter import ttk, filedialog
import threading
import time
from trainer import Trainer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

class PacmanGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Pacman RL Training")
        self.trainer = Trainer()
        self.running = False
        self.play_only = False
        self.batch_training = False
        self.speed = 50  # ms delay
        self.stats_window = None
        self.stats_update_id = None
        
        self._setup_ui()
        
    def _setup_ui(self):
        # Control panel
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.start_btn = ttk.Button(control_frame, text="Start Training", command=self.toggle_training)
        self.start_btn.grid(row=0, column=0, padx=5)
        
        self.play_btn = ttk.Button(control_frame, text="Play Only", command=self.toggle_play_only)
        self.play_btn.grid(row=0, column=1, padx=5)
        
        self.batch_btn = ttk.Button(control_frame, text="Batch Train (10)", command=self.batch_train)
        self.batch_btn.grid(row=0, column=2, padx=5)
        
        self.batch_100_btn = ttk.Button(control_frame, text="Batch Train (100)", command=lambda: self.batch_train(100))
        self.batch_100_btn.grid(row=0, column=3, padx=5)
        
        self.batch_1000_btn = ttk.Button(control_frame, text="Batch Train (1000)", command=lambda: self.batch_train(1000))
        self.batch_1000_btn.grid(row=0, column=4, padx=5)
        
        self.stop_batch_btn = ttk.Button(control_frame, text="Stop Batch", command=self.stop_batch_training, state='disabled')
        self.stop_batch_btn.grid(row=0, column=5, padx=5)
        
        self.load_btn = ttk.Button(control_frame, text="Load Custom Map", command=self.load_custom_map)
        self.load_btn.grid(row=0, column=6, padx=5)
        
        self.save_btn = ttk.Button(control_frame, text="Save Model", command=self.save_model)
        self.save_btn.grid(row=0, column=7, padx=5)
        
        self.load_model_btn = ttk.Button(control_frame, text="Load Model", command=self.load_model)
        self.load_model_btn.grid(row=0, column=8, padx=5)
        
        self.stats_btn = ttk.Button(control_frame, text="Show Stats", command=self.toggle_stats)
        self.stats_btn.grid(row=0, column=9, padx=5)
        
        ttk.Label(control_frame, text="Speed:").grid(row=0, column=10, padx=5)
        self.speed_scale = ttk.Scale(control_frame, from_=1, to=100, orient=tk.HORIZONTAL, 
                                      command=self.update_speed, length=200)
        self.speed_scale.set(50)
        self.speed_scale.grid(row=0, column=11, padx=5)
        
        # Stats panel
        stats_frame = ttk.Frame(self.root, padding="10")
        stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Row 1
        self.episode_label = ttk.Label(stats_frame, text="Episode: 0")
        self.episode_label.grid(row=0, column=0, padx=10)
        
        self.envs_label = ttk.Label(stats_frame, text="Envs: 0")
        self.envs_label.grid(row=0, column=1, padx=10)
        
        self.reward_label = ttk.Label(stats_frame, text="Avg Reward: 0")
        self.reward_label.grid(row=0, column=2, padx=10)
        
        self.last_reward_label = ttk.Label(stats_frame, text="Last: 0")
        self.last_reward_label.grid(row=0, column=3, padx=10)
        
        # Row 2
        self.pacman_win_label = ttk.Label(stats_frame, text="Pacman Wins: 0%")
        self.pacman_win_label.grid(row=1, column=0, padx=10)
        
        self.ghost_win_label = ttk.Label(stats_frame, text="Ghost Wins: 0%")
        self.ghost_win_label.grid(row=1, column=1, padx=10)
        
        self.length_label = ttk.Label(stats_frame, text="Avg Length: 0")
        self.length_label.grid(row=1, column=2, padx=10)
        
        self.pellets_label = ttk.Label(stats_frame, text="Avg Pellets: 0")
        self.pellets_label.grid(row=1, column=3, padx=10)
        
        # Row 3 - Advanced stats
        self.pacman_lr_label = ttk.Label(stats_frame, text="Pacman LR: 0")
        self.pacman_lr_label.grid(row=2, column=0, padx=10)
        
        self.ghost_lr_label = ttk.Label(stats_frame, text="Ghost LR: 0")
        self.ghost_lr_label.grid(row=2, column=1, padx=10)
        
        self.entropy_label = ttk.Label(stats_frame, text="Entropy: 0")
        self.entropy_label.grid(row=2, column=2, padx=10)
        
        self.epsilon_label = ttk.Label(stats_frame, text="Epsilon: 0")
        self.epsilon_label.grid(row=2, column=3, padx=10)
        
        # Game canvas
        self.cell_size = 20
        canvas_width = self.trainer.games[0].width * self.cell_size
        canvas_height = self.trainer.games[0].height * self.cell_size
        self.canvas = tk.Canvas(self.root, width=canvas_width, height=canvas_height, bg='black')
        self.canvas.grid(row=2, column=0, padx=10, pady=10)
        
    def toggle_training(self):
        self.running = not self.running
        if self.running:
            self.start_btn.config(text="Pause Training")
            threading.Thread(target=self.training_loop, daemon=True).start()
        else:
            self.start_btn.config(text="Start Training")
    
    def toggle_play_only(self):
        self.play_only = not self.play_only
        if self.play_only:
            self.play_btn.config(text="Stop Playing")
            threading.Thread(target=self.play_loop, daemon=True).start()
        else:
            self.play_btn.config(text="Play Only")
    
    def play_loop(self):
        """Watch agents play without training"""
        while self.play_only:
            # Get actions but don't update networks
            state = self.trainer.games[0].get_state()
            pacman_state = self.trainer.pacman_agent.get_state_repr(state)
            pacman_action = self.trainer.pacman_agent.get_action(pacman_state)
            ghost_actions = self.trainer.ghost_team.get_actions(state)
            
            # Execute step without training
            next_state, reward, done = self.trainer.games[0].step(pacman_action, ghost_actions)
            
            if done:
                self.trainer.games[0].reset()
            
            self.root.after(0, self.update_display)
            time.sleep(self.speed / 1000.0)
    
    def batch_train(self, episodes=10):
        if not self.running and not self.batch_training:
            threading.Thread(target=self._batch_train_worker, args=(episodes,), daemon=True).start()
    
    def stop_batch_training(self):
        self.batch_training = False
    
    def load_custom_map(self):
        if self.running:
            return
        
        filepath = filedialog.askopenfilename(
            title="Select Map Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                self.trainer = Trainer(custom_map_path=filepath)
                self._update_canvas_size()
                self.update_display()
            except Exception as e:
                print(f"Error loading map: {e}")
    
    def save_model(self):
        if self.running:
            return
        self.trainer.save_checkpoint('manual_save')
        print("Model saved to checkpoints/manual_save.pt")
    
    def load_model(self):
        if self.running:
            return
        try:
            self.trainer.load_checkpoint('manual_save')
            self.update_display()
            print("Model loaded from checkpoints/manual_save.pt")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def toggle_stats(self):
        if self.stats_window is None or not tk.Toplevel.winfo_exists(self.stats_window):
            self._create_stats_window()
            self.stats_btn.config(text="Hide Stats")
        else:
            self._close_stats_window()
            self.stats_btn.config(text="Show Stats")
    
    def _create_stats_window(self):
        self.stats_window = tk.Toplevel(self.root)
        self.stats_window.title("Training Statistics")
        self.stats_window.geometry("1000x800")
        self.stats_window.protocol("WM_DELETE_WINDOW", self._close_stats_window)
        
        # Create figure with subplots
        self.fig = Figure(figsize=(10, 8))
        
        # 4 subplots: rewards, win rates, episode length, pellets
        self.ax_reward = self.fig.add_subplot(2, 2, 1)
        self.ax_winrate = self.fig.add_subplot(2, 2, 2)
        self.ax_length = self.fig.add_subplot(2, 2, 3)
        self.ax_pellets = self.fig.add_subplot(2, 2, 4)
        
        self.fig.tight_layout(pad=3.0)
        
        # Embed in tkinter
        self.canvas_stats = FigureCanvasTkAgg(self.fig, master=self.stats_window)
        self.canvas_stats.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Start updating
        self._update_stats_plots()
    
    def _close_stats_window(self):
        if self.stats_update_id:
            self.root.after_cancel(self.stats_update_id)
            self.stats_update_id = None
        if self.stats_window:
            self.stats_window.destroy()
            self.stats_window = None
    
    def _update_stats_plots(self):
        if self.stats_window is None or not tk.Toplevel.winfo_exists(self.stats_window):
            return
        
        # Get data
        rewards = self.trainer.episode_rewards
        lengths = self.trainer.episode_lengths
        pellets = self.trainer.pellets_collected
        
        if len(rewards) < 2:
            self.stats_update_id = self.root.after(1000, self._update_stats_plots)
            return
        
        episodes = list(range(len(rewards)))
        
        # Calculate win rates over windows
        window = 50
        pacman_wins = []
        ghost_wins = []
        for i in range(window, len(rewards) + 1):
            window_rewards = rewards[i-window:i]
            p_wins = sum(1 for r in window_rewards if r >= 500)  # Approximate win
            g_wins = sum(1 for r in window_rewards if r <= -200)  # Approximate loss
            total = p_wins + g_wins
            pacman_wins.append(p_wins / total * 100 if total > 0 else 0)
            ghost_wins.append(g_wins / total * 100 if total > 0 else 0)
        
        # Moving average for smoothing
        def moving_average(data, window=50):
            if len(data) < window:
                return data
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        # Clear and plot
        self.ax_reward.clear()
        self.ax_reward.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')
        if len(rewards) >= 50:
            smooth_rewards = moving_average(rewards, 50)
            self.ax_reward.plot(range(49, len(rewards)), smooth_rewards, color='blue', linewidth=2, label='MA(50)')
        self.ax_reward.set_title('Episode Rewards')
        self.ax_reward.set_xlabel('Episode')
        self.ax_reward.set_ylabel('Reward')
        self.ax_reward.legend()
        self.ax_reward.grid(True, alpha=0.3)
        
        # Win rates
        self.ax_winrate.clear()
        if len(pacman_wins) > 0:
            win_episodes = list(range(window, len(rewards) + 1))
            self.ax_winrate.plot(win_episodes, pacman_wins, color='green', label='Pacman', linewidth=2)
            self.ax_winrate.plot(win_episodes, ghost_wins, color='red', label='Ghosts', linewidth=2)
            self.ax_winrate.legend()
        self.ax_winrate.set_title('Win Rates (50-episode window)')
        self.ax_winrate.set_xlabel('Episode')
        self.ax_winrate.set_ylabel('Win Rate (%)')
        self.ax_winrate.grid(True, alpha=0.3)
        
        # Episode length
        self.ax_length.clear()
        if lengths:
            self.ax_length.plot(range(len(lengths)), lengths, alpha=0.3, color='purple')
            if len(lengths) >= 50:
                smooth_lengths = moving_average(lengths, 50)
                self.ax_length.plot(range(49, len(lengths)), smooth_lengths, color='purple', linewidth=2)
        self.ax_length.set_title('Episode Length')
        self.ax_length.set_xlabel('Episode')
        self.ax_length.set_ylabel('Steps')
        self.ax_length.grid(True, alpha=0.3)
        
        # Pellets collected
        self.ax_pellets.clear()
        if pellets:
            self.ax_pellets.plot(range(len(pellets)), pellets, alpha=0.3, color='orange')
            if len(pellets) >= 50:
                smooth_pellets = moving_average(pellets, 50)
                self.ax_pellets.plot(range(49, len(pellets)), smooth_pellets, color='orange', linewidth=2)
        self.ax_pellets.set_title('Pellets Collected')
        self.ax_pellets.set_xlabel('Episode')
        self.ax_pellets.set_ylabel('Pellets')
        self.ax_pellets.grid(True, alpha=0.3)
        
        self.canvas_stats.draw()
        
        # Schedule next update
        self.stats_update_id = self.root.after(2000, self._update_stats_plots)
    
    def _update_canvas_size(self):
        self.cell_size = 20
        canvas_width = self.trainer.games[0].width * self.cell_size
        canvas_height = self.trainer.games[0].height * self.cell_size
        self.canvas.config(width=canvas_width, height=canvas_height)
    
    def _batch_train_worker(self, episodes_to_run=10):
        self.batch_training = True
        self.batch_btn.config(state='disabled')
        self.batch_100_btn.config(state='disabled')
        self.batch_1000_btn.config(state='disabled')
        self.stop_batch_btn.config(state='normal')
        
        episodes_completed = 0
        steps_since_stats_update = 0
        
        while episodes_completed < episodes_to_run and self.batch_training:
            done = self.trainer.train_step()
            steps_since_stats_update += 1
            
            # Update stats every 5 steps
            if steps_since_stats_update >= 5:
                self.root.after(0, self.update_stats_only)
                steps_since_stats_update = 0
            
            if done:
                episodes_completed += 1
        
        self.batch_training = False
        self.root.after(0, self.update_display)
        self.batch_btn.config(state='normal')
        self.batch_100_btn.config(state='normal')
        self.batch_1000_btn.config(state='normal')
        self.stop_batch_btn.config(state='disabled')
    
    def update_speed(self, val):
        self.speed = 101 - int(float(val))
    
    def update_stats_only(self):
        """Update only statistics labels without redrawing canvas"""
        stats = self.trainer.get_stats()
        self.episode_label.config(text=f"Episode: {stats['episode']}")
        self.envs_label.config(text=f"Envs: {stats['n_envs']}")
        self.reward_label.config(text=f"Avg Reward: {stats['avg_reward']:.1f}")
        self.last_reward_label.config(text=f"Last: {stats['last_reward']:.0f}")
        self.pacman_win_label.config(text=f"Pacman Wins: {stats['pacman_win_rate']:.1f}%")
        self.ghost_win_label.config(text=f"Ghost Wins: {stats['ghost_win_rate']:.1f}%")
        self.length_label.config(text=f"Avg Length: {stats['avg_length']:.0f}")
        self.pellets_label.config(text=f"Avg Pellets: {stats['avg_pellets']:.0f}")
        self.pacman_lr_label.config(text=f"Pacman LR: {stats['pacman_lr']:.2e}")
        self.ghost_lr_label.config(text=f"Ghost LR: {stats['ghost_lr']:.2e}")
        self.entropy_label.config(text=f"Entropy: {stats['entropy_coef']:.4f}")
        self.epsilon_label.config(text=f"Epsilon: {stats['ghost_epsilon']:.3f}")
    
    def training_loop(self):
        while self.running:
            self.trainer.train_step()
            self.root.after(0, self.update_display)
            time.sleep(self.speed / 1000.0)
    
    def update_display(self):
        self.canvas.delete("all")
        # Display most recently completed environment
        env_id = self.trainer.last_completed_env
        state = self.trainer.games[env_id].get_state()
        
        # Draw walls
        for wall in state['walls']:
            i, j = wall
            self.canvas.create_rectangle(
                j * self.cell_size, i * self.cell_size,
                (j + 1) * self.cell_size, (i + 1) * self.cell_size,
                fill='blue', outline='darkblue'
            )
        
        # Draw pellets
        for pellet in state['pellets']:
            x, y = pellet[1] * self.cell_size + self.cell_size // 2, pellet[0] * self.cell_size + self.cell_size // 2
            self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill='white')
        
        # Draw power pellets
        for pellet in state['power_pellets']:
            x, y = pellet[1] * self.cell_size + self.cell_size // 2, pellet[0] * self.cell_size + self.cell_size // 2
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill='white')
        
        # Draw Pacman
        px, py = state['pacman'][1] * self.cell_size + self.cell_size // 2, state['pacman'][0] * self.cell_size + self.cell_size // 2
        self.canvas.create_oval(px - 8, py - 8, px + 8, py + 8, fill='yellow', outline='orange')
        
        # Draw ghosts
        colors = ['red', 'pink', 'cyan', 'orange']
        for i, ghost in enumerate(state['ghosts']):
            gx, gy = ghost[1] * self.cell_size + self.cell_size // 2, ghost[0] * self.cell_size + self.cell_size // 2
            color = 'blue' if state['ghost_vulnerable'][i] else colors[i]
            self.canvas.create_oval(gx - 8, gy - 8, gx + 8, gy + 8, fill=color, outline='darkblue' if state['ghost_vulnerable'][i] else 'black')
        
        # Update stats
        stats = self.trainer.get_stats()
        self.episode_label.config(text=f"Episode: {stats['episode']}")
        self.envs_label.config(text=f"Envs: {stats['n_envs']}")
        self.reward_label.config(text=f"Avg Reward: {stats['avg_reward']:.1f}")
        self.last_reward_label.config(text=f"Last: {stats['last_reward']:.0f}")
        self.pacman_win_label.config(text=f"Pacman Wins: {stats['pacman_win_rate']:.1f}%")
        self.ghost_win_label.config(text=f"Ghost Wins: {stats['ghost_win_rate']:.1f}%")
        self.length_label.config(text=f"Avg Length: {stats['avg_length']:.0f}")
        self.pellets_label.config(text=f"Avg Pellets: {stats['avg_pellets']:.0f}")
        self.pacman_lr_label.config(text=f"Pacman LR: {stats['pacman_lr']:.2e}")
        self.ghost_lr_label.config(text=f"Ghost LR: {stats['ghost_lr']:.2e}")
        self.entropy_label.config(text=f"Entropy: {stats['entropy_coef']:.4f}")
        self.epsilon_label.config(text=f"Epsilon: {stats['ghost_epsilon']:.3f}")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    gui = PacmanGUI()
    gui.run()
