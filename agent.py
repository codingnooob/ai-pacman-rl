import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque, namedtuple
import random
import math

# Automatic device detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Using GPU: {gpu_name}")
else:
    print("Using device: CPU")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.hidden_cell = None
    
    def forward(self, state, hidden=None):
        # Handle batch dimension
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(state.shape) == 2:
            state = state.unsqueeze(1)  # Add sequence dimension
        
        lstm_out, self.hidden_cell = self.lstm(state, hidden)
        lstm_out = lstm_out[:, -1, :]  # Take last output
        
        action_probs = self.actor(lstm_out)
        state_value = self.critic(lstm_out)
        return action_probs, state_value
    
    def reset_hidden(self):
        self.hidden_cell = None

class PPOAgent:
    def __init__(self, state_dim, action_dim=4, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4, gae_lambda=0.95):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.gae_lambda = gae_lambda
        self.initial_lr = lr
        
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Learning rate scheduler (cosine annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=1e-5)
        
        self.mse_loss = nn.MSELoss()
        
        # Dynamic entropy coefficient
        self.entropy_coef = 0.01
        self.min_entropy_coef = 0.001
        self.entropy_decay = 0.9995
        
        # Memory
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []
        
        # Gradient tracking
        self.grad_norms = []
        
        # Episode counter for scheduling
        self.episode_count = 0
    
    def get_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            # Maintain hidden state across timesteps within episode
            action_probs, state_value = self.policy_old(state, self.policy_old.hidden_cell)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        
        return action.item(), action_logprob.item(), state_value.item()
    
    def store_transition(self, state, action, logprob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(done)
        self.values.append(value)
    
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def update(self):
        if len(self.states) < 2:
            self.states.clear()
            self.actions.clear()
            self.logprobs.clear()
            self.rewards.clear()
            self.is_terminals.clear()
            self.values.clear()
            return
        
        # Compute GAE advantages
        advantages = self.compute_gae(self.rewards, self.values, self.is_terminals)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute returns
        returns = advantages + torch.tensor(self.values, dtype=torch.float32).to(device)
        
        # Convert to tensors
        old_states = torch.FloatTensor(np.array(self.states)).to(device)
        old_actions = torch.LongTensor(self.actions).to(device)
        old_logprobs = torch.FloatTensor(self.logprobs).to(device)
        
        # Reset hidden state for training
        self.policy.reset_hidden()
        
        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluate old actions and values
            action_probs, state_values = self.policy(old_states)
            
            if torch.isnan(action_probs).any():
                print("Warning: NaN in action probs, skipping update")
                break
            
            dist = Categorical(action_probs)
            action_logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            
            state_values = state_values.squeeze(-1)
            
            # PPO loss
            ratios = torch.exp(action_logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, returns) - self.entropy_coef * dist_entropy
            
            if torch.isnan(loss).any():
                print("Warning: NaN in loss, skipping update")
                break
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            
            # Track gradient norms
            total_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.grad_norms.append(total_norm.item())
            
            self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step()
        
        # Decay entropy coefficient
        self.entropy_coef = max(self.min_entropy_coef, self.entropy_coef * self.entropy_decay)
        
        # Copy new weights
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.reset_hidden()
        
        # Clear memory
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        self.values.clear()
        
        self.episode_count += 1
    
    def get_grad_stats(self):
        if not self.grad_norms:
            return {'mean': 0, 'max': 0, 'min': 0}
        recent = self.grad_norms[-100:]
        return {
            'mean': np.mean(recent),
            'max': np.max(recent),
            'min': np.min(recent)
        }

# QMIX Components for Ghost Coordination
class DuelingQNetwork(nn.Module):
    """Dueling Q-network with LSTM for each ghost"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DuelingQNetwork, self).__init__()
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.hidden_cell = None
    
    def forward(self, state, hidden=None):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(state.shape) == 2:
            state = state.unsqueeze(1)
        
        lstm_out, self.hidden_cell = self.lstm(state, hidden)
        lstm_out = lstm_out[:, -1, :]
        
        value = self.value_stream(lstm_out)
        advantage = self.advantage_stream(lstm_out)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values.squeeze(0) if q_values.shape[0] == 1 else q_values
    
    def reset_hidden(self):
        self.hidden_cell = None

class QMixingNetwork(nn.Module):
    """Mixing network with attention mechanism"""
    def __init__(self, n_agents, state_dim, hidden_dim=32):
        super(QMixingNetwork, self).__init__()
        self.n_agents = n_agents
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=n_agents, num_heads=1, batch_first=True)
        
        # Hypernetworks for weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents * hidden_dim)
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Hypernetworks for biases
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, agent_qs, global_state):
        batch_size = agent_qs.size(0)
        
        # Apply attention to agent Q-values
        agent_qs_attended, _ = self.attention(
            agent_qs.unsqueeze(1), 
            agent_qs.unsqueeze(1), 
            agent_qs.unsqueeze(1)
        )
        agent_qs_attended = agent_qs_attended.squeeze(1)
        
        # Generate weights and biases
        w1 = torch.abs(self.hyper_w1(global_state))
        w1 = w1.view(batch_size, self.n_agents, -1)
        
        b1 = self.hyper_b1(global_state)
        b1 = b1.view(batch_size, 1, -1)
        
        # First layer
        hidden = torch.bmm(agent_qs_attended.unsqueeze(1), w1) + b1
        hidden = torch.relu(hidden)
        
        # Second layer
        w2 = torch.abs(self.hyper_w2(global_state))
        w2 = w2.view(batch_size, -1, 1)
        
        b2 = self.hyper_b2(global_state)
        
        # Output
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.squeeze()

# Prioritized Experience Replay
Transition = namedtuple('Transition', ['states', 'actions', 'reward', 'next_states', 'done', 'global_state', 'next_global_state'])

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling weight
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def push(self, *args):
        transition = Transition(*args)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None, None, None
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        samples = [self.buffer[idx] for idx in indices]
        return samples, indices, torch.FloatTensor(weights)
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            priority_val = float(priority)
            self.priorities[idx] = priority_val
            self.max_priority = max(self.max_priority, priority_val)
    
    def __len__(self):
        return len(self.buffer)

# Intrinsic Curiosity Module
class ICM(nn.Module):
    """Intrinsic Curiosity Module for exploration bonus"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ICM, self).__init__()
        
        # Forward model: predicts next state from current state and action
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Inverse model: predicts action from current and next state
        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state, action, next_state):
        # One-hot encode action
        action_onehot = torch.zeros(action.size(0), 4, device=device)
        action_onehot.scatter_(1, action.unsqueeze(1), 1)
        
        # Forward model prediction
        state_action = torch.cat([state, action_onehot], dim=1)
        pred_next_state = self.forward_model(state_action)
        
        # Inverse model prediction
        state_pair = torch.cat([state, next_state], dim=1)
        pred_action = self.inverse_model(state_pair)
        
        # Intrinsic reward is prediction error
        intrinsic_reward = 0.5 * ((pred_next_state - next_state) ** 2).mean(dim=1)
        
        return intrinsic_reward, pred_action

class QMIXAgent:
    """QMIX with prioritized replay, ICM, and learning rate scheduling"""
    def __init__(self, n_agents=4, state_dim=4, global_state_dim=10, action_dim=4, 
                 lr=5e-4, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Individual Q-networks (now Dueling)
        self.q_networks = [DuelingQNetwork(state_dim, action_dim).to(device) for _ in range(n_agents)]
        self.target_q_networks = [DuelingQNetwork(state_dim, action_dim).to(device) for _ in range(n_agents)]
        
        # Mixing network (now with attention)
        self.mixer = QMixingNetwork(n_agents, global_state_dim).to(device)
        self.target_mixer = QMixingNetwork(n_agents, global_state_dim).to(device)
        
        # Intrinsic Curiosity Module
        self.icm = ICM(state_dim, action_dim).to(device)
        
        # Copy weights to target networks
        for i in range(n_agents):
            self.target_q_networks[i].load_state_dict(self.q_networks[i].state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        
        # Optimizer for all networks
        params = []
        for net in self.q_networks:
            params += list(net.parameters())
        params += list(self.mixer.parameters())
        params += list(self.icm.parameters())
        self.optimizer = optim.Adam(params, lr=lr)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)
        
        # Prioritized replay buffer
        self.memory = PrioritizedReplayBuffer(capacity=10000)
        self.batch_size = 32
        
        # Gradient tracking
        self.grad_norms = []
    
    def get_actions(self, states):
        """Get actions for all ghosts"""
        actions = []
        for i, state in enumerate(states):
            if random.random() < self.epsilon:
                actions.append(random.randint(0, self.action_dim - 1))
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    # Check if hidden state exists and has correct batch size
                    hidden = self.q_networks[i].hidden_cell
                    if hidden is not None and hidden[0].size(1) != 1:
                        hidden = None  # Reset if batch size mismatch
                    q_values = self.q_networks[i](state_tensor, hidden)
                    actions.append(q_values.argmax().item())
        return actions
    
    def store_transition(self, states, actions, reward, next_states, done, global_state, next_global_state):
        self.memory.push(states, actions, reward, next_states, done, global_state, next_global_state)
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch with priorities
        batch, indices, weights = self.memory.sample(self.batch_size)
        if batch is None:
            return
        
        states_batch = [[] for _ in range(self.n_agents)]
        actions_batch = [[] for _ in range(self.n_agents)]
        rewards_batch = []
        next_states_batch = [[] for _ in range(self.n_agents)]
        dones_batch = []
        global_states_batch = []
        next_global_states_batch = []
        
        for transition in batch:
            for i in range(self.n_agents):
                states_batch[i].append(transition.states[i])
                actions_batch[i].append(transition.actions[i])
                next_states_batch[i].append(transition.next_states[i])
            rewards_batch.append(transition.reward)
            dones_batch.append(transition.done)
            global_states_batch.append(transition.global_state)
            next_global_states_batch.append(transition.next_global_state)
        
        # Convert to tensors
        rewards = torch.FloatTensor(rewards_batch).to(device)
        dones = torch.FloatTensor(dones_batch).to(device)
        global_states = torch.FloatTensor(np.array(global_states_batch)).to(device)
        next_global_states = torch.FloatTensor(np.array(next_global_states_batch)).to(device)
        weights = weights.to(device)
        
        # Compute intrinsic rewards
        intrinsic_rewards = torch.zeros(self.batch_size).to(device)
        for i in range(self.n_agents):
            states = torch.FloatTensor(np.array(states_batch[i])).to(device)
            actions = torch.LongTensor(actions_batch[i]).to(device)
            next_states = torch.FloatTensor(np.array(next_states_batch[i])).to(device)
            
            with torch.no_grad():
                int_reward, _ = self.icm(states, actions, next_states)
                intrinsic_rewards += int_reward
        
        intrinsic_rewards /= self.n_agents
        rewards = rewards + 0.1 * intrinsic_rewards  # Scale intrinsic reward
        
        # Get current Q-values
        agent_qs = []
        for i in range(self.n_agents):
            self.q_networks[i].reset_hidden()
            states = torch.FloatTensor(np.array(states_batch[i])).to(device)
            actions = torch.LongTensor(actions_batch[i]).to(device)
            q_values = self.q_networks[i](states)
            agent_q = q_values.gather(1, actions.unsqueeze(1)).squeeze()
            agent_qs.append(agent_q)
        
        agent_qs = torch.stack(agent_qs, dim=1)
        q_tot = self.mixer(agent_qs, global_states)
        
        # Get target Q-values
        with torch.no_grad():
            next_agent_qs = []
            for i in range(self.n_agents):
                self.target_q_networks[i].reset_hidden()
                next_states = torch.FloatTensor(np.array(next_states_batch[i])).to(device)
                next_q_values = self.target_q_networks[i](next_states)
                next_agent_q = next_q_values.max(1)[0]
                next_agent_qs.append(next_agent_q)
            
            next_agent_qs = torch.stack(next_agent_qs, dim=1)
            target_q_tot = self.target_mixer(next_agent_qs, next_global_states)
            target_q_tot = rewards + self.gamma * target_q_tot * (1 - dones)
        
        # Compute TD errors for priority update
        td_errors = torch.abs(q_tot - target_q_tot).detach().cpu().numpy()
        # Ensure it's a 1D array
        td_errors = np.atleast_1d(td_errors).flatten()
        self.memory.update_priorities(indices, td_errors + 1e-6)
        
        # Weighted loss
        loss = (weights * (q_tot - target_q_tot) ** 2).mean()
        
        if torch.isnan(loss):
            print("Warning: NaN in QMIX loss, skipping update")
            return
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Track and clip gradients
        total_norm = 0
        for net in self.q_networks:
            total_norm += torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        total_norm += torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), 1.0)
        self.grad_norms.append(total_norm.item())
        
        self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_networks(self):
        for i in range(self.n_agents):
            self.target_q_networks[i].load_state_dict(self.q_networks[i].state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
    
    def get_grad_stats(self):
        if not self.grad_norms:
            return {'mean': 0, 'max': 0, 'min': 0}
        recent = self.grad_norms[-100:]
        return {
            'mean': np.mean(recent),
            'max': np.max(recent),
            'min': np.min(recent)
        }

class PacmanAgent:
    def __init__(self):
        # State: [ghost_dx, ghost_dy, food_dx, food_dy, ghost_dist, vulnerable]
        self.agent = PPOAgent(state_dim=6, action_dim=4)
        self.last_state = None
        self.last_action = None
        self.last_logprob = None
        self.last_value = None
    
    def get_state_repr(self, game_state):
        pacman = game_state['pacman']
        ghosts = game_state['ghosts']
        vulnerable = game_state['ghost_vulnerable']
        
        # Find nearest ghost
        nearest_ghost_dist = float('inf')
        nearest_ghost_dir = [0, 0]
        nearest_vulnerable = 0
        
        for i, ghost in enumerate(ghosts):
            dist = abs(ghost[0] - pacman[0]) + abs(ghost[1] - pacman[1])
            if dist < nearest_ghost_dist:
                nearest_ghost_dist = dist
                nearest_ghost_dir = [ghost[0] - pacman[0], ghost[1] - pacman[1]]
                nearest_vulnerable = 1 if vulnerable[i] else 0
        
        # Normalize ghost direction
        if nearest_ghost_dist > 0:
            nearest_ghost_dir = [nearest_ghost_dir[0] / 10.0, nearest_ghost_dir[1] / 10.0]
        
        # Find nearest food
        all_food = list(game_state['pellets']) + list(game_state['power_pellets'])
        if all_food:
            nearest_food = min(all_food, key=lambda p: abs(p[0] - pacman[0]) + abs(p[1] - pacman[1]))
            food_dir = [nearest_food[0] - pacman[0], nearest_food[1] - pacman[1]]
            food_dist = abs(food_dir[0]) + abs(food_dir[1])
            if food_dist > 0:
                food_dir = [food_dir[0] / 10.0, food_dir[1] / 10.0]
        else:
            food_dir = [0, 0]
        
        # Normalize distance
        dist_normalized = min(nearest_ghost_dist / 20.0, 1.0)
        
        return np.array([
            nearest_ghost_dir[0], nearest_ghost_dir[1],
            food_dir[0], food_dir[1],
            dist_normalized, nearest_vulnerable
        ], dtype=np.float32)
    
    def get_action(self, state):
        action, logprob, value = self.agent.get_action(state)
        self.last_state = state
        self.last_action = action
        self.last_logprob = logprob
        self.last_value = value
        return action
    
    def update(self, state, action, reward, next_state, done):
        if self.last_state is not None:
            self.agent.store_transition(self.last_state, self.last_action, self.last_logprob, reward, done, self.last_value)
        
        if done:
            self.agent.update()
            # Reset LSTM hidden state on episode end
            self.agent.policy.reset_hidden()
            self.agent.policy_old.reset_hidden()

class GhostTeam:
    """Coordinated ghost team using QMIX"""
    def __init__(self):
        self.qmix = QMIXAgent(n_agents=4, state_dim=4, global_state_dim=10)
        self.update_counter = 0
        self.target_update_freq = 100
    
    def get_ghost_state(self, game_state, ghost_id):
        """Get individual ghost state"""
        ghost_pos = game_state['ghosts'][ghost_id]
        pacman = game_state['pacman']
        vulnerable = game_state['ghost_vulnerable'][ghost_id]
        
        # Relative position to Pacman
        rel_pos = [pacman[0] - ghost_pos[0], pacman[1] - ghost_pos[1]]
        distance = abs(rel_pos[0]) + abs(rel_pos[1])
        
        # Normalize
        if distance > 0:
            rel_pos = [rel_pos[0] / 10.0, rel_pos[1] / 10.0]
        dist_normalized = min(distance / 20.0, 1.0)
        
        return np.array([
            rel_pos[0], rel_pos[1],
            dist_normalized, 1 if vulnerable else 0
        ], dtype=np.float32)
    
    def get_global_state(self, game_state):
        """Get global state for mixing network"""
        pacman = game_state['pacman']
        ghosts = game_state['ghosts']
        
        # Average ghost position
        avg_ghost_pos = np.mean(ghosts, axis=0)
        
        # Pacman position normalized
        pacman_norm = [pacman[0] / 31.0, pacman[1] / 28.0]
        
        # Average distance to pacman
        avg_dist = np.mean([abs(g[0] - pacman[0]) + abs(g[1] - pacman[1]) for g in ghosts]) / 20.0
        
        # Ghost spread (how dispersed they are)
        ghost_spread = np.std([g[0] for g in ghosts] + [g[1] for g in ghosts]) / 10.0
        
        # Number of vulnerable ghosts
        num_vulnerable = sum(game_state['ghost_vulnerable']) / 4.0
        
        # Pellets remaining
        pellets_remaining = (len(game_state['pellets']) + len(game_state['power_pellets'])) / 200.0
        
        return np.array([
            pacman_norm[0], pacman_norm[1],
            avg_ghost_pos[0] / 31.0, avg_ghost_pos[1] / 28.0,
            avg_dist, ghost_spread, num_vulnerable, pellets_remaining,
            0, 0  # Padding to make it 10-dim
        ], dtype=np.float32)
    
    def get_actions(self, game_state):
        """Get coordinated actions for all ghosts"""
        states = [self.get_ghost_state(game_state, i) for i in range(4)]
        return self.qmix.get_actions(states)
    
    def update(self, game_state, actions, reward, next_game_state, done):
        """Update QMIX with team experience"""
        states = [self.get_ghost_state(game_state, i) for i in range(4)]
        next_states = [self.get_ghost_state(next_game_state, i) for i in range(4)]
        global_state = self.get_global_state(game_state)
        next_global_state = self.get_global_state(next_game_state)
        
        self.qmix.store_transition(states, actions, reward, next_states, done, global_state, next_global_state)
        self.qmix.update()
        
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.qmix.update_target_networks()
