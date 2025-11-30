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

PERCEPTION_RADIUS = 2
PERCEPTION_DIAMETER = PERCEPTION_RADIUS * 2 + 1
PERCEPTION_AREA = PERCEPTION_DIAMETER ** 2
PERCEPTION_CHANNELS = 3
PACMAN_GHOST_FEATURES = 5  # dx, dy, distance, vulnerability, visibility
PACMAN_HUNGER_PROGRESS_DIM = 6  # hunger meter, idle, score-freeze, pellet, power, unique ratios
PACMAN_STATE_DIM = PERCEPTION_AREA * PERCEPTION_CHANNELS + PACMAN_GHOST_FEATURES * 4 + PACMAN_HUNGER_PROGRESS_DIM
GHOST_ADDITIONAL_FEATURES = 15  # see GhostTeam.get_ghost_state docstring
GHOST_STATE_DIM = PERCEPTION_AREA * PERCEPTION_CHANNELS + GHOST_ADDITIONAL_FEATURES
GLOBAL_STATE_DIM = 18
MAX_VISIBILITY_DEPTH = 12


def _get_maze_dimensions(game_state):
    dims = game_state.get('dimensions', {}) or {}
    height = dims.get('height', 31)
    width = dims.get('width', 28)
    diag = math.sqrt(height ** 2 + width ** 2)
    return height, width, diag


def _safe_ratio(value, denom):
    if denom is None or denom <= 0:
        return 0.0
    return float(np.clip(value / denom, 0.0, 1.0))


def _hunger_meter_ratio(hunger_meter, hunger_config):
    limit = hunger_config.get('hunger_termination_limit', -150.0)
    if limit >= 0:
        return float(np.clip(hunger_meter / max(limit, 1.0), 0.0, 1.0))
    return float(np.clip(1.0 - (hunger_meter / limit), 0.0, 1.0))


def _extract_hunger_features(game_state):
    hunger_stats = game_state.get('hunger_stats', {}) or {}
    hunger_config = game_state.get('hunger_config', {}) or {}
    idle_threshold = max(hunger_config.get('hunger_idle_threshold', 1), 1)
    survival_grace = max(hunger_config.get('survival_grace_steps', 1), 1)
    stagnation_window = max(hunger_config.get('stagnation_tile_window', 1), 1)
    hunger_meter = hunger_stats.get('hunger_meter', 0.0)
    hunger_ratio = _hunger_meter_ratio(hunger_meter, hunger_config)
    steps_ratio = _safe_ratio(hunger_stats.get('steps_since_progress', 0), idle_threshold)
    score_freeze_ratio = _safe_ratio(hunger_stats.get('score_freeze_steps', 0), survival_grace)
    unique_ratio = _safe_ratio(hunger_stats.get('unique_tiles', 0), stagnation_window)
    return hunger_ratio, steps_ratio, score_freeze_ratio, unique_ratio


def _pellet_progress_features(game_state):
    pellets = game_state.get('pellets', set()) or set()
    power_pellets = game_state.get('power_pellets', set()) or set()
    initial_counts = game_state.get('initial_counts', {}) or {}
    pellet_total = initial_counts.get('pellets', len(pellets))
    power_total = initial_counts.get('power_pellets', len(power_pellets))
    pellet_ratio = _safe_ratio(len(pellets), pellet_total)
    power_ratio = _safe_ratio(len(power_pellets), power_total)
    return pellet_ratio, power_ratio


def _build_local_grid(game_state, center):
    walls = game_state.get('walls', set()) or set()
    pellets = game_state.get('pellets', set()) or set()
    power_pellets = game_state.get('power_pellets', set()) or set()
    height, width, _ = _get_maze_dimensions(game_state)
    features = []
    for dr in range(-PERCEPTION_RADIUS, PERCEPTION_RADIUS + 1):
        for dc in range(-PERCEPTION_RADIUS, PERCEPTION_RADIUS + 1):
            row = center[0] + dr
            col = center[1] + dc
            out_of_bounds = row < 0 or row >= height or col < 0 or col >= width
            if out_of_bounds:
                features.extend([1.0, 0.0, 0.0])
            else:
                cell = (row, col)
                features.extend([
                    1.0 if cell in walls else 0.0,
                    1.0 if cell in pellets else 0.0,
                    1.0 if cell in power_pellets else 0.0
                ])
    return features


def _bfs_distance(start, target, walls, width, height, max_depth=MAX_VISIBILITY_DEPTH):
    if start == target:
        return 0
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        (row, col), dist = queue.popleft()
        if dist >= max_depth:
            continue
        for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if (nr, nc) in walls or (nr, nc) in visited:
                continue
            next_dist = dist + 1
            if (nr, nc) == target:
                return next_dist
            visited.add((nr, nc))
            queue.append(((nr, nc), next_dist))
    return None


def _normalize_velocity(velocity):
    if not velocity:
        return 0.0, 0.0
    return (
        float(np.clip(velocity[0], -1, 1)),
        float(np.clip(velocity[1], -1, 1))
    )


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
        self.entropy_coef = 0.02
        self.min_entropy_coef = 0.005
        self.entropy_decay = 0.999
        self.dirichlet_alpha = 2.0
        self.dirichlet_mix = 0.12
        
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
            if self.dirichlet_mix > 0:
                concentration = torch.full((action_probs.shape[-1],), self.dirichlet_alpha, device=action_probs.device)
                noise = torch.distributions.Dirichlet(concentration).sample().unsqueeze(0)
                action_probs = (1 - self.dirichlet_mix) * action_probs + self.dirichlet_mix * noise
                action_probs = torch.clamp(action_probs, 1e-6, 1.0)
                action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
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
    def __init__(self, n_agents=4, state_dim=GHOST_STATE_DIM, global_state_dim=GLOBAL_STATE_DIM, action_dim=4, 
                 lr=5e-4, gamma=0.99, epsilon=1.0, epsilon_decay=0.997, epsilon_min=0.15):
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
    
    def update(self, done=False):
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
        
        # Decay epsilon ONLY at episode boundaries (when done=True)
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # ADDED: Exploration monitoring and periodic boosts
            self.episode_count = getattr(self, 'episode_count', 0) + 1
            
            # Periodic exploration boost every 1000 episodes
            if self.episode_count % 1000 == 0:
                self.epsilon = min(0.3, self.epsilon + 0.1)  # Boost exploration
                print(f"Exploration boost at episode {self.episode_count}: epsilon = {self.epsilon:.3f}")
            
            # Store epsilon history for analysis
            if not hasattr(self, 'epsilon_history'):
                self.epsilon_history = []
            self.epsilon_history.append(self.epsilon)
            
            # Alert if epsilon drops too low
            if self.epsilon < 0.1 and self.episode_count % 100 == 0:
                print(f"WARNING: Low exploration at episode {self.episode_count}: epsilon = {self.epsilon:.3f}")
    
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
        # State: local 5x5 perception (walls/pellets/power), per-ghost embeddings, hunger + progress cues
        self.agent = PPOAgent(state_dim=PACMAN_STATE_DIM, action_dim=4)
        self.last_state = None
        self.last_action = None
        self.last_logprob = None
        self.last_value = None
    
    def get_state_repr(self, game_state):
        pacman = game_state['pacman']
        ghosts = game_state['ghosts']
        vulnerable = game_state['ghost_vulnerable']
        hunger_ratio, steps_ratio, score_freeze_ratio, unique_ratio = _extract_hunger_features(game_state)
        pellet_ratio, power_ratio = _pellet_progress_features(game_state)
        height, width, diag = _get_maze_dimensions(game_state)
        walls = game_state.get('walls', set()) or set()
        perception = _build_local_grid(game_state, pacman)
        ghost_features = []
        for i, ghost in enumerate(ghosts):
            rel_row = (ghost[0] - pacman[0]) / max(height, 1)
            rel_col = (ghost[1] - pacman[1]) / max(width, 1)
            manhattan = abs(ghost[0] - pacman[0]) + abs(ghost[1] - pacman[1])
            manhattan_norm = min(manhattan / max(diag, 1.0), 1.0)
            visibility_dist = _bfs_distance(tuple(pacman), tuple(ghost), walls, width, height)
            visible_flag = 1.0 if visibility_dist is not None else 0.0
            ghost_features.extend([
                rel_row,
                rel_col,
                manhattan_norm,
                1.0 if vulnerable[i] else 0.0,
                visible_flag
            ])
        features = perception + ghost_features + [
            hunger_ratio,
            steps_ratio,
            score_freeze_ratio,
            pellet_ratio,
            power_ratio,
            unique_ratio
        ]
        return np.array(features, dtype=np.float32)
    
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
        self.qmix = QMIXAgent(n_agents=4, state_dim=GHOST_STATE_DIM, global_state_dim=GLOBAL_STATE_DIM)
        self.update_counter = 0
        self.target_update_freq = 100
    
    def get_ghost_state(self, game_state, ghost_id):
        """Get individual ghost observation with local context and team/power cues"""
        ghosts = game_state['ghosts']
        ghost_pos = ghosts[ghost_id]
        pacman = game_state['pacman']
        vulnerable = 1.0 if game_state['ghost_vulnerable'][ghost_id] else 0.0
        height, width, diag = _get_maze_dimensions(game_state)
        walls = game_state.get('walls', set()) or set()
        local_grid = _build_local_grid(game_state, ghost_pos)
        rel_row = (pacman[0] - ghost_pos[0]) / max(height, 1)
        rel_col = (pacman[1] - ghost_pos[1]) / max(width, 1)
        manhattan = abs(pacman[0] - ghost_pos[0]) + abs(pacman[1] - ghost_pos[1])
        manhattan_norm = min(manhattan / max(diag, 1.0), 1.0)
        visibility_dist = _bfs_distance(tuple(ghost_pos), tuple(pacman), walls, width, height)
        visible_flag = 1.0 if visibility_dist is not None else 0.0
        visible_distance_norm = min(visibility_dist / max(diag, 1.0), 1.0) if visibility_dist is not None else 0.0
        vel_row, vel_col = _normalize_velocity(game_state.get('pacman_velocity', (0, 0)))
        power_pellets = game_state.get('power_pellets', set()) or set()
        if power_pellets:
            nearest_power = min(
                power_pellets,
                key=lambda p: abs(p[0] - ghost_pos[0]) + abs(p[1] - ghost_pos[1])
            )
            power_dist = abs(nearest_power[0] - ghost_pos[0]) + abs(nearest_power[1] - ghost_pos[1])
            power_dist_norm = min(power_dist / max(diag, 1.0), 1.0)
        else:
            power_dist_norm = 0.0
        pacman_powered = 1.0 if any(game_state['ghost_vulnerable']) else 0.0
        ghost_one_hot = [1.0 if idx == ghost_id else 0.0 for idx in range(4)]
        teammate_distances = []
        for idx, other in enumerate(ghosts):
            if idx == ghost_id:
                continue
            teammate_distances.append(abs(other[0] - pacman[0]) + abs(other[1] - pacman[1]))
        avg_teammate_dist = np.mean(teammate_distances) if teammate_distances else manhattan
        avg_teammate_dist_norm = min(avg_teammate_dist / max(diag, 1.0), 1.0)
        features = local_grid + [
            rel_row,
            rel_col,
            manhattan_norm,
            visible_flag,
            visible_distance_norm,
            vel_row,
            vel_col,
            power_dist_norm,
            pacman_powered,
            vulnerable
        ] + ghost_one_hot + [avg_teammate_dist_norm]
        return np.array(features, dtype=np.float32)
    
    def get_global_state(self, game_state):
        """Global state for mixer with hunger, pellet density, and spread summaries"""
        pacman = game_state['pacman']
        ghosts = game_state['ghosts']
        height, width, diag = _get_maze_dimensions(game_state)
        pacman_row_norm = pacman[0] / max(height - 1, 1)
        pacman_col_norm = pacman[1] / max(width - 1, 1)
        avg_ghost_row = np.mean([g[0] for g in ghosts]) / max(height - 1, 1)
        avg_ghost_col = np.mean([g[1] for g in ghosts]) / max(width - 1, 1)
        distances = [abs(g[0] - pacman[0]) + abs(g[1] - pacman[1]) for g in ghosts]
        mean_dist_norm = min(np.mean(distances) / max(diag, 1.0), 1.0) if distances else 0.0
        std_dist_norm = min(np.std(distances) / max(diag, 1.0), 1.0) if len(distances) > 1 else 0.0
        vel_row, vel_col = _normalize_velocity(game_state.get('pacman_velocity', (0, 0)))
        hunger_ratio, steps_ratio, score_freeze_ratio, unique_ratio = _extract_hunger_features(game_state)
        pellet_ratio, power_ratio = _pellet_progress_features(game_state)
        vulnerability_fraction = float(np.mean(game_state['ghost_vulnerable'])) if game_state['ghost_vulnerable'] else 0.0
        pairwise = []
        for idx, g1 in enumerate(ghosts):
            for jdx, g2 in enumerate(ghosts):
                if jdx <= idx:
                    continue
                pairwise.append(abs(g1[0] - g2[0]) + abs(g1[1] - g2[1]))
        pairwise_mean = np.mean(pairwise) if pairwise else 0.0
        pairwise_norm = min(pairwise_mean / max(diag, 1.0), 1.0)
        pellets = list(game_state.get('pellets', set()) or [])
        power_pellets = list(game_state.get('power_pellets', set()) or [])
        all_food = pellets + power_pellets
        total_food = len(all_food)
        top_half = _safe_ratio(sum(1 for cell in all_food if cell[0] < height / 2), total_food)
        left_half = _safe_ratio(sum(1 for cell in all_food if cell[1] < width / 2), total_food)
        return np.array([
            pacman_row_norm,
            pacman_col_norm,
            avg_ghost_row,
            avg_ghost_col,
            mean_dist_norm,
            std_dist_norm,
            vel_row,
            vel_col,
            hunger_ratio,
            steps_ratio,
            score_freeze_ratio,
            pellet_ratio,
            power_ratio,
            unique_ratio,
            vulnerability_fraction,
            pairwise_norm,
            top_half,
            left_half
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
        self.qmix.update(done=done)
        
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.qmix.update_target_networks()
