# ============================================================
# AI Reinforcement Learning Agent for CartPole
# Using PyTorch and Gymnasium
# ============================================================

import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CHECK PYTORCH
# ============================================================

print("=" * 60)
print("AI REINFORCEMENT LEARNING - CARTPOLE PROJECT")
print("=" * 60)

print("\nPyTorch version:", torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# 2. CREATE CARTPOLE ENVIRONMENT
# ============================================================

env = gym.make("CartPole-v1")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

print("\nEnvironment: CartPole-v1")
print("State size:", state_size)
print("Number of actions:", action_size)

env.close()


# ============================================================
# 3. DEFINE THE NEURAL NETWORK
# ============================================================

class DQN(nn.Module):

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, action_size)
        )

    def forward(self, state):
        return self.network(state)


# Create the model
model = DQN(state_size, action_size).to(device)

# Target network
target_model = DQN(state_size, action_size).to(device)

target_model.load_state_dict(model.state_dict())
target_model.eval()

print("\nNeural network created successfully!")
print(model)


# ============================================================
# 4. REPLAY MEMORY
# ============================================================

memory = deque(maxlen=50000)


# ============================================================
# 5. HYPERPARAMETERS
# ============================================================

EPISODES = 300
MAX_STEPS = 500

BATCH_SIZE = 64

GAMMA = 0.99

LEARNING_RATE = 0.001

EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

TARGET_UPDATE = 10


# ============================================================
# 6. OPTIMIZER AND LOSS FUNCTION
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

loss_function = nn.SmoothL1Loss()


# ============================================================
# 7. SELECT ACTION
# ============================================================

def select_action(state, epsilon):

    # Exploration
    if random.random() < epsilon:
        return random.randrange(action_size)

    # Exploitation
    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = model(state_tensor)

    return q_values.argmax(dim=1).item()


# ============================================================
# 8. TRAIN THE AI
# ============================================================

def train_model():

    if len(memory) < BATCH_SIZE:
        return

    # Select random experiences
    batch = random.sample(memory, BATCH_SIZE)

    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    for experience in batch:

        state, action, reward, next_state, done = experience

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)

    states = torch.tensor(
        np.array(states),
        dtype=torch.float32,
        device=device
    )

    actions = torch.tensor(
        actions,
        dtype=torch.long,
        device=device
    ).unsqueeze(1)

    rewards = torch.tensor(
        rewards,
        dtype=torch.float32,
        device=device
    )

    next_states = torch.tensor(
        np.array(next_states),
        dtype=torch.float32,
        device=device
    )

    dones = torch.tensor(
        dones,
        dtype=torch.float32,
        device=device
    )

    # Current Q values
    current_q_values = model(states).gather(
        1,
        actions
    ).squeeze(1)

    # Next Q values
    with torch.no_grad():

        next_q_values = target_model(
            next_states
        ).max(1)[0]

        target_q_values = rewards + (
            GAMMA * next_q_values * (1 - dones)
        )

    # Calculate loss
    loss = loss_function(
        current_q_values,
        target_q_values
    )

    # Backpropagation
    optimizer.zero_grad()

    loss.backward()

    # Prevent very large gradients
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    optimizer.step()


# ============================================================
# 9. TRAINING LOOP
# ============================================================

print("\n" + "=" * 60)
print("STARTING AI TRAINING")
print("=" * 60)

rewards_list = []

epsilon = EPSILON

for episode in range(1, EPISODES + 1):

    env = gym.make("CartPole-v1")

    state, info = env.reset()

    total_reward = 0

    for step in range(MAX_STEPS):

        # Select action
        action = select_action(
            state,
            epsilon
        )

        # Perform action
        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        # Store experience
        memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )

        # Train the neural network
        train_model()

        # Move to next state
        state = next_state

        total_reward += reward

        if done:
            break

    env.close()

    # Store reward
    rewards_list.append(total_reward)

    # Reduce exploration
    epsilon = max(
        EPSILON_MIN,
        epsilon * EPSILON_DECAY
    )

    # Update target network
    if episode % TARGET_UPDATE == 0:

        target_model.load_state_dict(
            model.state_dict()
        )

    # Calculate average reward
    if len(rewards_list) >= 10:

        average_reward = np.mean(
            rewards_list[-10:]
        )

    else:

        average_reward = np.mean(
            rewards_list
        )

    # Display progress
    if episode % 10 == 0:

        print(
            f"Episode: {episode:3d} | "
            f"Reward: {total_reward:6.1f} | "
            f"Average Reward: {average_reward:6.1f} | "
            f"Epsilon: {epsilon:.3f}"
        )


# ============================================================
# 10. SAVE TRAINED MODEL
# ============================================================

torch.save(
    model.state_dict(),
    "cartpole_dqn_model.pth"
)

print("\nTraining completed!")

print(
    "Trained model saved as: "
    "cartpole_dqn_model.pth"
)


# ============================================================
# 11. SAVE TRAINING RESULTS
# ============================================================

results = pd.DataFrame({

    "Episode": range(
        1,
        EPISODES + 1
    ),

    "Reward": rewards_list

})

results.to_csv(
    "training_results.csv",
    index=False
)

print(
    "Training results saved as: "
    "training_results.csv"
)


# ============================================================
# 12. CREATE REWARD GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    results["Episode"],
    results["Reward"],
    alpha=0.4,
    label="Episode Reward"
)

# Moving average
moving_average = (
    results["Reward"]
    .rolling(window=20)
    .mean()
)

plt.plot(
    results["Episode"],
    moving_average,
    linewidth=2,
    label="20-Episode Moving Average"
)

plt.title(
    "CartPole AI Training Performance"
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Reward"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "training_graph.png",
    dpi=300
)

plt.show()


# ============================================================
# 13. TEST THE TRAINED AI
# ============================================================

print("\n" + "=" * 60)
print("TESTING TRAINED AI")
print("=" * 60)

test_env = gym.make(
    "CartPole-v1"
)

test_rewards = []

for test_episode in range(1, 6):

    state, info = test_env.reset()

    total_reward = 0

    for step in range(MAX_STEPS):

        # AI chooses the best action
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = model(
                state_tensor
            )

        action = q_values.argmax(
            dim=1
        ).item()

        # Perform action
        next_state, reward, terminated, truncated, info = test_env.step(action)

        total_reward += reward

        state = next_state

        if terminated or truncated:
            break

    test_rewards.append(total_reward)

    print(
        f"Test Episode {test_episode}: "
        f"Reward = {total_reward:.1f}"
    )

test_env.close()


# ============================================================
# 14. FINAL RESULTS
# ============================================================

average_test_reward = np.mean(
    test_rewards
)

best_test_reward = np.max(
    test_rewards
)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(
    f"Average Test Reward: "
    f"{average_test_reward:.2f}"
)

print(
    f"Best Test Reward: "
    f"{best_test_reward:.2f}"
)

print(
    "\nProject completed successfully!"
)

print("\nGenerated files:")

print("1. cartpole_dqn_model.pth")
print("2. training_results.csv")
print("3. training_graph.png")

print("=" * 60)