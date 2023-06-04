# Import statements
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import time

# Set the backend to Agg
import matplotlib
matplotlib.use('Agg')

# Define the neural network model
class Participant(nn.Module):
    def __init__(self):
        super(Participant, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 3)
        self.strength = random.uniform(0.5, 1.5)
        self.agility = random.uniform(0.5, 1.5)
        self.defeated = False
        self.supplies = 0
        self.location = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set the random seed for reproducibility
torch.manual_seed(42)
random.seed(42)

# Define the simulation parameters
num_participants = st.slider("Number of Participants", 5, 50, 10)
training_epochs = st.slider("Training Epochs", 5, 20, 10)
input_size = 4
output_size = 3
grid_size = st.slider("Grid Size", 5, 20, 10)

# Load saved model weights if available
load_model = st.checkbox("Load saved model")
participants = [Participant() for _ in range(num_participants)]
if load_model:
    model_path = st.file_uploader("Upload model weights (file format: .pt)", type=["pt"])
    if model_path is not None:
        model_dict = torch.load(model_path)
        for participant, state_dict in zip(participants, model_dict):
            participant.load_state_dict(state_dict)

# Create a grid for visualization
grid = [["" for _ in range(grid_size)] for _ in range(grid_size)]

# Define the loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.SGD(params=[param for participant in participants for param in participant.parameters()], lr=0.01)

# Streamlit visualization
st.title("Hunger Games Simulation")

# Create an empty placeholder for the animation
animation_placeholder = st.empty()

# Create an empty placeholder for the log
log_placeholder = st.empty()

# Create buttons for control
start_button = st.button("Start")
pause_button = st.button("Pause")
stop_button = st.button("Stop")

# Best model variables
best_model_loss = float("inf")
best_model_weights = None

# Game loop
running = False
while True:
    if start_button:
        running = True
        start_button = False  # Reset button state
    elif stop_button:
        running = False
        stop_button = False  # Reset button state

    if running:
        log = ""

        # Perform actions for each participant
        for i, participant in enumerate(participants):
            if participant.defeated:
                continue

            action = random.choice(["move", "search", "attack"])
            log += f"Participant {i+1} performs action: {action}\n"

            if action == "move":
                dx = random.randint(-1, 1)
                dy = random.randint(-1, 1)
                new_x = participant.location[0] + dx
                new_y = participant.location[1] + dy

                if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
                    participant.location = (new_x, new_y)

            elif action == "search":
                found_supplies = random.randint(1, 10)
                participant.supplies += found_supplies
                log += f"Participant {i+1} found {found_supplies} supplies.\n"

            elif action == "attack":
                target = random.choice(participants)
                if target != participant and not target.defeated:
                    attack_prob = participant.strength - target.agility
                    if attack_prob > 0:
                        target.defeated = True
                        log += f"Participant {i+1} attacked and defeated Participant {participants.index(target) + 1}.\n"
                    else:
                        if random.random() < torch.sigmoid(torch.tensor(attack_prob)).item():
                            target.defeated = True
                            log += f"Participant {i+1} attacked and defeated Participant {participants.index(target) + 1}.\n"
                        else:
                            log += f"Participant {i+1} attempted to attack Participant {participants.index(target) + 1} but failed.\n"

            grid[participant.location[0]][participant.location[1]] = f"P{i+1}"

        # Train the participants (neural networks) after each round
        for _ in range(training_epochs):
            inputs = torch.randn(num_participants, input_size)
            targets = torch.randn(num_participants, output_size)

            optimizer.zero_grad()

            outputs = torch.stack([participant(input_data) for participant, input_data in zip(participants, inputs)])

            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

        fig, ax = plt.subplots()
        ax.imshow([[1 if cell == "" else 0 for cell in row] for row in grid], cmap='binary')
        for i in range(grid_size):
            for j in range(grid_size):
                if grid[i][j] != "":
                    ax.text(j, i, grid[i][j], ha='center', va='center')
        ax.axis('off')

        # Display the figure and close it
        animation_placeholder.pyplot(fig)
        plt.close(fig)

        log_placeholder.text(log)

        # Check if there is only one survivor left
        if sum(1 for participant in participants if not participant.defeated) == 1:
            # Display the final results
            surviving_participant = next(participant for participant in participants if not participant.defeated)
            st.write(f"Simulation completed! The winner is Participant {participants.index(surviving_participant) + 1}.")

            # Save the best model weights
            if loss.item() < best_model_loss:
                best_model_loss = loss.item()
                best_model_weights = surviving_participant.state_dict()

            # Train the next generation using the best model
            participants.clear()

            for _ in range(num_participants):
                participant = Participant()
                participant.load_state_dict(best_model_weights)  # Initialize participant with the best model weights
                participants.append(participant)

            # Reset defeated status and supplies
            for participant in participants:
                participant.defeated = False
                participant.supplies = 0

            # Reset the grid for the next simulation
            grid = [["" for _ in range(grid_size)] for _ in range(grid_size)]

    # Clear the grid for the next round
    grid = [["" for _ in range(grid_size)] for _ in range(grid_size)]
    time.sleep(0.1)
