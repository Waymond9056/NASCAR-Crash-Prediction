import torch
import numpy as np
from torch.utils.data import Dataset
from ParseJson import ParseJson
from torch.utils.data import DataLoader
from torch import nn
import os

class NascarDataSetTrain(Dataset):
    def __init__(self):
        races = ["backend/JsonData/2022_Fall.json", "backend/JsonData/2024_Fall.json", "backend/JsonData/2023_Spring.json", "backend/JsonData/2024_Spring.json"]
        self.inputs = []
        self.outputs = []
        for race in races:
            crash_laps, green_laps, _ = ParseJson.get_crash_laps(race)
            green_laps = green_laps[0:len(crash_laps)]
            for lap in crash_laps:
                lap_tensor = ParseJson.get_lap_info(race, lap)
                self.inputs.append(lap_tensor)
                self.outputs.append(1)
            for lap in green_laps:
                lap_tensor = ParseJson.get_lap_info(race, lap)
                self.inputs.append(lap_tensor)
                self.outputs.append(0)
        self.inputs = torch.Tensor(self.inputs)
        self.outputs = torch.Tensor(self.outputs)

    def __len__(self):
        return self.outputs.size(0)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]
    
class NascarDataSetTest(Dataset):
    def __init__(self):
        races = ["backend/JsonData/2023_Fall.json"]
        self.inputs = []
        self.outputs = []
        for race in races:
            crash_laps, green_laps, caution_laps = ParseJson.get_crash_laps(race)
            for lap in crash_laps:
                lap_tensor = ParseJson.get_lap_info(race, lap)
                self.inputs.append(lap_tensor)
                self.outputs.append(1)
            for lap in green_laps:
                lap_tensor = ParseJson.get_lap_info(race, lap)
                self.inputs.append(lap_tensor)
                self.outputs.append(0)
            for lap in caution_laps:
                lap_tensor = ParseJson.get_lap_info(race, lap)
                self.inputs.append(lap_tensor)
                self.outputs.append(1)
        self.inputs = torch.Tensor(self.inputs)
        self.outputs = torch.Tensor(self.outputs)

    def __len__(self):
        return 185
        #return self.outputs.size(0)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

train_dataloader = DataLoader(NascarDataSetTrain(), batch_size=200, shuffle=True)
test_dataloader = DataLoader(NascarDataSetTest(), batch_size=260, shuffle=False)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(40*2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout layer
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return torch.sigmoid(logits)

    
model = NeuralNetwork()
# Initialize the loss function
loss_fn = nn.BCELoss()

# Intializae an optimzer
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred.squeeze(), y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * 64 + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            print(pred)
            test_loss += loss_fn(pred.squeeze(), y).item()
            pred_labels = (pred.squeeze() >= 0.5).float()  # Convert probs to 0 or 1
            correct += (pred_labels == y).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 4
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "backend/model_weights.pth")
    