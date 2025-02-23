from flask import Flask, jsonify
from torch import nn
import torch
from ParseJson import ParseJson


app = Flask(__name__)

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

model = torch.load('backend/model.pth', weights_only=False)
model.eval()

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/getdata/<int:race_id>/<int:timestamp>")
def get_data(race_id, timestamp):
    return "<p>" + str(race_id) + str(timestamp) + "<p>"

if __name__ == "__main__":
    print(model(ParseJson.get_lap_history("backend/JsonData/2023_Fall.json", 185)))
    app.run()