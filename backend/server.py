from flask import Flask, jsonify
from flask_cors import CORS
from ParseJson import ParseJson
from torch import nn
import torch
from ParseJson import ParseJson


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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
    return {"message" : "Hello World"}

@app.route('/<username>', methods=['GET'])
def get_user(username):
    return jsonify({"message": f"Hello, {username}!"})

# @app.route("/getdata/<int:race_id>/<int:timestamp>")
# def get_data(race_id, timestamp):
#     return "<p>" + str(race_id) + str(timestamp) + "<p>"

@app.route("/getlap/<int:timestamp>", methods=['GET'])
def get_lap(timestamp):
    return jsonify(ParseJson.get_lap_based_on_time("JsonData/2023_Fall.json", timestamp))

if __name__ == "__main__":
    print(model(ParseJson.get_lap_history("backend/JsonData/2023_Fall.json", 185)))
    app.run(debug=True)
