from flask import Flask, jsonify
from flask_cors import CORS
from ParseJson import ParseJson
from torch import nn
import torch
from ParseJson import ParseJson
from captum.attr import IntegratedGradients
import numpy as np


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(40*2, 32),
            nn.ReLU(),
            nn.Dropout(0.95),  # Dropout layer
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return torch.sigmoid(logits)

model = NeuralNetwork()
model.load_state_dict(torch.load('backend/model_weights.pth', weights_only=True))
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
    return jsonify(ParseJson.get_lap_based_on_time("backend/JsonData/2023_Fall.json", timestamp))

@app.route("/getmodel/<int:lap>", methods=['GET'])
def get_model(lap):
    return jsonify(model(ParseJson.get_lap_history("backend/JsonData/2023_Fall.json", lap)).tolist())

@app.route("/getcrashers/<int:lap_number>")
def get_crashers(lap_number):
    input_tensor = torch.Tensor(ParseJson.get_lap_info("backend/JsonData/2023_Fall.json", 153))
    input_tensor = input_tensor.view(-1, 80)
    ig = IntegratedGradients(model)
    attr, _ = ig.attribute(input_tensor, target=0, return_convergence_delta=True)
    attr = attr.numpy().tolist()[0]
    attr = attr[0:20]
    top_indices = sorted(range(len(attr)), key=lambda i: abs(attr[i]), reverse=True)[:3]
    
    json_dict = ParseJson.parse_json("backend/JsonData/2024_Fall.json")
    driver_laps = json_dict["laps"]
    dict_index_from_position = [-1] * 40
    ret = [-1] * 3
    for i in range(len(driver_laps)):
        lap_info = driver_laps[i]["Laps"]
        if lap_number >= len(lap_info):         # Check this condition
            continue
        lap = lap_info[lap_number]
        running_position = lap["RunningPos"]
        dict_index_from_position[running_position - 1] = i
    ret[0] = driver_laps[dict_index_from_position[top_indices[0]]]["Number"]
    ret[1] = driver_laps[dict_index_from_position[top_indices[1]]]["Number"]
    ret[2] = driver_laps[dict_index_from_position[top_indices[2]]]["Number"]

    return jsonify(ret)

if __name__ == "__main__":
    # print(model(ParseJson.get_lap_history("backend/JsonData/2023_Fall.json", 100)).tolist())
    app.run(debug=True)
