from flask import Flask, jsonify
from flask_cors import CORS
from ParseJson import ParseJson

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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
    app.run(debug=True)
