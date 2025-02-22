from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/getdata/<int:race_id>/<int:timestamp>")
def get_data(race_id, timestamp):
    return "<p>" + str(race_id) + str(timestamp) + "<p>"

if __name__ == "__main__":
    app.run()