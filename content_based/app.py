from flask import Flask, request, current_app, abort, jsonify
from functools import wraps
from recommend import recommendation_engine

app = Flask(__name__)
app.config.from_object("settings")


def token_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.headers.get("X-API-TOKEN", None) != current_app.config["API_TOKEN"]:
            abort(403)
        return f(*args, **kwargs)

    return decorated_function


@app.route("/predict", methods=["POST"])
@token_auth
def predict():

    item = request.json.get("item")
    num_predictions = request.json.get("num", 10)
    if not item:
        return jsonify({"query": None, "recommendations": []})
    return jsonify(recommendation_engine.predict(str(item), num_predictions))


@app.route("/train", methods=["POST"])
@token_auth
def train():
    data_url = request.json.get("data-url", None)
    recommendation_engine.train(data_url)
    return jsonify({"message": "Success!", "success": 1})


if __name__ == "__main__":
    app.debug = True
    app.run()
