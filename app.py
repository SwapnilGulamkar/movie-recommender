"""Flask API + static frontend for the movie recommender."""
import os

from flask import Flask, jsonify, request, send_from_directory

from recommender import Recommender

app = Flask(__name__, static_folder="static", static_url_path="")
rec = Recommender()


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/genres")
def genres():
    return jsonify(rec.genres())


@app.get("/api/search")
def search():
    return jsonify(rec.search(request.args.get("q", "")))


@app.get("/api/browse")
def browse():
    return jsonify(rec.browse(request.args.get("genre"), int(request.args.get("limit", 12))))


@app.post("/api/recommend")
def recommend():
    body = request.get_json(silent=True) or {}
    liked = body.get("liked", [])
    if not isinstance(liked, list) or not liked:
        return jsonify({"error": "Pick at least one movie you like."}), 400
    n = min(int(body.get("n", 8)), 20)
    return jsonify(rec.recommend([int(i) for i in liked], n=n, genre=body.get("genre") or None))


if __name__ == "__main__":
    # Local run. In production the host starts the app with gunicorn instead.
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", host="0.0.0.0",
            port=int(os.environ.get("PORT", 5000)))
