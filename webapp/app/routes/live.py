from flask import Blueprint, render_template, request, send_file, current_app
from app.services.game_manager import game_manager
import os

live_bp = Blueprint("live", __name__)

@live_bp.route("/session")
def live_session():
    return render_template("live/session.html")

@live_bp.route("/upload_video", methods=["POST"])
def upload_video():
    file = request.files.get("video")
    if not file:
        return "No video uploaded", 400

    # Save to disk
    save_path = "static/videos/latest_pin_video.mp4"
    file.save(save_path)

    return "OK", 200

@live_bp.route("/upload_score", methods=["POST"])
def upload_score():
    data = request.json
    fallen_pins = data.get("fallen_pins", [])
    game_manager.ProcessShot(fallen_pins)
    return "OK", 200

@live_bp.route("/game_state")
def game_state():
    return game_manager.GetGameState()

@live_bp.route("/pin_video")
def pin_video():
    full_path = current_app.root_path + "/static/videos/latest_pin_video.mp4"
    return send_file(full_path, mimetype="video/mp4", as_attachment=False)

