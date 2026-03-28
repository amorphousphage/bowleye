from flask import Blueprint, render_template, request, jsonify, send_file
import cv2
import io
from datetime import datetime

calibration_bp = Blueprint("calibration", __name__)

VIDEO_DEVICES = [
    "/dev/video0",
    "/dev/video1",
    "/dev/video2",
    "/dev/video3",
    "/dev/video4"
]

def grab_frame(device_path):
    cap = cv2.VideoCapture(device_path)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None

    print("Frame shape:", frame.shape)  # (height, width, channels)

    ret, buf = cv2.imencode(".jpg", frame)
    if not ret:
        return None

    return buf.tobytes()


@calibration_bp.route("/")
def index():
    return render_template("calibration/index.html", devices=VIDEO_DEVICES)


@calibration_bp.route("/api/frame")
def api_frame():
    device = request.args.get("device")
    if device not in VIDEO_DEVICES:
        return "Invalid device", 400

    img_bytes = grab_frame(device)
    if img_bytes is None:
        return "Could not grab frame", 500

    return send_file(
        io.BytesIO(img_bytes),
        mimetype="image/jpeg",
        as_attachment=False,
        download_name="frame.jpg",
    )


@calibration_bp.route("/api/save_calibration", methods=["POST"])
def save_calibration():
    data = request.get_json(force=True)

    print("=== Calibration data received ===")
    print(data)
    print("================================")

    return jsonify({
        "status": "ok",
        "received_at": datetime.utcnow().isoformat() + "Z",
        "data": data,
    })
