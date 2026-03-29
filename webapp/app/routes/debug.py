from flask import Blueprint, render_template, send_file, abort, request
import numpy as np
import io
import cv2

debug_bp = Blueprint("debug", __name__)

# This will be set from your coordinator
DEBUG_IMAGES = {}

@debug_bp.route("/view/<name>")
def view_debug(name):
    return render_template("debug/view.html", name=name)

@debug_bp.route("/img/<name>")
def debug_image(name):
    img = DEBUG_IMAGES.get(name)
    if img is None:
        abort(404)

    ok, buf = cv2.imencode(".png", img)
    if not ok:
        abort(500)

    return send_file(
        io.BytesIO(buf.tobytes()),
        mimetype="image/png",
        as_attachment=False
    )

@debug_bp.route("/upload/<name>", methods=["POST"])
def upload_debug(name):
    file = request.files.get("image")
    if not file:
        return "No image uploaded", 400

    # Read image into OpenCV format
    file_bytes = file.read()
    img_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return "Invalid image", 400

    DEBUG_IMAGES[name] = img
    return "OK", 200
