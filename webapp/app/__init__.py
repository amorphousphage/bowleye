from flask import Flask
from .config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Register blueprints
    from .routes.calibration import calibration_bp
    app.register_blueprint(calibration_bp, url_prefix="/calibration")

    from .routes.debug import debug_bp
    app.register_blueprint(debug_bp, url_prefix="/debug")

    from .routes.live import live_bp
    app.register_blueprint(live_bp, url_prefix="/live")

    return app
