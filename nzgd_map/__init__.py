from pathlib import Path
from typing import Any

from flask import Flask


def create_app(test_config: Any = None):
    """Build a flask app for serving."""
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app_path = Path(app.instance_path)

    # Set a secret key for session management
    app.config["SECRET_KEY"] = (
        "dev-key-for-nzgd-map-app"  # In production, use a proper secret key
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile("config.py", silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # import our views and register them with the app.
    from nzgd_map import views

    app.register_blueprint(views.bp)

    # ensure the instance folder exists
    app_path.mkdir(exist_ok=True)

    return app
