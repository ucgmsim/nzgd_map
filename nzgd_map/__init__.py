import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from flask import Flask

from . import constants


def create_app(
    test_config: Mapping[str, Any] | None = None, *, instance_path: str | None = None
):
    """Build a flask app for serving."""
    app = Flask(__name__, instance_relative_config=True, instance_path=instance_path)
    app_path = Path(app.instance_path)
    app.config.from_mapping(
        SECRET_KEY=os.getenv("SECRET_KEY"),
        DATABASE=os.getenv(
            "NZGD_DATABASE_PATH", str(app_path / constants.database_file_name)
        ),
        LAST_NZGD_RETRIEVAL_DATE=os.getenv("NZGD_LAST_RETRIEVAL_DATE"),
        GEONET_STATIONS_PATH=os.getenv(
            "NZGD_GEONET_STATIONS_PATH", str(app_path / "geoNet_stats+2023-06-28.ll")
        ),
        GEONET_UPLOAD_FOLDER=str(app_path / "geonet_uploads"),
        GEONET_MAX_UPLOADS=100,
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile("config.py", silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    if not app.config["SECRET_KEY"]:
        raise RuntimeError("Set SECRET_KEY in the environment or instance/config.py.")

    # import our views and register them with the app.
    from nzgd_map import records, views

    app.register_blueprint(views.bp)
    app.register_blueprint(records.bp)

    # ensure the instance folder exists
    app_path.mkdir(exist_ok=True)

    return app
