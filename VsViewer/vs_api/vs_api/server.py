import flask
import logging

from pathlib import Path

from custom_log_handler import MultiProcessSafeTimedRotatingFileHandler


app = flask.Flask("vs_api")

logfile = Path(Path(__file__).parent / "logs/logfile.log")
logfile.mkdir(parents=True, exist_ok=True)

TRFhandler = MultiProcessSafeTimedRotatingFileHandler(filename=logfile, when="midnight")

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
    level=logging.DEBUG,
    handlers=[TRFhandler],
)

TRFhandler.setLevel(logging.DEBUG)
# To prevent having a same log twice
app.logger.propagate = False
app.logger.addHandler(TRFhandler)
logging.getLogger("matplotlib").setLevel(logging.ERROR)


# Error handler
class AuthError(Exception):
    def __init__(self, error, status_code):
        self.error = error
        self.status_code = status_code


@app.errorhandler(AuthError)
def handle_auth_error(ex):
    response = flask.jsonify(ex.error)
    response.status_code = ex.status_code
    return response


# Add the endpoints
from api import cpt