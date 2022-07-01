import os
import flask
import logging

from custom_log_handler import MultiProcessSafeTimedRotatingFileHandler


app = flask.Flask("vs_api")

logfile = os.path.join(os.path.dirname(__file__), "logs/logfile.log")
os.makedirs(os.path.dirname(logfile), exist_ok=True)

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