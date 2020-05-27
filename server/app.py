from flask import Flask

app = Flask(__name__)

@app.route("/")
def application_root():
    return "hello earthlings"
