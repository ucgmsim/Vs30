from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def application_root():
    return render_template("index.html")