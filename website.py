from flask import Flask, redirect, url_for, render_template, request
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/", methods=["POST", "GET"])
def check():
    if request.method == "POST":
        user = request.form["name"]
    return render_template("index.html")


if __name__ == "__main__":
    app.run()
