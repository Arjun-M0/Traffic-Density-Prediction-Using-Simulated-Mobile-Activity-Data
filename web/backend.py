from flask import Flask, render_template, request
import random

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":

        # Temporary random prediction (replace with real model later)
        value = random.randint(50, 500)

        if value < 150:
            prediction = "Low"
        elif value < 300:
            prediction = "Medium"
        else:
            prediction = "High"

        return render_template("result.html",
                               prediction=prediction,
                               value=value)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)