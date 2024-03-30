from flask import Flask, jsonify, request

# Import your nmrcraft module here. For example:
# from nmrcraft import some_function

app = Flask(__name__)


@app.route("/")
def home():
    return "Welcome to the nmrcraft API!"


@app.route("/api/function1", methods=["GET", "POST"])
def function1():
    if request.method == "POST":
        # Assuming you're sending data as JSON
        # data = request.json
        # Call a function from your nmrcraft project here
        # For example, result = some_function(data['input'])
        result = {"message": "Replace this with your function output"}
        return jsonify(result)
    else:
        return jsonify({"message": "This endpoint supports POST requests."})


if __name__ == "__main__":
    app.run(debug=True)
