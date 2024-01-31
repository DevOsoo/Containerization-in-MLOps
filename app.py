# Import the modules
from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from joblib import load

# Load the iris dataset and the trained model
iris = load_iris()
model = load("model.joblib")

# Create a Flask app
app = Flask(__name__)

# Define a route for the prediction
@app.route("/predict", methods=["GET", "POST"])
def predict():
  # If the request is a GET request
  if request.method == "GET":
    # Return a simple HTML form that allows the user to enter the features
    return """
    <html>
      <head>
        <title>Iris Prediction</title>
      </head>
      <body>
        <h1>Iris Prediction</h1>
        <form action="/predict" method="post">
          <p>Enter the sepal length (cm): <input type="number" name="sepal_length" /></p>
          <p>Enter the sepal width (cm): <input type="number" name="sepal_width" /></p>
          <p>Enter the petal length (cm): <input type="number" name="petal_length" /></p>
          <p>Enter the petal width (cm): <input type="number" name="petal_width" /></p>
          <p><input type="submit" value="Predict" /></p>
        </form>
      </body>
    </html>
    """
  # If the request is a POST request
  elif request.method == "POST":
    # Get the features from the request body or form
    features = request.get_json() or request.form

    # Validate the features
    if not features or len(features) != 4:
      return jsonify({"error": "Invalid features"}), 400

    # Predict the species using the model
    prediction = model.predict([features])[0]
    species = iris.target_names[prediction]

    # Return the prediction as a JSON response
    return jsonify({"species": species}), 200


# Run the app
if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000)
