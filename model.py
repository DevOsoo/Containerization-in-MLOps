# Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib


# Load the iris dataset
iris = load_iris()
X = iris.data # Features
y = iris.target # Labels


# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model on the training set
knn.fit(X_train, y_train)

# Predict the labels on the test set
y_pred = knn.predict(X_test)

# Evaluate the accuracy of the model
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")


# Save the model as a pickle in a file
joblib.dump(knn, 'model.joblib') 

# Load the model from the file
model = joblib.load('model.joblib') # Add this line to load the model