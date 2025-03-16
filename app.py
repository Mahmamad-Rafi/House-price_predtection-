import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("Model/house_price_model.pkl", "rb"))

# Define feature names
feature_columns = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income", "ocean_proximity"
]

# Define categories for ocean_proximity
ocean_categories = {"NEAR BAY": 0, "INLAND": 1, "NEAR OCEAN": 2, "ISLAND": 3, "<1H OCEAN": 4}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get numerical inputs
        longitude = float(request.form.get("longitude"))
        latitude = float(request.form.get("latitude"))
        housing_median_age = float(request.form.get("housing_median_age"))
        total_rooms = float(request.form.get("total_rooms"))
        total_bedrooms = float(request.form.get("total_bedrooms"))
        population = float(request.form.get("population"))
        households = float(request.form.get("households"))
        median_income = float(request.form.get("median_income"))

        # Get categorical input
        ocean_proximity = request.form.get("ocean_proximity")
        ocean_category = ocean_categories.get(ocean_proximity, 0)  # Default to 0 if not found

        # Convert input into Pandas DataFrame
        input_data = pd.DataFrame([[longitude, latitude, housing_median_age, total_rooms,
                                    total_bedrooms, population, households, median_income, ocean_category]],
                                  columns=feature_columns)

        # Predict house price
        prediction = model.predict(input_data)
        output = round(prediction[0], 2)

        return render_template("index.html", prediction_text=f"Predicted House Price: ${output}")

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
