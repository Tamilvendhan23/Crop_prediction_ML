from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the trained model and scalers
model = pickle.load(open('model.pkl', 'rb'))
minmax_scaler = pickle.load(open('minmaxscaler.pkl', 'rb'))
standard_scaler = pickle.load(open('standscaler.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

app = Flask(__name__)

# Home route displaying index.html
@app.route('/')
def index():
    return render_template("index.html")

@app.route("/guide_crop")
def guide_crop():
    return render_template("crop.html")


# Crop recommendation page
@app.route("/crop_suggestion")
def crop_suggestion():
    return render_template("newcrop.html")

# Route to handle prediction when the form is submitted
@app.route("/predict", methods=['POST'])
def predict():
    # Get form input values
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    # Create feature array and preprocess
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Apply scaling transformations
    scaled_minmax = minmax_scaler.transform(single_pred)
    final_features = standard_scaler.transform(scaled_minmax)

    # Predict and decode crop label
    prediction = model.predict(final_features)
    crop_name = label_encoder.inverse_transform(prediction)[0]

    # Return result to be displayed on crop.html
    result = f"{crop_name} is the best crop to be cultivated right there"
    return render_template('newcrop.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
