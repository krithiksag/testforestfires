from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Initialize Flask application
application = Flask(__name__)
app = application

# Load models with error handling
try:
    ridge_model = pickle.load(open(r'D:\vscode\udemy\machinelearning\project\models\ridge.pkl', 'rb'))
    scaler_model = pickle.load(open(r'D:\vscode\udemy\machinelearning\project\notebooks\scaler.pkl', 'rb'))
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure the model files exist in the 'models' directory.")
    ridge_model, scaler_model = None, None

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Retrieve form data
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))
            
            # Prepare data for prediction
            new_data = [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
            new_data_scaled = scaler_model.transform(new_data)  # Use the loaded scaler instance
            
            # Predict using the ridge model
            result = ridge_model.predict(new_data_scaled)
            return render_template('home.html', result=result[0])
        except Exception as e:
            return render_template('home.html', result=f"Error: {e}")
    else:
        return render_template("home.html", result=None)

# Run the application
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
