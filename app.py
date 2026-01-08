from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Model load karein jo aapne abhi banaya hai
model = joblib.load('best_svm_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Form se saari values lena [01:13:30]
        # Video ke according 13 features hain
        try:
            features = [float(x) for x in request.form.values()]
            final_features = [np.array(features)]
            
            # Prediction karna
            prediction = model.predict(final_features)
            
            output = "Risk of Heart Disease" if prediction[0] == 1 else "No Heart Disease"
            color = "red" if prediction[0] == 1 else "green"

            return render_template('index.html', prediction_text=f'Result: {output}', color=color)
        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: Please check your inputs.')

if __name__ == "__main__":
    app.run(debug=True)