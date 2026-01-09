from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('best_svm_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            features = [float(x) for x in request.form.values()]
            final_features = [np.array(features)]
            
            prediction = model.predict(final_features)
            
            output = "Risk of Heart Disease" if prediction[0] == 1 else "No Heart Disease"
            color = "red" if prediction[0] == 1 else "green"

            return render_template('index.html', prediction_text=f'Result: {output}', color=color)
        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: Please check your inputs.')

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)