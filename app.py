from flask import Flask, request, render_template
import pickle
import pandas as pd
import statsmodels.api as sm

app = Flask(__name__)

# Load the ARIMA models
delhi_model = pickle.load(open('delhi_model.pkl', 'rb'))
mumbai_model = pickle.load(open('mumbai_model.pkl', 'rb'))
chennai_model = pickle.load(open('chennai_model.pkl', 'rb'))
kolkata_model = pickle.load(open('kolkata_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/About_Us')
def About_Us():
    return render_template('about.html')

@app.route('/Contact_Us', methods=['GET', 'POST'])
def Contact_Us():
    if request.method == "POST":
        name = request.form.get('name')
        phnnumber = request.form.get('phnnumber')
        mail = request.form.get('mail')
        message = request.form.get('message')
        print(f"Name: {name}\nPhone Number: {phnnumber}\nEmail: {mail}\nMessage/Query: {message}")
        return render_template('thankyou.html', Name=name)
    return render_template('contact.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        city_code = request.form['city']
        city_names = {'0': 'DELHI (P)', '1': 'MUMBAI(P)', '2': 'CHENNAI(P)', '3': 'KOLKATA(P)'}
        city_name = city_names[city_code]
        prediction_type = request.form['prediction-type']
        
        # Load the petrol price data for the selected city
        petrol_data = pd.read_csv('petrol.csv')
        city_data = petrol_data[city_name].copy()
        
        # Make predictions using the corresponding model
        if city_name == 'DELHI (P)':
            model = delhi_model
        elif city_name == 'MUMBAI(P)':
            model = mumbai_model
        elif city_name == 'CHENNAI(P)':
            model = chennai_model
        elif city_name == 'KOLKATA(P)':
            model = kolkata_model
        
        predictions = model.predict(start=len(city_data), end=len(city_data) + int(prediction_type) - 1)
        
        # Format the predictions as a list
        prediction_list = predictions.tolist()
        
        return render_template('result.html', city_name=city_name, prediction_type=prediction_type, predictions=prediction_list)
    
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
