import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('laptop_price_predictor.pkl')

# Feature mappings for categorical inputs
brand_map = {1: 'ASUS', 2: 'Avita', 3: 'DELL', 4: 'HP', 5: 'Lenovo', 6: 'MSI', 7: 'acer'}
processor_brand_map = {1: 'Intel', 2: 'M1'}
processor_name_map = {1: 'Core i3', 2: 'Core i5', 3: 'Core i7', 4: 'Core i9', 5: 'M1', 
                      6: 'Pentium Quad', 7: 'Ryzen 3', 8: 'Ryzen 5', 9: 'Ryzen 7', 10: 'Ryzen 9'}
ram_type_map = {1: 'DDR4', 2: 'DDR5', 3: 'LPDDR3', 4: 'LPDDR4', 5: 'LPDDR4X'}
os_map = {1: 'Mac', 2: 'Windows'}
weight_map = {1: 'Gaming', 2: 'ThinNlight'}

# Home route to display the form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Collect user input from the form
    data = []
    
    brand = int(request.form.get('brand'))
    data.extend([1 if brand_map[brand] == feature else 0 for feature in brand_map.values()])

    processor_brand = int(request.form.get('processor_brand'))
    data.extend([1 if processor_brand_map[processor_brand] == feature else 0 for feature in processor_brand_map.values()])

    processor_name = int(request.form.get('processor_name'))
    data.extend([1 if processor_name_map[processor_name] == feature else 0 for feature in processor_name_map.values()])

    ram_type = int(request.form.get('ram_type'))
    data.extend([1 if ram_type_map[ram_type] == feature else 0 for feature in ram_type_map.values()])

    os = int(request.form.get('os'))
    data.extend([1 if os_map[os] == feature else 0 for feature in os_map.values()])

    weight = int(request.form.get('weight'))
    data.extend([1 if weight_map[weight] == feature else 0 for feature in weight_map.values()])

    touchscreen = int(request.form.get('touchscreen'))
    msoffice = int(request.form.get('msoffice'))
    ram_size = float(request.form.get('ram'))
    ssd_size = float(request.form.get('ssd'))
    hdd_size = float(request.form.get('hdd'))
    graphic_card = float(request.form.get('graphic_card'))
    warranty = int(request.form.get('warranty'))

    # Append numerical inputs
    data.extend([touchscreen, msoffice, ram_size, ssd_size, hdd_size, graphic_card, warranty])

    # Convert data into a NumPy array and reshape for the model
    user_input = np.array([data])

    # Predict the log price
    log_price = model.predict(user_input)

    # Convert back to the original price
    original_price_inr = np.exp(log_price)
    original_price_lkr = original_price_inr * 3.42  # Convert to LKR
    original_price_usd = original_price_inr * 0.012  # Convert to USD

    return jsonify({
        'price_in_inr': round(original_price_inr[0], 2),
        'price_in_lkr': round(original_price_lkr[0], 2),
        'price_in_usd': round(original_price_usd[0], 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
