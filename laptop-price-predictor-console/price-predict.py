import numpy as np
import joblib

# Load the trained model
model = joblib.load('laptop_price_predictor.pkl')

# Feature mapping for numeric inputs
brand_map = {1: 'ASUS', 2: 'Avita', 3: 'DELL', 4: 'HP', 5: 'Lenovo', 6: 'MSI', 7: 'acer'}
processor_brand_map = {1: 'Intel', 2: 'M1'}
processor_name_map = {1: 'Core i3', 2: 'Core i5', 3: 'Core i7', 4: 'Core i9', 5: 'M1', 
                      6: 'Pentium Quad', 7: 'Ryzen 3', 8: 'Ryzen 5', 9: 'Ryzen 7', 10: 'Ryzen 9'}
ram_type_map = {1: 'DDR4', 2: 'DDR5', 3: 'LPDDR3', 4: 'LPDDR4', 5: 'LPDDR4X'}
os_map = {1: 'Mac', 2: 'Windows'}
weight_map = {1: 'Gaming', 2: 'ThinNlight'}

# Collect user input
def get_user_input():
    data = []
    
    # Brand
    print("Select Brand:")
    for k, v in brand_map.items():
        print(f"{k}. {v}")
    brand = int(input("Enter the number corresponding to the brand: "))
    for feature in brand_map.values():
        data.append(1 if brand_map[brand] == feature else 0)

    # Processor Brand
    print("\nSelect Processor Brand:")
    for k, v in processor_brand_map.items():
        print(f"{k}. {v}")
    processor_brand = int(input("Enter the number corresponding to the processor brand: "))
    for feature in processor_brand_map.values():
        data.append(1 if processor_brand_map[processor_brand] == feature else 0)

    # Processor Name
    print("\nSelect Processor Name:")
    for k, v in processor_name_map.items():
        print(f"{k}. {v}")
    processor_name = int(input("Enter the number corresponding to the processor name: "))
    for feature in processor_name_map.values():
        data.append(1 if processor_name_map[processor_name] == feature else 0)

    # RAM Type
    print("\nSelect RAM Type:")
    for k, v in ram_type_map.items():
        print(f"{k}. {v}")
    ram_type = int(input("Enter the number corresponding to the RAM type: "))
    for feature in ram_type_map.values():
        data.append(1 if ram_type_map[ram_type] == feature else 0)

    # OS
    print("\nSelect OS:")
    for k, v in os_map.items():
        print(f"{k}. {v}")
    os = int(input("Enter the number corresponding to the OS: "))
    for feature in os_map.values():
        data.append(1 if os_map[os] == feature else 0)

    # Weight Category
    print("\nSelect Weight Category:")
    for k, v in weight_map.items():
        print(f"{k}. {v}")
    weight = int(input("Enter the number corresponding to the weight category: "))
    for feature in weight_map.values():
        data.append(1 if weight_map[weight] == feature else 0)

    # Touchscreen
    touchscreen = int(input("\nIs it touchscreen? (1 for Yes, 0 for No): "))
    data.append(touchscreen)

    # MS Office
    msoffice = int(input("\nDoes it include MS Office? (1 for Yes, 0 for No): "))
    data.append(msoffice)

    # Numeric features
    data.append(float(input("\nEnter RAM size (GB): ")))
    data.append(float(input("Enter SSD size (GB): ")))
    data.append(float(input("Enter HDD size (GB): ")))
    data.append(float(input("Enter graphic card memory (GB): ")))
    data.append(int(input("Enter warranty period (years): ")))

    return np.array([data])

# Predict laptop price
def predict_price():
    print("Laptop Price Prediction")
    print("========================")
    user_input = get_user_input()
    log_price = model.predict(user_input)  # Logarithmic price
    original_price_inr = np.exp(log_price)    # Convert back to original price in INR
    original_price_lkr = original_price_inr * 3.42  # Convert to LKR
    original_price_usd = original_price_inr * 0.012  # Convert to USD
    #print(f"\nPredicted Laptop Price in INR: {original_price_inr[0]:.2f}")
    print(f"Predicted Laptop Price in LKR: {original_price_lkr[0]:.2f}")
    print(f"Predicted Laptop Price in USD: {original_price_usd[0]:.2f}")

if __name__ == "__main__":
    predict_price()
