import joblib
import numpy as np

onehot_encoder = joblib.load('oneHotEncoder.joblib')
scaler = joblib.load('standardScaler.joblib')
random_forest_model = joblib.load('random_forest_model.joblib')

brand = input("请输入品牌: ")
gear = input("请输入自动/手动档： ")
mileage = float(input("请输入行驶里程（公里）: "))
engine_size = float(input("请输入排量（升）: "))
car_age = float(input("请输入车龄（年）: "))

raw_data = np.array([[brand, gear, mileage, engine_size, car_age]])

categorical_features = raw_data[:, [0, 1]]
encoded_features = onehot_encoder.transform(categorical_features).toarray()

numeric_features = raw_data[:, [2, 3, 4]].astype(float)
scaled_numeric_features = scaler.transform(numeric_features)
processed_data = np.hstack((scaled_numeric_features, encoded_features))

predicted_price = random_forest_model.predict(processed_data)
print(f"预测的二手车价格: {predicted_price[0]:.2f} 元")