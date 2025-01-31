import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras
import joblib
import tensorflow as tf
import json

# importing the dataset


df= pd.read_csv("dataset/car_price_dataset.csv")

# creating two columns for year and mileage
df["Car_Age"] = 2025 - df["Year"]
df["Mileage_Range"] = pd.cut(df["Mileage"],bins=[0,50000,100000,150000,200000,np.inf],labels=["Low","Medium","High","Very High","Extremly High"])

numerical_cols = ["Car_Age","Engine_Size","Mileage","Doors","Owner_Count"]
categorical_cols = ["Brand","Fuel_Type","Transmission"]

# Create dummy variables for categorical columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

# Save the feature names for later use in the Streamlit app
feature_names = numerical_cols + [col for col in df.columns if col.startswith(tuple(categorical_cols))]

# Extract features and target
X = df[feature_names].values
y = df["Price"].values


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalize features
xscaler = MinMaxScaler(feature_range=(0,1))
X_train = xscaler.fit_transform(X_train)
X_test = xscaler.transform(X_test)

# Normalize target
yscaler = MinMaxScaler(feature_range=(0,1))
y_train = yscaler.fit_transform(y_train.reshape(-1,1))
y_test = yscaler.transform(y_test.reshape(-1,1))

# Define the model
input_dim = X_train.shape[1]
model = keras.models.Sequential()
model.add(keras.layers.Dense(units = 64, kernel_initializer="he_uniform",activation = "relu",input_dim = input_dim))
model.add(keras.layers.Dense(units = 16, kernel_initializer="he_uniform",activation = "relu"))
model.add(keras.layers.Dense(units= 1,kernel_initializer = "he_uniform",activation ="linear"))

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss = "mse",metrics=["mean_absolute_error"])

#Train the model
model.fit(X_train, y_train, batch_size = 32, epochs = 100, validation_data = (X_test,y_test))

# Save the model
model.save("car_price_model.keras")

# Save the scalers

joblib.dump(xscaler, "xscaler.pkl")
joblib.dump(yscaler, "yscaler.pkl")

# Save feature names for consistency
with open("feature_names.json", "w") as f:
    json.dump(feature_names, f)

print("Model and scalers saved successfully!")