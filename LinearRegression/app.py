import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
# Streamlit Title
st.title("House Price Prediction")
# Load Dataset
df = pd.read_csv("Housing.csv")

# Encode categorical columns
encoder = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = encoder.fit_transform(df[col])
st.write("Dataset:")
st.dataframe(df.head())
# Features and Target
X = df.drop("price", axis=1)
y = df["price"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()

model.fit(X_train, y_train)



# User Inputs
area = st.number_input("Area")
bedrooms = st.number_input("Bedrooms")
bathrooms = st.number_input("Bathrooms")
stories = st.number_input("Stories")
parking = st.number_input("Parking")

# Predict Button
if st.button("Predict"):

    input_data = [[
        area,
        bedrooms,
        bathrooms,
        stories,
        1,
        0,
        0,
        0,
        1,
        parking,
        1,
        0
    ]]

    prediction = model.predict(input_data)

    st.success(f"Predicted Price: {prediction[0]:,.2f}")

# Regression Line Graph
st.subheader("Regression Line")

plt.figure(figsize=(8,5))

plt.scatter(df["area"], df["price"])

single_X = df[["area"]]

single_model = LinearRegression()

single_model.fit(single_X, y)

plt.plot(
    df["area"],
    single_model.predict(single_X),
    linewidth=3
)

plt.xlabel("Area")
plt.ylabel("Price")

st.pyplot(plt)