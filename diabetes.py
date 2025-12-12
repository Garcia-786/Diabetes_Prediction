import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("Diabetes Prediction")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Diabetes Prediction\diabetes - diabetes.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# DATA INSIGHTS SECTION
st.subheader("Data Insights")

col1, col2 = st.columns(2)

with col1:
    st.write("Outcome Count")
    fig, ax = plt.subplots()
    sns.countplot(x=df["Outcome"], ax=ax)
    st.pyplot(fig)

with col2:
    st.write("Glucose Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Glucose"], kde=True, ax=ax)
    st.pyplot(fig)

st.write("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(df.corr(), annot=False, cmap="Blues")
st.pyplot(fig)

# MODEL TRAINING
st.subheader("Model Training")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4
)

model = RandomForestClassifier(random_state=4)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.markdown(
    f"<h3 style='color:#1a73e8; text-align:center;'>Model Accuracy: {accuracy:.2f}</h3>",
    unsafe_allow_html=True)

# USER INPUT SECTION
st.subheader("Predict for Custom Input")

preg = st.number_input("Pregnancies", 0, 20, 1)
glu = st.number_input("Glucose", 0, 300, 120)
bp = st.number_input("Blood Pressure", 0, 200, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
ins = st.number_input("Insulin", 0, 900, 85)
bmi = st.number_input("BMI", 0.0, 70.0, 25.5)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.45)
age = st.number_input("Age", 1, 120, 30)

input_data = [[preg, glu, bp, skin, ins, bmi, dpf, age]]

if st.button("Predict"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.markdown(
            "<h3 style='color:red; text-align:center;'>The model predicts: High chance of diabetes</h3>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            "<h3 style='color:green; text-align:center;'>The model predicts: Low chance of diabetes</h3>",
            unsafe_allow_html=True)
