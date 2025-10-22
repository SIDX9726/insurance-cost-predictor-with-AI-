import streamlit as st
import pandas as pd
import joblib
import google.generativeai as genai

# --- CONFIGURE GEMINI API ---
genai.configure(api_key="AIzaSyApRxv3ghHhfbmp2kjIFkDSe2enacR7Vco")

# Load ML model
model = joblib.load("insurance_model.pkl")

# Initialize Gemini model globally
model_ai = genai.GenerativeModel("gemini-2.5-flash")

st.set_page_config(page_title="Medical Cost Predictor + AI Assistant", page_icon="🏥", layout="centered")

st.title("🏥 Medical Insurance Cost Predictor + AI Assistant")
st.write("Predict your insurance cost and get an AI explanation for it.")

# --- Input fields ---
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sex = st.selectbox("Sex", ("male", "female"))
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ("yes", "no"))
region = st.selectbox("Region", ("southwest", "southeast", "northwest", "northeast"))

# Map region to numeric encoding
region_map = {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}

# Prepare input DataFrame
input_df = pd.DataFrame({
    "age": [age],
    "sex": [1 if sex == "male" else 0],
    "bmi": [bmi],
    "children": [children],
    "smoker": [1 if smoker == "yes" else 0],
    "region": [region_map[region]]
})

# --- Predict and Explain ---
if st.button("🔮 Predict"):
    pred = model.predict(input_df)[0]
    st.success(f"💰 **Estimated Insurance Cost: ${pred:.2f}**")

    prompt = f"""
    A person with the following details:
    - Age: {age}
    - Sex: {sex}
    - BMI: {bmi}
    - Children: {children}
    - Smoker: {smoker}
    - Region: {region}
    got an estimated insurance cost of ${pred:.2f}.
    Explain in simple, friendly language why this cost might be high or low.
    """

    # Spinner animation while AI generates response
    with st.spinner("🧠 AI is thinking... generating explanation..."):
        ai_response = model_ai.generate_content(prompt)
    
    st.subheader("💬 AI Explanation:")
    st.write(ai_response.text)

# --- Chatbot Section ---
st.write("---")
st.subheader("💭 Ask the AI Assistant Anything")
user_query = st.text_input("Ask something about insurance, health, or cost factors:")

if user_query:
    with st.spinner("🤖 AI is thinking..."):
        chatbot_prompt = f"You are an insurance assistant. Answer this question simply and clearly: {user_query}"
        chat_response = model_ai.generate_content(chatbot_prompt)
    st.write(chat_response.text)
