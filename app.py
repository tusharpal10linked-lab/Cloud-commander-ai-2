import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Load environment
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.error("Missing OPENAI_API_KEY in .env")
    st.stop()

# Initialize OpenAI
client = OpenAI(api_key=OPENAI_KEY)

# Constants
DATA_DIR = "data"
COST_CSV = os.path.join(DATA_DIR, "synthetic_aws_billing_data.csv")
PRED_CSV = os.path.join(DATA_DIR, "batch_output.csv")

# Load data
@st.cache_data
def load_data():
    cost_df = pd.read_csv(COST_CSV)
    pred_df = pd.read_csv(PRED_CSV)
    return cost_df, pred_df

cost_df, pred_df = load_data()

# UI
st.title("AWS Cost Optimization Chat + Analyzer")

page = st.sidebar.radio("Go to:", ["Chat with Assistant", "View Tables"])

if page == "Chat with Assistant":
    st.header("💬 Chat with Assistant")
    st.markdown("You can ask questions about the cost prediction data, such as “Which services show the highest predicted costs?” or “Which ones should I stop?”")

    # Provide preview for context
    st.subheader("⚠️ Flagged Predictions (Prediction > threshold)")
    threshold = st.slider("Threshold for predicted cost", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
    flagged = pred_df[pred_df["PredictedCost"] > threshold]
    st.dataframe(flagged)

    prompt = st.text_input("Ask your Assistant:")
    if st.button("Send"):
        context = flagged.to_string(index=False)
        messages = [
            {"role": "system", "content": "You are an AWS cost optimization assistant."},
            {"role": "user", "content": f"Here is the table of flagged predictions:\n{context}\nUser question: {prompt}"}
        ]
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7
            )
            assistant_reply = response.choices[0].message.content
        except Exception as e:
            assistant_reply = f"Error contacting OpenAI: {e}"

        st.subheader("Assistant Response")
        st.write(assistant_reply)

elif page == "View Tables":
    st.header("📊 View Data Tables")
    st.subheader("Raw Cost Data")
    st.dataframe(cost_df.head(20))

    st.subheader("Prediction Data")
    st.dataframe(pred_df.head(20))

    st.subheader("Flagged Predictions")
    threshold = st.slider("Threshold for predicted cost", min_value=0.0, max_value=100.0, value=20.0, step=1.0, key="thresh2")
    flagged2 = pred_df[pred_df["PredictedCost"] > threshold]
    st.dataframe(flagged2)

    st.markdown("**Note:** In this version you cannot trigger AWS Lambda from the UI—this is a platform-agnostic staging version.")
