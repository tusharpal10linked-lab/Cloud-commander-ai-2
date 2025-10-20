import os
import streamlit as st
import pandas as pd
import openai

# === PAGE CONFIG ===
st.set_page_config(page_title="AWS Cost Optimization Assistant", layout="wide")

st.title("💰 AWS Cost Optimization Assistant")
st.markdown("Analyze, chat, and optimize your AWS cost data interactively!")

# === STEP 1: Enter API Key ===
OPENAI_KEY = st.text_input("🔑 Enter your OpenAI API Key:", type="password")

if not OPENAI_KEY:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

# Set API key for openai
openai.api_key = OPENAI_KEY

# === STEP 2: Load Data ===
DATA_DIR = "data"
COST_CSV = os.path.join(DATA_DIR, "synthetic_aws_billing_data.csv")
PRED_CSV = os.path.join(DATA_DIR, "batch_output.csv")

@st.cache_data
def load_data():
    cost_df = pd.read_csv(COST_CSV)
    pred_df = pd.read_csv(PRED_CSV)
    return cost_df, pred_df

cost_df, pred_df = load_data()

# === STEP 3: Sidebar Navigation ===
page = st.sidebar.radio("📘 Go to Page:", ["Chat Assistant", "View Cost Tables"])

# === STEP 4: Chat Assistant Page ===
if page == "Chat Assistant":
    st.header("💬 Chat with Your AWS Cost Assistant")

    threshold = st.slider("Flag resources above cost threshold (USD):", 0, 100, 20)
    flagged_df = pred_df[pred_df["PredictedCost"] > threshold]

    st.subheader("⚠️ Flagged High-Cost Predictions")
    st.dataframe(flagged_df, use_container_width=True)

    # Input box for user query
    user_input = st.text_area("🧑‍💻 Ask a question about your cost data:")

    if st.button("Send"):
        if not user_input.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("💭 Thinking..."):
                context = flagged_df.to_string(index=False)
                prompt = f"""
                You are an AWS cost optimization assistant.
                Here is the current cost data table:
                {context}
                Now answer this user question: {user_input}
                """

                try:
                    response = openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                    )
                    answer = response.choices[0].message.content
                    st.success("✅ Assistant Reply")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error connecting to OpenAI: {e}")

# === STEP 5: View Tables Page ===
elif page == "View Cost Tables":
    st.header("📊 AWS Cost Data Viewer")

    st.subheader("Raw Cost Data")
    st.dataframe(cost_df.head(20), use_container_width=True)

    st.subheader("Predicted Cost Data")
    st.dataframe(pred_df.head(20), use_container_width=True)

    threshold = st.slider("Highlight costs above threshold", 0, 100, 20, key="thresh2")
    flagged2 = pred_df[pred_df["PredictedCost"] > threshold]

    st.subheader("⚠️ Flagged Services")
    st.dataframe(flagged2, use_container_width=True)

    st.info("This version does not trigger AWS Lambda; it focuses on interactive chat and analysis.")
