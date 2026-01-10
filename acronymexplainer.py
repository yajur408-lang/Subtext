from google import genai
import streamlit as st

API_KEY =   # replace with your actual API key
MODEL = "gemini-2.5-flash"  # valid Gemini model

st.title("Acronym and Tone Explainer")
client = genai.Client(api_key=API_KEY)

def ask_gemini():
    question = st.text_input("Enter your acronym:")
    if not question.strip():
        st.warning("No input provided. Please enter an acronym.")
        return
    try:
        prompt = (
            "You are precise. Provide one definitive answer and explain briefly. "
            f"Text: {question}\nWhat does it mean as an acronym and tell me the tone? and slang if any. and emojis if any"
        )

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )

        answer = response.text.strip()
        if not answer:
            st.write(f'"{question}" is not a commonly recognized acronym with a standardized meaning.')
        else:
            st.write(f"Q: {question}")
            st.write(f"A: {answer}")

    except Exception as e:
        st.error(f"Server down or API error: {e}")

ask_gemini()
