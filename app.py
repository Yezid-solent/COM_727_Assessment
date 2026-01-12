import streamlit as st
from tensorflow.keras.models import load_model
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

# Import chatbot functions from chatbot.py
from chatbot import predict_class, get_response, intents

print("Script is running")

st.set_page_config(page_title="Cuisine Guide Chatbot", page_icon="🍽️", layout="centered")
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color: #f0f2f6;
#     }
#     ...
#     </style>
# """,
#     unsafe_allow_html=True,
# )
st.markdown(
    """
    <style>
    /* Improve overall readability */
    body {
        color: inherit;
    }

    /* Chat message containers */
    .stChatMessage {
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 8px;
    }

    /* User messages */
    .stChatMessage.user {
        background-color: rgba(0, 123, 255, 0.15);
    }

    /* Assistant messages */
    .stChatMessage.assistant {
        background-color: rgba(40, 167, 69, 0.15);
    }

    /* Input box */
    textarea {
        border-radius: 8px !important;
    }

    /* Buttons */
    button {
        border-radius: 8px !important;
        font-weight: 600;
    }

    /* Header text */
    h1, h2, h3 {
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Load model
@st.cache_resource
def load_chatbot_resources():
    lemmatizer = WordNetLemmatizer()
    intents = json.load(open("intents.json"))
    words = pickle.load(open("words.pkl", "rb"))
    classes = pickle.load(open("classes.pkl", "rb"))
    model = load_model("chatbot_model.keras")
    return lemmatizer, intents, words, classes, model


lemmatizer, intents, words, classes, model = load_chatbot_resources()


# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

    # Show welcome message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": "Hello! I can help you find places to eat in the Southampton. What would you like to try today?",
        }
    )


# Disclaimer
st.warning(
    """ 
    This is a  prototype demonstrating an AI chatbot. 
    Restaurant data is limited and for educational purposes only.

    Please verify restaurant information through Google Maps, Yelp, 
    or official sources before visiting.

    Not intended for real-world use.
    """
)
# App Header
st.title("🍽️ Cuisine Guide Chatbot")
st.markdown(
    "*Ask me about where to find English, Nigerian, Indian, Chinese, Italian, or Mexican food!*"
)
st.divider()

# Display all previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Accept input from user
if user_input := st.chat_input("Say something..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get and display bot response
    with st.chat_message("assistant"):
        with st.spinner("Searching ......"):
            intents_list = predict_class(user_input)
            response = get_response(intents_list, intents)
        st.markdown(response)

    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

st.divider()

# Add button to clear chat

if st.button("Clear chat"):
    st.session_state.messages = []
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": "Hello! I can help you find places to eat in the Southampton. What would you like to try today?",
        }
    )

    st.rerun()

# st.divider()

st.caption("Built by the team")
