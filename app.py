import streamlit as st
from tensorflow.keras.models import load_model

print("Script is running")

st.set_page_config(page_title="Cuisine Guide Chatbot", page_icon="🍽️", layout="centered")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    ...
    </style>
""",
    unsafe_allow_html=True,
)

# # Display message from bot

# with st.chat_message("assistant"):
#     st.write("Hello! I'm a bot")

# # Display message from user
# with st.chat_message("user"):
#     st.write("Hi bot!")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Accept input from user
if user_input := st.chat_input("Say something..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

# Send sample bot response

bot_response = f"You said : {user_input}"

# Add bot message to history
st.session_state.messages.append({"role": "assistant", "content": bot_response})

# st.rerun()


# Cache model loading so as to improve performance
@st.cache_resource
def load_my_model():
    model = load_model("chatbot_model.keras")
    return model


model = load_my_model()
