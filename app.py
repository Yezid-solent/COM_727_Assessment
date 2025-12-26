import streamlit as st

print("Script is running")

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
