import streamlit as st
import sys
import asyncio
from pathlib import Path

# Initialize event loop for Streamlit
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from data_processing.query_vectorstore import query_index

# Custom CSS styling
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Global text styling */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp p, .stApp div, .stApp span {
        color: #333333;
    }
    
    /* Chat container styling */
    .chat-container {
        max-width: 640px;
        margin: 0 auto;
        background: white;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        padding: 1rem;
    }
    
    /* Message styling */
    .user-message {
        background-color: #007bff;
        color: #333333;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        max-width: 70%;
        margin-left: auto;
    }
    
    .bot-message {
        background-color: #f8f9fa;
        color: #333333;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        max-width: 70%;
        border: 1px solid #dee2e6;
    }
    
    /* Input area styling */
    .stTextInput > div > div > input {
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        border: 2px solid #007bff;
        background-color: #f8f9fa;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #0056b3;
        box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: none;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #0056b3;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("Bicycle Manuals LLM Assistant")
    
    # Initialize chat history in session state if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I help you with bicycle manuals today?"}
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add assistant response to chat history
        with st.chat_message("assistant"):
            # Get response from the query processing system
            response = query_index(prompt)
            if response:
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                error_message = "I apologize, but I encountered an error processing your query. Please try again."
                st.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()
