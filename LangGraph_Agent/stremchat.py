import streamlit as st
from typing import Dict, List
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

# Set up OpenRouter configuration
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Initialize the LLM
llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    temperature=0.7,
    max_tokens=1000
)

# Custom CSS and animations
def local_css():
    st.markdown("""
    <style>
        /* Main container */
        .main {
            background-color: #f8f9fa;
        }
        
        /* Chat container */
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        /* Message bubbles */
        .message {
            padding: 12px 16px;
            margin: 8px 0;
            border-radius: 18px;
            max-width: 70%;
            animation: fadeIn 0.5s ease-out;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .user-message {
            background: linear-gradient(135deg, #6e8efb, #a777e3);
            color: white;
            border-radius: 18px 18px 0 18px;
            margin-left: auto;
        }
        
        .assistant-message {
            background: #f1f1f1;
            color: #333;
            border-radius: 18px 18px 18px 0;
        }
        
        /* Typing indicator */
        .typing-indicator {
            display: flex;
            padding: 12px 16px;
            margin: 8px 0;
            max-width: 70%;
        }
        
        .typing-dot {
            width: 8px;
            height: 8px;
            background-color: #a777e3;
            border-radius: 50%;
            margin: 0 2px;
            animation: typingAnimation 1.4s infinite ease-in-out;
        }
        
        .typing-dot:nth-child(1) {
            animation-delay: 0s;
        }
        
        .typing-dot:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .typing-dot:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes typingAnimation {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-5px); }
        }
        
        /* Input area */
        .stTextInput>div>div>input {
            border-radius: 20px !important;
            padding: 12px 16px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
            border: 1px solid #e0e0e0 !important;
        }
        
        /* Submit button */
        .stButton>button {
            border-radius: 20px !important;
            background: linear-gradient(135deg, #6e8efb, #a777e3) !important;
            color: white !important;
            border: none !important;
            padding: 10px 20px !important;
            transition: all 0.3s !important;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(106, 115, 251, 0.3) !important;
        }
        
        /* Sidebar */
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        
        /* Project explanation cards */
        .feature-card {
            background: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: all 0.3s;
            border-left: 4px solid #6e8efb;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        }
        
        .feature-card h4 {
            color: #6e8efb;
            margin-top: 0;
        }
        
        /* Title styling */
        .title {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #6e8efb, #a777e3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1.5rem;
            animation: titleAnimation 2s infinite alternate;
        }
        
        @keyframes titleAnimation {
            from { background-position: 0% 50%; }
            to { background-position: 100% 50%; }
        }
    </style>
    """, unsafe_allow_html=True)

# Define our state
class State(dict):
    message: List[Dict[str, str]]  # Type hint for IDE, linters, etc.

    def __init__(self):
        super().__init__()
        self["message"] = []

# Define our chatbot node
def chatbot(state: State) -> State:
    # Send the ENTIRE message history to the LLM
    response = llm.invoke(state["message"])
    state["message"].append({"role": "assistant", "content": response.content})
    return state

# Build our graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# Initialize session state
if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = State()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Project explanation in sidebar
def show_sidebar_explanation():
    st.sidebar.title("About This Project")
    st.sidebar.markdown("""
    <div class="feature-card">
        <h4>LangGraph Chat Assistant</h4>
        <p>An AI-powered conversational agent built with LangGraph and Mistral-7B, designed to handle complex conversations using state machines.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
    <div class="feature-card">
        <h4>How It Works</h4>
        <p>1. User sends a message<br>
        2. System processes through LangGraph<br>
        3. Mistral-7B generates response<br>
        4. Conversation state updates</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
    <div class="feature-card">
        <h4>Key Features</h4>
        <p>• Stateful conversations<br>
        • Animated UI<br>
        • Context-aware responses<br>
        • OpenRouter integration</p>
    </div>
    """, unsafe_allow_html=True)

# Streamlit app
def main():
    local_css()
    show_sidebar_explanation()
    
    st.markdown('<h1 class="title">LangGraph Chat Assistant</h1>', unsafe_allow_html=True)
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="message user-message">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="message assistant-message">{message["content"]}</div>', unsafe_allow_html=True)
    
    # Input form with submit button
    with st.form(key='chat_form'):
        user_input = st.text_input(
            "Type your message...", 
            key="user_input",
            placeholder="Ask me anything...",
            label_visibility="collapsed"
        )
        submit_button = st.form_submit_button(label="Send")
    
    if submit_button and user_input:
        # Add user message to state and display
        st.session_state.conversation_state["message"].append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Show typing indicator
        with chat_container:
            typing_html = """
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
            """
            st.markdown(typing_html, unsafe_allow_html=True)
        
        # Get assistant response
        for event in graph.stream(st.session_state.conversation_state):
            for value in event.values():
                assistant_response = value["message"][-1]["content"]
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        
        # Clear the input by rerunning
        st.rerun()

if __name__ == "__main__":
    main()