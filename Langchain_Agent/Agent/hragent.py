import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain.agents import initialize_agent, Tool
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import os
import time

# -------------------------------
# 🔐 1. Load environment variables
# -------------------------------
load_dotenv()

DB_HOST = os.getenv("MYSQL_HOST")
DB_USER = os.getenv("MYSQL_USER")
DB_PASS = os.getenv("MYSQL_PASSWORD")
DB_NAME = os.getenv("MYSQL_DB")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_PORT = os.getenv("MYSQL_PORT", "3306")

# -------------------------------
# 🔗 2. Setup DB connection
# -------------------------------
DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DB_URL)
db = SQLDatabase(engine)

# -------------------------------
# 🔧 3. Define the SQL Tool with @tool

@tool
def hr_sql_tool(query: str) -> str:
    """Answer HR questions using SQL database. Returns raw data for formatting."""
    try:
        # Get raw results without any LLM interpretation
        result = db.run(query)
        return str(result)  # Return as string to be parsed later
    except Exception as e:
        # Error handling remains the same
        print("⚠ Query failed. Trying auto-correction...")
        schema = db.get_table_info()
        correction_prompt = f"""
            Rewrite this SQL query using correct schema:
            Schema: {schema}
            Query: {query}
            Only return the corrected SQL query.
            """
        try:
            corrected_query = llm.predict(correction_prompt).strip()
            print(f"🛠 Corrected SQL:\n{corrected_query}")
            corrected_result = db.run(corrected_query)
            return str(corrected_result)
        except Exception as inner_e:
            return f"❌ Error: {str(inner_e)}"


# -------------------------------
# 🤖 4. Setup LangChain Agent
# -------------------------------
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

agent = initialize_agent(
    tools=[hr_sql_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    return_intermediate_steps=True,  # Add this to get raw data
    handle_parsing_errors=True,
    max_iterations=10,
    agent_kwargs={
        "prefix": """You are an HR data assistant. Follow these rules:
        1. Always return raw data from queries
        2. Never add interpretation unless asked
        3. Return complete data sets""",
        # ... rest of your config
    }
)
# -------------------------------
# 🎨 Ultra-Visual Streamlit UI
# -------------------------------
def main():
    # Page configuration
    st.set_page_config(
        page_title="✨ HR Assistant Pro+",
        page_icon="🧑‍💼",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Dynamic CSS for light/dark mode
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@500;600;700&display=swap');
        
        :root {
            --primary: #6e48aa;
            --secondary: #9d50bb;
            --accent: #4776e6;
            --text: #2d3748;
            --bg: #f8f9fa;
            --card-bg: #ffffff;
            --error: #ff4d4f;
        }
        
        [data-theme="dark"] {
            --primary: #9d50bb;
            --secondary: #6e48aa;
            --accent: #4776e6;
            --text: #f8f9fa;
            --bg: #1a1a2e;
            --card-bg: #16213e;
            --error: #ff6b6b;
        }
        
        * {
            font-family: 'Poppins', sans-serif;
            transition: all 0.3s ease;
        }
        
        h1, h2, h3 {
            font-family: 'Montserrat', sans-serif;
        }
        
        .main {
            background: var(--bg);
            color: var(--text);
        }
        
        .stTextInput input {
            border-radius: 25px !important;
            padding: 14px 20px !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
            border: 2px solid var(--primary) !important;
            background: var(--card-bg) !important;
            color: var(--text) !important;
            font-size: 16px !important;
        }
        
        .chat-message {
            padding: 18px 22px;
            border-radius: 22px;
            margin: 12px 0;
            max-width: 82%;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            animation: fadeIn 0.4s cubic-bezier(0.22, 1, 0.36, 1);
            position: relative;
            overflow: hidden;
        }
        
        .chat-message::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 100%);
            z-index: 1;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(15px) scale(0.98); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }
        
        .user-message {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            margin-left: auto;
            border-top-right-radius: 8px;
        }
        
        .assistant-message {
            background: var(--card-bg);
            margin-right: auto;
            border-top-left-radius: 8px;
            border: 1px solid rgba(0,0,0,0.1);
            color: var(--text);
        }
        
        .sidebar .sidebar-content {
            background: var(--card-bg) !important;
            box-shadow: 5px 0 15px rgba(0,0,0,0.1);
        }
        
        .welcome-banner {
            background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
            color: white;
            padding: 25px;
            border-radius: 18px;
            margin-bottom: 25px;
            animation: pulse 2.5s infinite, float 6s ease-in-out infinite;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
            position: relative;
            overflow: hidden;
        }
        
        .welcome-banner::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0) 70%);
            animation: rotate 15s linear infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255,154,158,0.4); }
            50% { transform: scale(1.01); }
            100% { transform: scale(1); box-shadow: 0 0 0 15px rgba(255,154,158,0); }
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
        }
        
        @keyframes rotate {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        .feature-card {
            background: var(--card-bg);
            border-radius: 15px;
            padding: 18px;
            margin: 12px 0;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            border: 1px solid rgba(0,0,0,0.05);
        }
        
        .feature-card:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 12px 30px rgba(0,0,0,0.15);
        }
        
        .feature-card b {
            color: var(--primary);
            font-weight: 600;
        }
        
        .typing-indicator {
            display: flex;
            padding: 10px 15px;
        }
        
        .typing-dot {
            width: 8px;
            height: 8px;
            background-color: var(--text);
            border-radius: 50%;
            margin: 0 3px;
            animation: typingAnimation 1.4s infinite ease-in-out;
        }
        
        .typing-dot:nth-child(1) { animation-delay: 0s; }
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }
        
        @keyframes typingAnimation {
            0%, 60%, 100% { transform: translateY(0); opacity: 0.6; }
            30% { transform: translateY(-8px); opacity: 1; }
        }
        
        .stSpinner > div {
            background-color: var(--primary) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar with enhanced animations
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; margin-bottom:30px; position: relative; z-index: 2;">
            <h1 style="color: var(--primary); margin-bottom:5px; font-size: 28px;">✨ HR Assistant Pro+</h1>
            <p style="color: var(--text); font-size: 15px; opacity: 0.8;">Your AI-Powered HR Companion</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="welcome-banner" style="position: relative; z-index: 2;">
            <h3 style="margin-bottom: 8px; position: relative; z-index: 3;">How can I help you today?</h3>
            <p style="margin: 0; position: relative; z-index: 3;">Ask me anything about your HR data!</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🌟 Popular Queries")
        st.markdown("""
        <div class="feature-card">
            <b>👥 Team Insights</b><br>
            "Show me the Marketing team"
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <b>📊 Hiring Analytics</b><br>
            "Hires by department last quarter"
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <b>💰 Compensation</b><br>
            "Salary distribution in Engineering"
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align:center; color:var(--text); font-size:14px; opacity: 0.7;">
            Powered by <b>LangChain</b> & <b>Streamlit</b>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <h1 style="color: var(--primary); margin-bottom: 5px;">HR Assistant Chat</h1>
        <p style="color: var(--text); opacity: 0.8; margin-top: 0;">Conversational AI for your HR data</p>
        """, unsafe_allow_html=True)
    
    # Initialize chat history with animated welcome
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": """
            <div style='animation: fadeIn 1s ease-in-out;'>
                <span style='font-size: 18px;'>👋 Welcome to your <b>HR Assistant Pro+</b>!</span><br><br>
                I can help you with:<br>
                • Employee records<br>
                • Department analytics<br>
                • Hiring trends<br>
                • Compensation data<br><br>
                Try asking:<br>
                <span style='color: var(--primary);'>• "Can you show engineers with more than 5 years of experience?"</span><br>
                <span style='color: var(--secondary);'>• "What is the average salary by each job level?"</span><br>
                <span style='color: var(--accent);'>• "Who are the recent hires in the Sales department?"</span><br>
                <span style='color: var(--info);'>• "What department does Priya Sharma work in?"</span><br>
                <span style='color: var(--info);'>• "Which projects has Priya Sharma worked on?"</span><br>
                <span style='color: var(--info);'>• "What are the recent leave details of Priya Sharma?"</span>

            </div>
            """}
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask your HR question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        # Get assistant response
    # Get assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Show typing animation
            with message_placeholder.container():
                st.markdown("""
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
                """, unsafe_allow_html=True)
            
            # Replace your try block in the chat handling with:
            try:
                import time

                start_time = time.time()
                agent_response = agent({"input": prompt})

                # Step 1: Extract raw result from agent
                if agent_response.get('intermediate_steps'):
                    raw_response = agent_response['intermediate_steps'][-1][-1]
                else:
                    raw_response = agent_response['output']

                # Step 2: Beautify using LLM
                if "No results" not in str(raw_response):
                    beautify_prompt = f"""You are a helpful assistant. Beautify the following SQL result and present it in a clean, readable way for the user
                    and make it more engaging and if table requirem generate table with row column:\n\n{raw_response}"""
                    print(f"💬 Beautifying response: {beautify_prompt}")

                    # Pass raw response to LLM
                    llm_response = llm.invoke(beautify_prompt)
                    pretty_response = llm_response.content

                    # Step 3: Wrap in HTML
                    final_response = f"""
                        <div style='margin-bottom: 15px; font-family: Arial, sans-serif;'>
                            {pretty_response}
        
                            
                    """
                else:
                    final_response = str(raw_response)

                # Step 4: Display in Streamlit
                message_placeholder.markdown(final_response, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": final_response})


                
            except Exception as e:
                error_msg = f"""
                <div style='color: var(--error); animation: fadeIn 0.5s ease-out;'>
                    ⚠️ Oops! I couldn't process that request.<br><br>
                    <i>Error: {str(e)[:200]}</i>
                </div>
                """
                message_placeholder.markdown(error_msg, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()