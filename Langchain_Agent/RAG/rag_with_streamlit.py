import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import FileChatMessageHistory

# Load environment variables
load_dotenv()

# Initialize the vector store
@st.cache_resource
def initialize_vectorstore():
    embeddings = OpenAIEmbeddings()
    persist_dir = "./chroma_db"
    
    if not os.path.exists(persist_dir):
        st.info("Creating vector DB...")
        docs = TextLoader("./data.txt", encoding="utf-8").load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10).split_documents(docs)
        Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings).as_retriever()

retriever = initialize_vectorstore()

# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

llm = get_llm()

# Initialize memory
@st.cache_resource
def get_memory():
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=FileChatMessageHistory("chat_history.txt")
    )

memory = get_memory()

# Prompt template
prompt = PromptTemplate.from_template("""
You are a helpful assistant for  Company.
Maintain basic context from recent conversation.

Chat History:
{chat_history}

Context from Documents:
{context}

Question: {question}

Answer:""")

# Streamlit UI with enlarged query message section
st.markdown("""
<style>
    /* Main chat container */
    .stChatFloatingInputContainer {
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        padding: 15px;  /* Added padding around the input container */
    }
    
    /* Individual message bubbles */
    .stChatMessage {
        padding: 12px 16px;
        border-radius: 18px;
        margin-bottom: 16px;
        max-width: 80%;
        line-height: 1.5;
        font-size: 15px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    /* User message styling */
    [data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"] {
        background-color: #4f46e5 !important;
    }
    
    .user-message {
        background-color: #4f46e5;
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    
    /* Assistant message styling */
    [data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
        background-color: #10b981 !important;
    }
    
    .assistant-message {
        background-color: #f8fafc;
        color: #1e293b;
        margin-right: auto;
        border-bottom-left-radius: 4px;
        border: 1px solid #e2e8f0;
    }
    
    /* ENLARGED Input box styling */
    .stTextInput>div>div>input {
        border-radius: 12px !important;
        padding: 18px 20px !important;  /* Increased padding */
        border: 1px solid #e2e8f0 !important;
        font-size: 16px !important;  /* Larger font size */
        min-height: 60px !important;  /* Minimum height */
    }
    
    /* Make the input container wider */
    .stChatFloatingInputContainer>div {
        width: 100% !important;
        max-width: 800px !important;  /* Wider input area */
        margin: 0 auto !important;
    }
    
    /* Header styling */
    .stApp header {
        background-color: #4f46e5;
        color: white;
    }
    
    /* Chat header */
    .stApp h1 {
        color: #4f46e5;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* Timestamp styling */
    .stChatMessage .timestamp {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 4px;
        text-align: right;
    }
    
    /* Animation for new messages */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stChatMessage {
        animation: fadeIn 0.3s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# Add some decorative elements to your header
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 style="color: #4f46e5; font-weight: 700; margin-bottom: 0.5rem;">🤖 Company Knowledge Assistant</h1>
    <p style="color: #64748b; margin-bottom: 1.5rem;">Ask me anything about Consultancy Services</p>
    <div style="height: 4px; background: linear-gradient(90deg, #4f46e5, #10b981); border-radius: 2px; margin: 0 auto 1.5rem; width: 100px;"></div>
</div>
""", unsafe_allow_html=True)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_input := st.chat_input("Ask about Company..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get relevant document context
    docs = retriever.invoke(user_input, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Get memory from buffer - properly handle list of messages
    memory_vars = memory.load_memory_variables({})
    chat_history = memory_vars["chat_history"]
    
    # Convert messages to string format
    chat_history_str = "\n".join([
        f"{'User' if i % 2 == 0 else 'Assistant'}: {msg.content}" 
        for i, msg in enumerate(chat_history[-10:])  # Keep last 5 exchanges (10 messages)
    ])

    # Format prompt
    full_prompt = prompt.format(
        chat_history=chat_history_str,
        context=context,
        question=user_input
    )
    
    # Get response
    response = llm.invoke(full_prompt)
    
    # Update memory
    memory.save_context({"input": user_input}, {"output": response.content})
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response.content})
    with st.chat_message("assistant"):
        st.markdown(response.content)

# Add About section with technology cards
# Technologies section using st.write() with proper styling
st.write("""
<style>
    .tech-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .tech-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: transform 0.3s ease;
        border-left: 4px solid;
        margin-bottom: 1rem;
    }
    .tech-card:hover {
        transform: translateY(-5px);
    }
    .tech-card h3 {
        margin-top: 0;
    }
    .tech-card p {
        color: #64748b;
        margin-bottom: 0;
    }
    @keyframes cardEntrance {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .tech-card {
        animation: cardEntrance 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# Section header
st.markdown("## Technologies Powering This Assistant")

# Technology cards using st.write()
with st.container():
    cols = st.columns(3)
    
    with cols[0]:
        st.write("""
        <div class="tech-card" style="border-left-color: #4f46e5;">
            <h3 style="color: #4f46e5;">LangChain</h3>
            <p>Framework for building AI applications with LLMs</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("""
        <div class="tech-card" style="border-left-color: #10b981;">
            <h3 style="color: #10b981;">Streamlit</h3>
            <p>Web framework for creating interactive apps</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.write("""
        <div class="tech-card" style="border-left-color: #6366f1;">
            <h3 style="color: #6366f1;">ChromaDB</h3>
            <p>Vector database for storing embeddings</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("""
        <div class="tech-card" style="border-left-color: #f59e0b;">
            <h3 style="color: #f59e0b;">Memory</h3>
            <p>Persistent conversation history</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.write("""
        <div class="tech-card" style="border-left-color: #ec4899;">
            <h3 style="color: #ec4899;">RAG</h3>
            <p>Retrieval-Augmented Generation</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("""
        <div class="tech-card" style="border-left-color: #6b7280;">
            <h3 style="color: #6b7280;">OpenAI</h3>
            <p>GPT-3.5-turbo LLM</p>
        </div>
        """, unsafe_allow_html=True)