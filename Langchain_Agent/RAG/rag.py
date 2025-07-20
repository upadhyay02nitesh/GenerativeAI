import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import FileChatMessageHistory
load_dotenv()

# Step 1: Load vector store
embeddings = OpenAIEmbeddings()
persist_dir = "./chroma_db"

if not os.path.exists(persist_dir):
    print("Creating vector DB...")
    docs = TextLoader("./data.txt", encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10).split_documents(docs)
    Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)

retriever = Chroma(persist_directory=persist_dir, embedding_function=embeddings).as_retriever()

# Step 2: Initialize LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Step 3: Memory — Just simple conversation buffer
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=FileChatMessageHistory("chat_history.txt")
)
# Step 4: Prompt with memory
prompt = PromptTemplate.from_template("""
You are a helpful assistant for Tata Consultancy Services (TCS).
Maintain basic context from recent conversation.

Chat History:
{chat_history}

Context from Documents:
{context}

Question: {question}

Answer:""")

# Chat loop
print("🤖 TCS Assistant with Buffer Memory")
print("Type 'quit' to exit")
print("----------------------------------")

while True:
    question = input("\nYou: ").strip()
    if question.lower() in ['quit', 'exit']:
        break

    # Get relevant document context
    docs = retriever.invoke(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # Get memory from buffer
   # Load existing chat history
    chat_history = memory.load_memory_variables({})["chat_history"]

    # Split into messages (assumes newlines between entries)
    history_lines = chat_history.strip().split("\n")

    # Keep only the last 10 lines
    last_10 = history_lines[-10:]

    # Convert back to string format
    chat_history = "\n".join(last_10)

    print(f"\nChat History (Last 10):\n{chat_history}")
    # Format prompt
    full_prompt = prompt.format(
        chat_history=chat_history,
        context=context,
        question=question
    )

    # Get response
    response = llm.invoke(full_prompt)

    # Update memory
    memory.save_context({"input": question}, {"output": response.content})

    # Print reply
    print(f"\nAssistant: {response.content}")
