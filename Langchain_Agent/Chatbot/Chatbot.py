from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4", temperature=0.1, max_tokens=100)

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can answer questions and provide explanations."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

chat_history = []

# ✅ Load history from talk.txt (if exists) and convert to message objects
try:
    with open("talk.txt", "r") as file:
        lines = [line.strip() for line in file.readlines()]
        for i, line in enumerate(lines):
            if i % 2 == 0:
                chat_history.append(HumanMessage(content=line))
            else:
                chat_history.append(AIMessage(content=line))
    # print("Chat history loaded from talk.txt.",chat_history)
except FileNotFoundError:
    pass

# Chat loop
while True:
    query = input("Enter your query: ")

    if query.lower() == "exit":
        # ✅ Save structured chat to talk.txt
        with open("talk.txt", "a") as file:
            for msg in chat_history:
                file.write(f"{msg.content}\n")
        break

    # ✅ Append user query
    chat_history.append(HumanMessage(content=query))

    # Format prompt and get model response
    prompt = chat_template.format_prompt(chat_history=chat_history, query=query)
    response = model.invoke(prompt)

    # ✅ Append model response
    chat_history.append(AIMessage(content=response.content))

    print("Assistant:", response.content)
