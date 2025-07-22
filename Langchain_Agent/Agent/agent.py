from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub
from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools.wikipedia.tool import WikipediaQueryRun

import os
import requests
from datetime import datetime

# Load environment variables
load_dotenv()
WEATHER_API_KEY = os.getenv("WEATHERAPI_KEY")

# Define tools
@tool
def get_weather(location: str) -> str:
    """Fetches current weather data for a specified location using WeatherAPI."""
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={location}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        weather_data = response.json()
        return (
            f"Current Weather in {weather_data['location']['name']}, {weather_data['location']['country']}:\n"
            f"- Temperature: {weather_data['current']['temp_c']}°C\n"
            f"- Condition: {weather_data['current']['condition']['text']}\n"
            f"- Wind: {weather_data['current']['wind_kph']} kph\n"
            f"- Humidity: {weather_data['current']['humidity']}%"
        )
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

@tool
def get_date() -> str:
    """Returns the current date and time."""
    return f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# Tools
search_tool = DuckDuckGoSearchRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

tools = [search_tool, get_weather, wikipedia, get_date]

# LLM and prompt
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=1000)
react_prompt = hub.pull("hwchase17/react")

# Create agent
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

# Agent executor with error handling enabled
agent_executer = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,  # ✅ Add this to prevent crashing on invalid outputs
)

# Run agent
response = agent_executer.invoke({
    "input": (
        "I want to visit Mawsynram, Meghalaya. What's the current weather there? "
        "Also find information about popular tourist spots from Wikipedia "
        "and tell me what date I should plan my visit for optimal weather."
    )
})

# Output result
print(response['output'])
