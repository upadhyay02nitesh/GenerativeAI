import streamlit as st
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
import time

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Travel Weather Agent",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI (blue/teal theme)
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextInput input {
        border-radius: 20px !important;
        padding: 10px !important;
        border: 1px solid #0077b6 !important;
    }
    .stButton button {
        border-radius: 20px !important;
        background-color: #0096c7 !important;
        color: white !important;
        border: none !important;
        font-weight: 500 !important;
    }
    .weather-card {
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        background: white;
        box-shadow: 0 4px 12px 0 rgba(0,119,182,0.1);
        border-left: 5px solid #00b4d8;
    }
    .tool-pill {
        display: inline-block;
        padding: 5px 12px;
        margin-right: 8px;
        margin-bottom: 8px;
        border-radius: 20px;
        background-color: #caf0f8;
        color: #0077b6;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .output-card {
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        background: white;
        box-shadow: 0 4px 12px 0 rgba(0,119,182,0.1);
        border-left: 5px solid #0096c7;
    }
    .section-title {
        color: #0077b6;
        margin-bottom: 10px;
    }
    .spinner {
        color: #0096c7 !important;
    }
    .temperature-display {
        font-size: 2.5rem;
        font-weight: 700;
        color: #023e8a;
        margin: 10px 0;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    </style>
    """, unsafe_allow_html=True)

# App header with animation
st.markdown("""
    <div class="fade-in">
        <h1 style="color: #0077b6;">🌦️ Travel Weather Agent </h1>
        <p>Get comprehensive weather and travel information for any destination</p>
    </div>
""", unsafe_allow_html=True)

# Initialize tools with caching
@st.cache_resource
def load_tools():
    @tool
    def get_weather(location: str) -> dict:
        """Fetches current weather data for a specified location using WeatherAPI."""
        WEATHER_API_KEY = os.getenv("WEATHERAPI_KEY")
        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={location}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            weather_data = response.json()
            icon_url = f"https:{weather_data['current']['condition']['icon']}"
            return {
                "location": f"{weather_data['location']['name']}, {weather_data['location']['country']}",
                "temperature": f"{weather_data['current']['temp_c']}°C",
                "feels_like": f"{weather_data['current']['feelslike_c']}°C",
                "condition": weather_data['current']['condition']['text'],
                "wind": f"{weather_data['current']['wind_kph']} kph {weather_data['current']['wind_dir']}",
                "humidity": f"{weather_data['current']['humidity']}%",
                "icon": icon_url,
                "last_updated": weather_data['current']['last_updated']
            }
        except Exception as e:
            return {"error": str(e)}

    @tool
    def get_date() -> str:
        """Returns the current date and time."""
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    search_tool = DuckDuckGoSearchRun()
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    
    return {
        "Web Search": search_tool,
        "Weather API": get_weather,
        "Wikipedia": wikipedia,
        "Date/Time": get_date
    }

# Initialize LLM with caching
@st.cache_resource
def load_llm():
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7
    )

# Loading animation
def show_loading_animation():
    placeholder = st.empty()
    for i in range(3):
        placeholder.markdown(f"🌍 Gathering information{'.'*(i+1)}")
        time.sleep(0.5)
    placeholder.empty()

# Main app function
def main():
    # User input
    with st.form(key='travel_form'):
        location = st.text_input(
            "Where would you like to travel?",

            
            placeholder="Enter a city or destination",
            help="Example: Mawsynram, Meghalaya"
        )
        submit_button = st.form_submit_button(label="Get Travel Info")

    if submit_button and location:
        with st.spinner("🌍 Gathering travel information..."):
            try:
                # Show loading animation
                show_loading_animation()
                
                # Load tools and LLM
                tools = load_tools()
                llm = load_llm()
                react_prompt = hub.pull("hwchase17/react")
                
                # Compact tools display
                st.markdown("### 🛠️ Tools Being Used")
                tools_html = " ".join([f'<span class="tool-pill">{name}</span>' for name in tools.keys()])
                st.markdown(tools_html, unsafe_allow_html=True)
                st.markdown("---")
                
                # Create agent
                agent = create_react_agent(
                    llm=llm,
                    tools=list(tools.values()),
                    prompt=react_prompt
                )
                agent_executor = AgentExecutor(
                        agent=agent,
                        tools=list(tools.values()),
                        verbose=False,
                        handle_parsing_errors=True,
                        max_iterations=10,  # Increased from default 5
                        max_execution_time=30,  # 30 seconds max
                        early_stopping_method="generate"  # Better handling of long processes
                    )

                # Execute agent
                response = agent_executor.invoke({
                    "input": (
                        f"Provide detailed information about visiting {location}. "
                        f"Include: 1) Current weather conditions, 2) Top attractions from Wikipedia, "
                        f"3) Best time to visit based on climate, and 4) Any travel tips."
                    )
                })

                # Display results with animation
                st.success("✅ Here's your comprehensive travel guide!")
                
                # Weather card
                weather_data = tools["Weather API"](location)
                if "error" not in weather_data:
                    st.markdown("### 🌤️ Current Weather")
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        try:
                            st.image(weather_data["icon"], width=80)
                        except:
                            st.warning("Weather icon unavailable")
                    with col2:
                        st.markdown(f"""
                            <div class="weather-card fade-in">
                                <h3>{weather_data['location']}</h3>
                                <div class="temperature-display">{weather_data['temperature']}</div>
                                <p><b>Feels Like:</b> {weather_data['feels_like']}</p>
                                <p><b>Condition:</b> {weather_data['condition']}</p>
                                <p><b>Wind:</b> {weather_data['wind']}</p>
                                <p><b>Humidity:</b> {weather_data['humidity']}</p>
                                <p><small>Last updated: {weather_data['last_updated']}</small></p>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error(f"Weather data error: {weather_data['error']}")
                
                # Main output card
                st.markdown("### 📝 Travel Guide")
                output_content = response.get('output', 'No information available')
                if output_content:
                    try:
                        processed_text = str(output_content).replace('\n', '<br>')
                        st.markdown(
                            f"""<div class="output-card fade-in">{processed_text}</div>""",
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.error(f"Error displaying results: {str(e)}")
                else:
                    st.warning("No travel information was generated")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Technologies Used section
st.sidebar.markdown("""
    <style>
    .fade-in {
        animation: fadeIn 2s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .tech-box {
        background: linear-gradient(135deg, #1f1c2c, #928DAB);
        color: white;
        border-radius: 12px;
        padding: 15px;
        margin-top: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        font-family: 'Segoe UI', sans-serif;
    }
    .tech-box p {
        margin: 6px 0;
        font-size: 15px;
        line-height: 1.4;
    }
    .tech-box b {
        color: #FFD700;
    }
    .section-title {
        font-size: 20px;
        font-weight: bold;
        color: #00FFCC;
        margin-bottom: 10px;
        animation: glow 1.5s infinite alternate;
    }
    @keyframes glow {
        from { text-shadow: 0 0 5px #00FFCC; }
        to { text-shadow: 0 0 15px #00FFCC, 0 0 5px #00FFCC; }
    }
    </style>

    <div class="tech-box fade-in">
        <div class="section-title">🛠️ Technologies Used</div>
        <p><b>Core Framework:</b> Streamlit</p>
        <p><b>AI/ML:</b> LangChain, OpenAI GPT-3.5</p>
        <p><b>APIs:</b> WeatherAPI, Wikipedia API</p>
        <p><b>Search:</b> DuckDuckGo Search</p>
        <p><b>Styling:</b> Custom CSS</p>
        <p><b>Animations:</b> CSS Keyframes</p>
    </div>
""", unsafe_allow_html=True)


# Run the app
if __name__ == "__main__":
    main()