from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Any
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.types import interrupt, Command
import sqlite3
import requests
import math
import os

load_dotenv()

alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
)

# Embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

def ingest_rag_document(file_path):
    DB_PATH = "faiss_db"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(DB_PATH)


def get_retriever():
    DB_PATH = "faiss_db"
    vector_store = FAISS.load_local(
            folder_path=DB_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    return retriever

# rag tool

@tool
def rag_tool(query: str) -> str:
    """
    Retrieve relevant information from the PDF document.

    Use this tool when the user asks factual or conceptual questions
    that may be answered using the stored PDF documents.

    Args:
        query: The question or search query used to retrieve PDF content.
    """
    retriever = get_retriever()
    documents = retriever.invoke(query)

    if not documents:
        return "No relevant information was found in the PDF."

    formatted_documents = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")

        formatted_documents.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(formatted_documents)




search_tool = TavilySearch(
    max_results = 5,
    topic = "general",
    search_depth = "advanced"
)

@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """

    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={alpha_vantage_key}"
    r = requests.get(url)
    return r.json()


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """
    # This pauses the graph and returns control to the caller
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,
        }


@tool
def get_current_weather(location: str) -> str:
    """
    Get the current real-time weather for a given city or location.

    Args:
        location: City or location name, for example:
                  "Dhaka", "London, UK", or "New York, US".

    Returns:
        A formatted current weather report.
    """

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return (
            "Weather API key is missing. "
            "Set the OPENWEATHER_API_KEY environment variable."
        )

    try:
        # Step 1: Convert the location name into latitude and longitude
        geocoding_url = "https://api.openweathermap.org/geo/1.0/direct"

        geocoding_params = {
            "q": location,
            "limit": 1,
            "appid": api_key,
        }

        geo_response = requests.get(
            geocoding_url,
            params=geocoding_params,
            timeout=10,
        )
        geo_response.raise_for_status()

        locations: list[dict[str, Any]] = geo_response.json()

        if not locations:
            return f"Could not find the location: {location}"

        latitude = locations[0]["lat"]
        longitude = locations[0]["lon"]
        resolved_name = locations[0].get("name", location)
        country = locations[0].get("country", "")
        state = locations[0].get("state", "")

        # Step 2: Get current weather using latitude and longitude
        weather_url = "https://api.openweathermap.org/data/2.5/weather"

        weather_params = {
            "lat": latitude,
            "lon": longitude,
            "appid": api_key,
            "units": "metric",
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10,
        )
        weather_response.raise_for_status()

        weather_data = weather_response.json()

        temperature = weather_data["main"]["temp"]
        feels_like = weather_data["main"]["feels_like"]
        humidity = weather_data["main"]["humidity"]
        pressure = weather_data["main"]["pressure"]
        description = weather_data["weather"][0]["description"]
        wind_speed = weather_data.get("wind", {}).get("speed", "N/A")
        visibility_meters = weather_data.get("visibility")

        visibility_km = (
            round(visibility_meters / 1000, 1)
            if visibility_meters is not None
            else "N/A"
        )

        location_parts = [resolved_name]

        if state:
            location_parts.append(state)

        if country:
            location_parts.append(country)

        display_location = ", ".join(location_parts)

        return (
            f"Current weather in {display_location}:\n"
            f"- Condition: {description.title()}\n"
            f"- Temperature: {temperature}°C\n"
            f"- Feels like: {feels_like}°C\n"
            f"- Humidity: {humidity}%\n"
            f"- Pressure: {pressure} hPa\n"
            f"- Wind speed: {wind_speed} m/s\n"
            f"- Visibility: {visibility_km} km"
        )
    except requests.Timeout:
        return "The weather service request timed out. Please try again."

    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response else "unknown"

        if status_code == 401:
            return "The OpenWeather API key is invalid or inactive."

        return f"Weather API returned an HTTP error: {status_code}"

    except requests.RequestException as error:
        return f"Could not connect to the weather service: {error}"

    except (KeyError, TypeError, ValueError) as error:
        return f"Unexpected weather API response: {error}"

# Make tool list
tools = [search_tool,calculator, get_stock_price,get_current_weather, rag_tool, purchase_stock]

# Make the LLM tool-aware
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    """LLM node that can answer directly or call an appropriate tool."""

    system_message = SystemMessage(
        content="""
You are a helpful Agentic AI Chatbot with access to several tools.

Your primary goal is to answer the user's question accurately and helpfully while
using tools only when they are genuinely necessary.

TOOL USAGE RULES:

1. RAG / DOCUMENT TOOL
- Use `rag_tool` for questions about the uploaded PDF or document.
- Always retrieve relevant document content before answering a PDF/document-related question.
- Do not invent, assume, or hallucinate information from the uploaded document.
- Base document-specific answers on the retrieved content.
- If the user asks about a PDF/document but no document is available, ask the user to upload one.

2. WEB SEARCH TOOL
- Use `search_tool` for current events, recent information, changing information,
  or information that genuinely requires an internet search.
- Do NOT use `search_tool` for simple general-knowledge questions that you can
  answer reliably from your own knowledge.
- Do not use web search merely because a question is factual.

3. CALCULATOR TOOL
- Use `calculator` for mathematical calculations.
- Do not manually calculate complex expressions when the calculator is available.
- For simple conversational arithmetic, you may answer directly when appropriate.

4. STOCK PRICE TOOL
- Use `get_stock_price` when the user asks for the current/latest stock price
  or other real-time stock information supported by the tool.

5. PURCHASE STOCK TOOL
- Use `purchase_stock` when the user explicitly wants to purchase a stock.
- This is an external/financial action, so follow any human-approval requirements
  configured in the graph before executing the action.
- Never execute a purchase merely because the user is asking for information
  or recommendations about a stock.

6. WEATHER TOOL
- Use `get_current_weather` when the user asks for current or real-time weather
  information for a specific location.

GENERAL RULES:

- Answer general questions directly when no tool is required.
- Choose the most appropriate tool based on the user's intent.
- Do not call multiple tools when one tool is sufficient.
- Do not use a tool simply because it is available.
- When a tool is required, use it before giving the final answer.
- After receiving a tool result, use that result to produce a clear and helpful
  final answer.
- Never claim that you used a tool when you did not.
- Never fabricate tool results.
- If a tool fails or does not provide sufficient information, clearly explain
  the limitation instead of making up an answer.
"""
    )

    messages = [
        system_message,
        *state["messages"]
    ]

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}


# Nodes 2 - tool node
tool_node = ToolNode(tools)

# checkpointer
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint_saver = SqliteSaver(conn)

# ============================================================
# THREAD TITLE DATABASE
# ============================================================

def initialize_thread_titles_table():

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS thread_titles (
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL
        )
    """)

    conn.commit()


initialize_thread_titles_table()

def save_thread_title(thread_id, title):

    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO thread_titles
        (thread_id, title)
        VALUES (?, ?)
    """, (thread_id, title))

    conn.commit()


# graph
graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

#add edges
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')

#compile the graph
chatbot = graph.compile(checkpointer=checkpoint_saver)

all_threads = set()

def get_all_threads():
    for chkpt in checkpoint_saver.list(None):
        all_threads.add(chkpt.config['configurable']['thread_id'])
    return list(all_threads)


def get_thread_title(thread_id):

    cursor = conn.cursor()

    cursor.execute("""
        SELECT title
        FROM thread_titles
        WHERE thread_id = ?
    """, (thread_id,))

    result = cursor.fetchone()

    if result:
        return result[0]

    return None

def get_all_thread_titles():

    cursor = conn.cursor()

    cursor.execute("""
        SELECT thread_id, title
        FROM thread_titles
    """)

    rows = cursor.fetchall()

    return {
        thread_id: title
        for thread_id, title in rows
    }