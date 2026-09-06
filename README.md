# 🤖 Agent-X — Multi-Tool Agentic AI Assistant

> **An intelligent, persistent, tool-augmented AI assistant built with LangGraph, Gemini, RAG, Human-in-the-Loop workflows, and SQLite persistence.**

Agent-X is a **stateful Agentic AI assistant** designed to go beyond traditional question-answering chatbots.

Instead of simply generating text, Agent-X can **reason about a user's intent, select appropriate tools, retrieve information from uploaded documents, perform calculations, access real-time information, execute controlled actions with human approval, and maintain persistent conversation state across sessions.**

---

## ✨ Why Agent-X?

Traditional chatbots generally follow:

```text
User → LLM → Response
```

Agent-X follows a more capable architecture:

```text
                         ┌──────────────┐
                         │     User     │
                         └──────┬───────┘
                                │
                                ▼
                     ┌────────────────────┐
                     │   Gemini LLM      │
                     │  Agent Controller  │
                     └─────────┬──────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
             RAG Tool      Web Search      Calculator
                │              │              │
                ▼              ▼              ▼
             PDF Data      Tavily API     Math Engine
                │              │              │
                └──────────────┼──────────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
             Weather         Stocks        Actions
                │              │              │
                └──────────────┼──────────────┘
                               │
                               ▼
                         ┌───────────┐
                         │ LangGraph │
                         │ Stateflow │
                         └─────┬─────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
          SQLite Persistence            Human Approval
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                         Final Response
```

The system dynamically decides **when to answer directly and when to invoke a specialized tool**.

---

# 🚀 Features

### 🧠 Agentic Tool Calling

Agent-X can select tools according to the user's intent rather than relying on a fixed sequence.

Currently supported tools include:

* 🔎 Web Search
* 🧮 Calculator
* 📈 Stock Price Lookup
* 🌤️ Current Weather
* 📄 PDF RAG
* 💳 Simulated Stock Purchase with Human Approval

The agent is instructed to use tools only when they are genuinely necessary.

---

### 📚 Retrieval-Augmented Generation (RAG)

Upload a PDF directly through the chat interface and ask questions about its contents.

The RAG pipeline follows:

```text
PDF Upload
    ↓
PyPDFLoader
    ↓
Text Extraction
    ↓
Recursive Character Splitting
    ↓
Embeddings
    ↓
FAISS Vector Store
    ↓
Similarity Retrieval
    ↓
Relevant Context
    ↓
Gemini
    ↓
Grounded Answer
```

Current configuration:

* `PyPDFLoader`
* `RecursiveCharacterTextSplitter`
* `GoogleGenerativeAIEmbeddings`
* `FAISS`
* Similarity retrieval
* Top-K document retrieval

---

### 🧑‍⚖️ Human-in-the-Loop (HITL)

Agent-X does not blindly execute sensitive actions.

For example:

```text
User:
"Buy 10 shares of AAPL"

        ↓

Gemini
        ↓
purchase_stock()
        ↓
interrupt()
        ↓
┌────────────────────────────┐
│ Human approval required    │
│                            │
│ Buy 10 shares of AAPL?     │
│                            │
│ [Approve]     [Reject]     │
└────────────────────────────┘
        ↓
   Human decision
        ↓
   Command(resume=...)
        ↓
      Tool
        ↓
 Final AI response
```

This demonstrates an important Agentic AI principle:

> **AI can recommend or initiate an action, while humans retain control over consequential operations.**

---

### 💾 Persistent Conversations

Agent-X uses **LangGraph persistence with SQLite**.

Conversations are associated with unique thread IDs:

```text
Thread ID
    ↓
LangGraph Checkpointer
    ↓
Conversation State
    ↓
Messages / Tool State / HITL State
```

Closing and reopening the application does not automatically erase persisted LangGraph state.

---

### 🏷️ Persistent Conversation Titles

Each conversation can have a generated title based on its first user message.

For example:

```text
Chat History

🧠 Black Hole Singularity
📄 Research Paper Analysis
🌦️ Delhi Weather
🐍 Python Recursion
🔗 LangGraph Persistence
```

Thread titles are stored separately in the same SQLite database rather than modifying LangGraph's internal checkpoint schema.

---

### ⚡ Streaming Responses

Agent-X streams assistant responses instead of waiting for the entire response to complete.

The UI also provides tool activity feedback such as:

```text
🔧 Using search_tool...
✅ search_tool completed
```

This makes the agent's execution easier to understand and provides a more interactive experience.

---

### 📄 In-Chat PDF Upload

Users can upload PDFs directly from the chat input.

```text
┌────────────────────────────────────────┐
│ Type your question...          📎      │
└────────────────────────────────────────┘
```

The uploaded document is processed and indexed into the local FAISS vector store.

---

### 🔍 Intelligent Tool Selection

Agent-X does not use web search for every question.

The agent follows intent-based tool selection:

| User Intent              | Tool                 |
| ------------------------ | -------------------- |
| General knowledge        | LLM                  |
| Current information      | Web Search           |
| Uploaded PDF question    | RAG                  |
| Mathematical calculation | Calculator           |
| Current stock price      | Stock Tool           |
| Current weather          | Weather Tool         |
| Stock purchase           | Purchase Tool + HITL |

This helps demonstrate the difference between a **tool-enabled LLM** and an **agentic workflow**.

---

# 🛠️ Technology Stack

| Technology          | Purpose                                  |
| ------------------- | ---------------------------------------- |
| **Python**          | Core programming language                |
| **LangGraph**       | Agent orchestration and state management |
| **LangChain**       | LLM/tool integration                     |
| **Google Gemini**   | LLM and embeddings                       |
| **Streamlit**       | Interactive web interface                |
| **FAISS**           | Vector similarity search                 |
| **SQLite**          | Persistent state and thread metadata     |
| **Tavily**          | Web search                               |
| **OpenWeather API** | Weather information                      |
| **Alpha Vantage**   | Stock market data                        |
| **LangSmith**       | Observability and tracing                |
| **PyPDF**           | PDF processing                           |

---

# 🏗️ Architecture

```text
                         ┌─────────────────────┐
                         │      Streamlit       │
                         │     Chat Interface   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     LangGraph       │
                         │    State Machine    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Gemini LLM      │
                         │   Agent Controller  │
                         └──────────┬──────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 │                  │                  │
                 ▼                  ▼                  ▼
             RAG Tool          Search Tool        Calculator
                 │                  │                  │
                 ▼                  ▼                  ▼
              FAISS              Tavily            Math
                 │
                 │
       ┌─────────┴─────────┐
       │                   │
       ▼                   ▼
    Weather             Stocks
       │                   │
       │                   ▼
       │              Purchase Tool
       │                   │
       │                   ▼
       │              HITL Interrupt
       │                   │
       └─────────┬─────────┘
                 ▼
          Final AI Response

                 │
                 ▼
        ┌──────────────────┐
        │ SQLite Persistence│
        ├──────────────────┤
        │ LangGraph State  │
        │ Thread Titles    │
        └──────────────────┘
```

---

# 🔄 Agent Execution Flow

A typical request follows:

```text
1. User sends message
          ↓
2. LangGraph receives state
          ↓
3. Gemini analyzes intent
          ↓
4. Decide:
      ├── Answer directly
      └── Call a tool
          ↓
5. Tool executes
          ↓
6. Tool result returned to agent
          ↓
7. Gemini processes result
          ↓
8. Final response streamed to user
          ↓
9. State persisted in SQLite
```

For actions requiring approval:

```text
User Request
     ↓
LLM Tool Call
     ↓
HITL Interrupt
     ↓
Checkpoint Saved
     ↓
Human Approval
     ↓
Command(resume=decision)
     ↓
Tool Continues
     ↓
Final Response
```

---

# 📁 Project Structure

```text
Agent-X/
│
├── app.py
│
├── chatbot_tools_backend_final.py
│
├── requirements.txt
│
├── .gitignore
│
├── .env.example
│
├── README.md
│
├── chatbot.db              # Local only — ignored by Git
│
└── faiss_db/               # Local only — ignored by Git
```

> The exact filenames may vary depending on the current development version.

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/shashank-alt/Agent-X.git
```

```bash
cd Agent-X
```

---

## 2. Create a virtual environment

### Windows

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file in the project root.

```env
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=agent-x
```

### ⚠️ Security

**Never commit `.env` to GitHub.**

The repository's `.gitignore` excludes:

```text
.env
.venv/
chatbot.db
faiss_db/
```

Use `.env.example` to document required environment variables without exposing secrets.

---

# ▶️ Running the Application

Start Streamlit:

```bash
streamlit run app.py
```

Then open the local Streamlit URL displayed in the terminal.

---

# 💡 Example Queries

### General Question

```text
Explain Murphy's Law.
```

The agent can answer directly without unnecessary tool usage.

---

### Web Search

```text
What are the latest developments in AI agents?
```

The agent can invoke the web search tool.

---

### Calculator

```text
Calculate the compound interest on ₹100000 at 8% for 5 years.
```

The calculator tool can be selected.

---

### Weather

```text
What is the current weather in Delhi?
```

The weather tool retrieves current weather information.

---

### Stock Price

```text
What is the current price of AAPL?
```

The stock tool retrieves market information.

---

### PDF RAG

Upload a PDF and ask:

```text
What are the main conclusions of this document?
```

The RAG pipeline retrieves relevant chunks before generating the response.

---

### Human-in-the-Loop

```text
Buy 10 shares of AAPL.
```

Agent-X pauses and requests human approval before the simulated purchase proceeds.

---

# 🧠 Design Principles

Agent-X was designed around several important Agentic AI concepts.

### 1. Tool Selection

The LLM determines whether a specialized tool is necessary.

### 2. Stateful Execution

LangGraph maintains conversation state and workflow execution.

### 3. Persistence

SQLite allows state to survive application restarts.

### 4. Grounded Generation

RAG allows the model to answer questions using retrieved document context.

### 5. Human Oversight

HITL prevents autonomous execution of sensitive actions.

### 6. Observability

LangSmith can be used to inspect agent executions, tool calls, and traces.

### 7. Modular Architecture

Tools are implemented independently and exposed to the agent through LangGraph's tool-calling mechanism.

---

# 🔬 What This Project Demonstrates

This project demonstrates practical implementation of:

* Agentic AI
* LLM tool calling
* LangGraph state machines
* Conditional graph routing
* ToolNode
* RAG
* Vector databases
* Embeddings
* PDF ingestion
* Human-in-the-Loop workflows
* Interrupt/resume execution
* Persistent conversations
* SQLite checkpointing
* Streaming LLM responses
* API integration
* Environment variable management
* Agent observability
* Streamlit application development

---

# 🚧 Future Improvements

The project is designed to be extended.

### Planned / Possible Improvements

* [ ] MySQL/PostgreSQL application database
* [ ] User authentication
* [ ] User-specific conversation history
* [ ] Better conversation-title management
* [ ] Long-term semantic memory
* [ ] More sophisticated agent routing
* [ ] Tool error recovery
* [ ] Retry mechanisms
* [ ] Structured tool outputs
* [ ] Better RAG document management
* [ ] Multiple document collections
* [ ] Document-level citations
* [ ] RAG evaluation
* [ ] Automated agent evaluation
* [ ] Improved HITL workflows
* [ ] Docker containerization
* [ ] Cloud deployment
* [ ] Production-grade logging
* [ ] Automated testing
* [ ] Rate-limit handling
* [ ] PostgreSQL-based production persistence

---

# 🔒 Security Considerations

Agent-X is designed as an educational/portfolio project and should not be treated as a production financial trading system.

Important security practices include:

* Never expose API keys in source code.
* Store secrets in environment variables.
* Do not commit `.env`.
* Validate external API responses.
* Use human approval for consequential actions.
* Restrict calculator execution.
* Add authentication before multi-user deployment.
* Add proper authorization before exposing sensitive tools.
* Use a production database for multi-user deployments.

---

# 📊 Project Goals

The primary goal of Agent-X is to demonstrate how modern AI applications can move beyond simple prompt-response systems toward **stateful, tool-using, retrieval-augmented and human-supervised AI agents**.

The project focuses on combining these concepts into one practical application rather than implementing each feature independently.

---

# 👨‍💻 Author

**Shashank Jha**

B.Tech — Information Technology

Interested in:

* 🤖 Agentic AI
* 🧠 Machine Learning
* 💻 Software Development
* 🔍 AI Engineering
* ☁️ Cloud & DevOps

---

# ⭐ If You Find This Project Interesting

If Agent-X helps you understand Agentic AI, consider giving the repository a ⭐ on GitHub.

---

## 📜 License

This project is intended for educational and portfolio purposes.

Add an appropriate open-source license before distributing the project publicly.
