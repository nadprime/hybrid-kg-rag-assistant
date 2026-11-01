---
title: Hybrid Knowledge Graph RAG Assistant
emoji: 🤖
colorFrom: red
colorTo: red
sdk: streamlit
sdk_version: 1.46.1
app_file: app.py
pinned: false
python_version: 3.11
tags:
- streamlit
- rag
- knowledge-graph
- crewai
- neo4j
- gemma-3
short_description: A hybrid AI assistant combining RAG with Knowledge Graph.
---


# Hybrid Knowledge Graph RAG Assistant 🤖

A sophisticated AI assistant that combines Retrieval-Augmented Generation (RAG) with Knowledge Graph technology to provide intelligent, context-aware responses. Built with CrewAI, Neo4j, and Streamlit, this system leverages both vector similarity search and graph-based reasoning for enhanced information retrieval.

#### Video Link: https://youtu.be/nccqkSy7UZg

> Note: The currently deployed demo may not work as expected, as the Neo4j instance might have been automatically deleted by the time you try to use it.

## 🌟 Features

- **Hybrid Architecture**: Combines traditional RAG with knowledge graph reasoning
- **Multi-Agent System**: Powered by CrewAI for specialized task handling
- **Vector Database**: ChromaDB for efficient similarity search
- **Knowledge Graph**: Neo4j integration for relationship-based queries
- **Interactive UI**: Clean Streamlit interface for easy interaction
- **Persistent Memory**: Entity memory, Long-term, and short-term memory for context retention
- **Specialized Agents**: Expert agents for data retrieval and conversational responses

## 🏗️ Architecture

The system consists of several key components:

### Core Components
1. **Streamlit Application** (`app.py`): User interface and application orchestration
2. **CrewAI Framework** (`src/hybrid_kg_rag_assistant/crew.py`): Multi-agent coordination
3. **Custom Tools** (`src/hybrid_kg_rag_assistant/tools/`): Neo4j integration and custom functionalities
4. **Vector Database** (`chroma_db/`): ChromaDB for document embeddings
5. **Knowledge Base** (`knowledge/`): MOSDAC scraped data and other sources

### Agent Architecture
- **Conversational Data Retrieval Expert**: Handles complex queries and data retrieval
- **Expert Conversational Response Specialist**: Provides human-like, contextual responses
- **Memory Systems**: 
  - Long-term memory for persistent context across sessions
  - Short-term memory for current conversation context
  - Entity memory for storing and retrieving information about specific entities

### How It Works

1. **Query Processing**: User queries are processed through the Streamlit interface
2. **Multi-Agent Coordination**: CrewAI orchestrates specialized agents for different tasks
3. **Hybrid Retrieval**: 
   - Vector similarity search in ChromaDB for semantic matching
   - Knowledge graph queries in Neo4j for relationship-based information
4. **Response Generation**: Agents collaborate to generate comprehensive, accurate responses
5. **Memory Integration**: Conversation context is stored for future reference

## 🚀 Local Setup

### Prerequisites

- Python 3.11 or higher
- Neo4j database (local or remote)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd hybrid_kg_rag_assistant
   ```

2. **Install dependencies using uv (recommended)**
   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Navigate to the project directory
   cd hybrid_kg_rag_assistant
   
   # Install project dependencies
   uv sync
   ```

   **Alternative: Using pip**
   ```bash
   # Navigate to the project directory
   cd hybrid_kg_rag_assistant
   
   # Create a virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Set up Neo4j**
   - Install Neo4j locally:
     ```bash
     # Download and install Neo4j Community Edition
     # Or use Neo4j Desktop for easier management
     ```
   - Alternative: Use Neo4j AuraDB (cloud-based)
   - Start Neo4j service and note the connection details (URI, username, password)

4. **Configure environment variables**
   Create a `.env` file in the project root directory:
   ```bash
   # Navigate to project root if not already there
   cd hybrid_kg_rag_assistant
   
   # Create .env file
   touch .env
   ```
   
   Add the following content to `.env`:
   ```env
   NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password
   GOOGLE_API_KEY=your_google_api_key_here
   MODEL=gemini/gemma-3-27b-it
   # Add other necessary API keys for your LLM provider
   ```

5. **Initialize the database and load data**
   ```bash
   # Ensure you're in the project directory
   cd hybrid_kg_rag_assistant
   
   # Run the initial setup (if needed)
   uv run python -m src.hybrid_kg_rag_assistant.main
   ```

### Running the Application

#### Option 1: Using Streamlit (Web Interface)

1. **Start the Streamlit app**
   ```bash
   # Ensure you're in the project directory
   cd hybrid_kg_rag_assistant
   
   # Run the Streamlit application
   uv run streamlit run app.py
   ```

   **Alternative with pip:**
   ```bash
   # If using pip installation
   streamlit run app.py
   ```

   **Or use the VS Code task:**
   ```bash
   # If using VS Code, you can run the predefined task
   # This will run: uv run streamlit run app.py
   ```

2. **Access the application**
   Open your browser and navigate to `http://localhost:8501`

#### Option 2: Using CrewAI CLI (Terminal Interface)

1. **Run with CrewAI CLI**
   ```bash
   # Ensure you're in the project directory
   cd hybrid_kg_rag_assistant
   
   # Run the CrewAI application directly
   crewai run
   ```

   **Or using uv:**
   ```bash
   # Using uv to run CrewAI
   uv run crewai run
   ```

2. **Interactive terminal session**
   - The application will start in terminal mode
   - Enter your questions directly in the command line
   - The multi-agent system will process your queries
   - Receive responses directly in the terminal

#### Starting Your Query Session

**For Web Interface:**
- Enter your questions in the chat interface
- The system will process your query using both RAG and knowledge graph approaches
- Receive comprehensive, context-aware responses

**For Terminal Interface:**
- Type your questions when prompted
- Press Enter to submit
- Wait for the multi-agent system to process and respond

## 🛠️ Usage

### Basic Usage
1. Launch the application following the setup instructions
2. Enter your question in the chat interface
3. Wait for the multi-agent system to process your query
4. Review the response, which combines information from multiple sources

### Advanced Features
- **Entity Memory**: The system maintains detailed information about entities encountered in conversations
- **Conversation Memory**: The system remembers previous conversations for context
- **Complex Queries**: Ask multi-part questions that require reasoning across different data sources
- **Knowledge Graph Exploration**: Queries can traverse relationships in the knowledge graph
- **Vector Similarity**: Find semantically similar information even with different wording
- **Dual Interface**: Choose between web-based (Streamlit) or terminal-based (CrewAI CLI) interaction

## 📊 Data Sources

The system currently includes:
- **MOSDAC Data**: Scraped meteorological and satellite data (`knowledge/mosdac_scraped_data.csv`)
- **Vector Embeddings**: Pre-computed embeddings stored in ChromaDB
- **Knowledge Graph**: Structured relationships in Neo4j

## 🔧 Configuration

### Agent Configuration
Edit `src/hybrid_kg_rag_assistant/config/agents.yaml` to customize:
- Agent roles and responsibilities
- System prompts and behavior

### Task Configuration
Modify `src/hybrid_kg_rag_assistant/config/tasks.yaml` to adjust:
- Task definitions and workflows
- Agent assignments
- Expected outputs

### Custom Tools
Extend functionality by adding new tools in `src/hybrid_kg_rag_assistant/tools/`:
- Follow the existing pattern in `custom_tool.py`
- Integrate with Neo4j or other external services

## 🐳 Docker Deployment

A Dockerfile is provided for containerized deployment:

```bash
# Build the Docker image
docker build -t hybrid-kg-rag-assistant .

# Run the container
docker run -p 8501:8501 hybrid-kg-rag-assistant
```

## 📁 Project Structure

```
hybrid_kg_rag_assistant/
├── app.py                          # Main Streamlit application
├── Dockerfile                      # Docker configuration
├── pyproject.toml                  # Project dependencies and configuration
├── requirements.txt                # Python dependencies
├── knowledge/                      # Data sources
│   └── mosdac_scraped_data.csv    # Primary dataset
├── src/hybrid_kg_rag_assistant/   # Core application code
│   ├── crew.py                    # CrewAI agent coordination
│   ├── main.py                    # Main application logic
│   ├── config/                    # Configuration files
│   │   ├── agents.yaml           # Agent definitions
│   │   └── tasks.yaml            # Task configurations
│   ├── tools/                     # Custom tools and integrations
│   │   └── custom_tool.py        # Neo4j integration
│   └── crewai_storage/           # Memory storage
│       ├── entities/             # Entity memory storage
│       ├── short_term/           # Short-term memory
│       └── long_term_memory_storage.db  # Long-term memory
├── chroma_db/                     # Vector database storage
└── tests/                         # Test files
```

## 🎯 Key Benefits

- **Enhanced Accuracy**: Combines multiple retrieval methods for better results
- **Context Awareness**: Maintains conversation history and context
- **Scalability**: Agent-based architecture allows for easy extension
- **Flexibility**: Configurable agents and tasks for different use cases
- **Performance**: Efficient vector search combined with graph traversal
- **User-Friendly**: Clean, intuitive Streamlit interface

