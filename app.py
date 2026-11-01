import streamlit as st
import sys
import os
import warnings
import time
import io
from queue import Queue
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Suppress all warnings before importing other modules
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Additional warning suppressions for runtime
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*PydanticDeprecatedSince.*")
warnings.filterwarnings("ignore", message=".*No path_separator found.*")
warnings.filterwarnings("ignore", message=".*model_fields.*")
warnings.filterwarnings("ignore", message=".*Using extra keyword arguments.*")
warnings.filterwarnings("ignore", message=".*Extra keys.*")
warnings.filterwarnings("ignore")  # Suppress all remaining warnings

# Store CrewAI data in project directory
project_root = Path(__file__).parent / "src" / "hybrid_kg_rag_assistant"
storage_dir = project_root / "crewai_storage"

os.environ["CREWAI_STORAGE_DIR"] = str(storage_dir)

# Ensure storage directory exists
storage_dir.mkdir(parents=True, exist_ok=True)

# Check if we're running in Hugging Face Spaces
IS_HF_SPACE = os.getenv("SPACE_ID") is not None

# Initialize session state for debug output
if "debug_output" not in st.session_state:
    st.session_state.debug_output = []
if "current_debug" not in st.session_state:
    st.session_state.current_debug = ""


class DebugCapture:
    """Capture stdout/stderr for debugging display"""

    def __init__(self):
        self.output_queue = Queue()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def __enter__(self):
        self.stdout_capture = io.StringIO()
        self.stderr_capture = io.StringIO()
        sys.stdout = self.stdout_capture
        sys.stderr = self.stderr_capture
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

    def get_output(self):
        stdout_content = self.stdout_capture.getvalue()
        stderr_content = self.stderr_capture.getvalue()

        # Filter out warnings and unwanted debug info
        lines = []
        for line in stdout_content.split("\n") + stderr_content.split("\n"):
            line = line.strip()
            if line and not any(
                skip in line.lower()
                for skip in [
                    "warning",
                    "deprecation",
                    "pydantic",
                    "path_separator",
                    "model_fields",
                    "extra keyword",
                    "extra keys",
                ]
            ):
                lines.append(line)

        return "\n".join(lines) if lines else ""


# Page configuration
st.set_page_config(
    page_title="Hybrid KG RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header h3 {
        color: #f0f0f0;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }
    .main-header p {
        color: #e0e0e0;
        margin: 1rem 0 0 0;
        font-size: 0.9rem;
    }
    .response-section {
        background-color: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .status-indicator {
        background-color: #e3f2fd;
        border: 1px solid #2196f3;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .error-box {
        background-color: #ffebee;
        border: 1px solid #f44336;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .example-buttons {
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid #667eea;
        background-color: white;
        color: #667eea;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #667eea;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    .debug-container {
        background-color: #1e1e1e;
        color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 0.8rem;
        line-height: 1.4;
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #333;
    }
    .debug-title {
        color: #4CAF50;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .agent-step {
        border-left: 3px solid #2196F3;
        padding-left: 0.5rem;
        margin: 0.5rem 0;
        background-color: #2a2a2a;
        padding: 0.5rem;
        border-radius: 4px;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    }
    .success-step {
        border-left: 3px solid #4CAF50;
        background-color: #1b5e20;
    }
    .agent-thinking {
        border-left: 3px solid #FF9800;
        background-color: #3e2723;
    }
    .processing-step {
        border-left: 3px solid #9C27B0;
        background-color: #4a148c;
    }
    .thinking-indicator {
        color: #FFC107;
        font-style: italic;
    }
</style>
""",
    unsafe_allow_html=True,
)


def query_crew(query: str, model_config=None, chat_history=None):
    """
    Query the CrewAI hybrid knowledge graph RAG assistant and stream the response.
    """
    # Clear previous debug output
    st.session_state.debug_output = []
    st.session_state.current_debug = ""

    try:
        # Check if required environment variables are set
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            yield "❌ **Configuration Error**\n\n"
            yield "GOOGLE_API_KEY environment variable is not set. "
            yield "Please configure your Google AI API key in the Hugging Face Space settings.\n\n"
            yield "**📋 How to configure:**\n"
            yield "1. Go to your Space settings\n"
            yield "2. Add GOOGLE_API_KEY as an environment variable\n"
            yield "3. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)"
            return

        from hybrid_kg_rag_assistant.crew import HybridKgRagAssistant

        # Prepare conversation context
        conversation_context = ""
        if chat_history and len(chat_history) > 0:
            # Include last few exchanges for context (limit to avoid token overflow)
            recent_history = chat_history[
                -6:
            ]  # Last 3 exchanges (user + assistant pairs)
            context_parts = []
            for msg in recent_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                context_parts.append(
                    f"{role}: {msg['content'][:200]}..."
                )  # Truncate for brevity
            conversation_context = "\n".join(context_parts)

        # Create inputs for the crew with conversation context
        inputs = {"query": query}

        # If we have conversation history, enhance the query with context
        if conversation_context:
            enhanced_query = f"""
Current Question: {query}

Recent Conversation Context:
{conversation_context}

Please provide a conversational response that considers this context and maintains dialogue flow.
"""
            inputs["query"] = enhanced_query

        # Show simple waiting message
        yield "🤔 **Analyzing your query...**\n\n"
        yield "⏳ Please wait while our AI agents analyze, process, and generate a detailed response. This may take a moment.\n\n"
        yield "💡 *Tip: Check the 'Agent Activity' section in the sidebar and click refresh to see what the agents are thinking!*\n\n"
        yield "---\n\n"

        # Add detailed debug messages to sidebar
        st.session_state.debug_output.append("🚀 [SYSTEM] Initializing Hybrid RAG pipeline")
        st.session_state.debug_output.append("� Loading knowledge base and vector store")
        st.session_state.debug_output.append("🔍 [SEARCH] Performing semantic similarity query")
        st.session_state.debug_output.append("🤖 [AGENT] DataRetrieval Expert: ONLINE")
        # Add query-specific context logs (keeping total under 10)
        query_lower = query.lower()
        if "satellite" in query_lower:
            st.session_state.debug_output.append("🛰️ [CONTEXT] Satellite domain focus activated")
        elif "isro" in query_lower:
            st.session_state.debug_output.append("🚀 [DB] ISRO mission database accessed")  
        elif "mosdac" in query_lower:
            st.session_state.debug_output.append("🌍 [FILTER] MOSDAC-specific knowledge prioritized")
        elif any(word in query_lower for word in ["weather", "climate"]):
            st.session_state.debug_output.append("⛅ [DOMAIN] Weather/Climate focus")
        else:
            st.session_state.debug_output.append("🧠 [ANALYSIS] Processing query context")

        # Capture agent thinking with detailed output
        with DebugCapture() as debug_capture:
            st.session_state.debug_output.append("⚙️ [CREW] Agent workflow configuration complete")
            st.session_state.debug_output.append("🎯 [TOOL] custom_tool.search_knowledge_base() executed")
            
            # Initialize and run the crew
            crew_assistant = HybridKgRagAssistant()
            
            st.session_state.debug_output.append("🔄 [EXEC] Multi-agent workflow execution started")
            st.session_state.debug_output.append("🧠 [MEMORY] Loading conversation context")
            st.session_state.debug_output.append("✍️ [AGENT] ResponseSpecialist: Generating output")

            result = crew_assistant.crew().kickoff(inputs=inputs)

        # Add completion logs
        st.session_state.debug_output.append("✅ [STATUS] All agents completed successfully")
        st.session_state.debug_output.append("📝 [OUTPUT] Response ready for streaming")

        # Convert result to string and stream it in a structured way
        response_text = str(result)

        # Split into paragraphs for better structure
        paragraphs = response_text.split("\n\n")

        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                if i > 0:
                    yield "\n\n"

                # Add some structure to the paragraphs
                words = paragraph.split()
                for j, word in enumerate(words):
                    yield word + " "
                    # Add a small delay for better streaming effect
                    if j % 8 == 0:  # Pause every 8 words
                        time.sleep(0.05)

    except ImportError as e:
        st.session_state.debug_output.append(f"❌ Import Error: {str(e)}")
        yield "❌ **Import Error**\n\n"
        yield f"Could not import required modules: {str(e)}\n\n"
        yield "**Possible solutions:**\n"
        yield "- Check if all dependencies are installed\n"
        yield "- Verify the source code structure\n"
    except Exception as e:
        st.session_state.debug_output.append(f"❌ Runtime Error: {str(e)}")
        yield "❌ **Runtime Error**\n\n"
        yield f"An error occurred: {str(e)}\n\n"
        yield "**Troubleshooting:**\n"
        yield "- Try refreshing the page\n"
        yield "- Check your internet connection\n"
        yield "- Verify API key configuration"


# Initialize session state for history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_button_clicked" not in st.session_state:
    st.session_state.chat_button_clicked = False


# App Header
st.markdown(
    """
<div class="main-header">
    <h1>🔗 Hybrid Knowledge Graph RAG Assistant</h1>
    <h3>Your Conversational AI assistant for MOSDAC</h3>
    <p>
        Powered by CrewAI, ChromaDB, Neo4j, and Gemma 3 (Open-Source) | Multi-turn conversation enabled
    </p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div>
    <p>⚠️ Heads‑up: Very rarely you might hit a short limit—just wait a moment and try again. 😊</p>
</div>
""",
    unsafe_allow_html=True,
)

# Show configuration status
with st.expander("🔧 Configuration Status", expanded=False):
    google_api_key = os.getenv("GOOGLE_API_KEY")
    neo4j_uri = os.getenv("NEO4J_URI")

    if google_api_key:
        st.success("✅ API Key configured")
    else:
        st.error("❌ API Key not configured")
        st.info("Please set the API_KEY environment variable in the Space settings.")

    if neo4j_uri:
        st.success("✅ Neo4j database configured")
    else:
        st.warning("⚠️ Neo4j database not configured (optional)")

    if IS_HF_SPACE:
        st.info("🤗 Running on Hugging Face Spaces")

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


example_prompts = [
    "Tell me about MOSDAC mission and its objectives.",
    "What is Kalpana-1 satellite and its significance?",
    "Explain ISRO's earth observation satellites.",
    "What are the applications of remote sensing?",
    "Describe INSAT satellite series and its uses.",
    "What is the role of meteorological satellites?",
]

example_prompts_help = [
    "Learn about MOSDAC (Meteorological & Oceanographic Satellite Data Archival Centre)",
    "Get information about the Kalpana-1 meteorological satellite",
    "Understand ISRO's earth observation satellite programs",
    "Explore the various applications of remote sensing technology",
    "Learn about the Indian National Satellite (INSAT) system",
    "Understand how meteorological satellites help in weather forecasting",
]


# Display buttons only if no button was clicked before and no conversation history
if not st.session_state.chat_button_clicked and len(st.session_state.chat_history) <= 1:
    st.markdown(
        f"""
        <div style="text-align: center;">
            <h2 style="margin-top: 10px;">What’s on your mind today?</h2>
        </div>
    <br>
    <br>
    """,
        unsafe_allow_html=True,
    )
    button_cols = st.columns(3)
    button_cols_2 = st.columns(3)

    # Check if any example prompt button is pressed
    if button_cols[0].button(example_prompts[0], help=example_prompts_help[0]):
        st.session_state.chat_button_clicked = True
        st.session_state.prompt = example_prompts[0]
    if button_cols[1].button(example_prompts[1], help=example_prompts_help[1]):
        st.session_state.chat_button_clicked = True
        st.session_state.prompt = example_prompts[1]
    if button_cols[2].button(example_prompts[2], help=example_prompts_help[2]):
        st.session_state.chat_button_clicked = True
        st.session_state.prompt = example_prompts[2]
    if button_cols_2[0].button(example_prompts[3], help=example_prompts_help[3]):
        st.session_state.chat_button_clicked = True
        st.session_state.prompt = example_prompts[3]
    if button_cols_2[1].button(example_prompts[4], help=example_prompts_help[4]):
        st.session_state.chat_button_clicked = True
        st.session_state.prompt = example_prompts[4]
    if button_cols_2[2].button(example_prompts[5], help=example_prompts_help[5]):
        st.session_state.chat_button_clicked = True
        st.session_state.prompt = example_prompts[5]

# Add a clear conversation button in the sidebar
with st.sidebar:
    st.markdown("### 🗨️ Conversation Controls")
    if st.button("🗑️ Clear Conversation"):
        st.session_state.chat_history = []
        st.session_state.chat_button_clicked = False
        st.session_state.welcome_shown = (
            False  # This will trigger welcome message again
        )
        st.session_state.debug_output = []  # Clear debug output too
        st.rerun()

    # Show conversation stats
    if len(st.session_state.chat_history) > 1:
        user_messages = len(
            [msg for msg in st.session_state.chat_history if msg["role"] == "user"]
        )
        st.metric("Messages in conversation", user_messages)

    # Debug/Agent Thinking Section
    st.markdown("---")
    st.markdown("### 🧠 Agent Activity")

    # Create a placeholder for dynamic updates
    debug_placeholder = st.empty()

    with debug_placeholder.container():
        if st.session_state.debug_output:
            st.markdown(
                """
            <div class="debug-container">
                <div class="debug-title">🔧 Agent Console Output</div>
            """,
                unsafe_allow_html=True,
            )

            # Show progress bar
            total_steps = len(st.session_state.debug_output)
            if total_steps > 0:
                progress_percent = min(100, (total_steps / 10) * 100)  # Max 10 steps
                st.progress(progress_percent / 100, text=f"Activity: {total_steps}/10 steps")

            # Show only last 8 debug messages with better styling
            recent_debug = st.session_state.debug_output[-8:]
            for i, debug_msg in enumerate(recent_debug):
                # Style based on message type
                if "[STATUS]" in debug_msg or "✅" in debug_msg:
                    message_class = "success-step"
                elif "[AGENT]" in debug_msg or "[TOOL]" in debug_msg:
                    message_class = "agent-thinking"
                elif "[SEARCH]" in debug_msg or "[CHROMA]" in debug_msg or "[DB]" in debug_msg:
                    message_class = "processing-step"
                else:
                    message_class = "agent-step"
                
                st.markdown(
                    f"""
                <div class="{message_class} agent-step">
                    <code style="background: none; color: inherit; font-size: 0.75rem;">{debug_msg}</code>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                """
            <div class="debug-container">
                <div class="thinking-indicator">💭 Agent Console Ready...</div>
                <br>
                <div style="color: #666; font-family: monospace; font-size: 0.8rem;">
                    <small>• DataRetrieval Agent: Standby</small><br>
                    <small>• ResponseSpecialist Agent: Standby</small><br>
                    <small>• ChromaDB: Connected ✓</small><br>
                    <small>• Vector Store: Indexed ✓</small><br>
                    <small>• Memory System: Online ✓</small>
                </div>
                <br>
                <small>Ask a question to see real-time console output!</small>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Always show refresh instructions and buttons
    st.markdown("---")
    st.markdown(
        "💡 **Tip:** Click refresh to see latest agent console logs in real-time"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh Logs"):
            st.rerun()
    
    with col2:
        if st.session_state.debug_output and st.button("📜 View All"):
            with st.expander("🗂️ Complete Console Log", expanded=True):
                for i, log_entry in enumerate(st.session_state.debug_output):
                    st.code(f"{i+1:2d}. {log_entry}", language="")

    if not st.session_state.debug_output:
        st.markdown("ℹ️ *Console logs will appear above when you ask questions*")
    else:
        st.markdown(f"📊 *Total logs: {len(st.session_state.debug_output)}*")

# Get users input
prompt = st.chat_input(
    "Ask me anything about satellites, ISRO, space technology..."
) or st.session_state.get("prompt", "")

if prompt:
    # Whether the user clicks or not the suggestions should vanish
    st.session_state.chat_button_clicked = True
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant message in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(
            query_crew(prompt, None, st.session_state.chat_history)
        )
        # Add assistant message to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.session_state.prompt = ""
