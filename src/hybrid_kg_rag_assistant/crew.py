import os
import warnings
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from dotenv import load_dotenv
from crewai_tools import RagTool
from .tools.custom_tool import Neo4jAuraDBTool

# Suppress common warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="alembic")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="chromadb")
warnings.filterwarnings("ignore", message=".*PydanticDeprecatedSince20.*")
warnings.filterwarnings("ignore", message=".*PydanticDeprecatedSince211.*")
warnings.filterwarnings("ignore", message=".*Using extra keyword arguments.*")
warnings.filterwarnings("ignore", message=".*Extra keys.*")
warnings.filterwarnings("ignore")  # Catch all remaining warnings

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL = os.getenv("MODEL")

# Check if API key is available
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Configure the LLM for all agents
llm = LLM(model=MODEL, api_key=GOOGLE_API_KEY)


@CrewBase
class HybridKgRagAssistant:
    """HybridKgRagAssistant crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def hybrid_data_retriever(self) -> Agent:
        # Create RAG tool with ChromaDB and Google configuration
        # Using absolute path to ensure data files are found regardless of working directory
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        csv_path = os.path.join(current_dir, "knowledge", "mosdac_scraped_data.csv")

        # Configure RAG tool with Embedding model and ChromaDB
        rag_config = {
            "embedder": {
                "provider": "google",
                "config": {"model": "models/text-embedding-004"},
            },
            "vectordb": {
                "provider": "chroma",
                "config": {
                    "collection_name": "hybrid-kg-rag-collection",
                    "dir": os.path.join(current_dir, "chroma_db"),
                    "allow_reset": True,
                },
            },
            "chunker": {
                "chunk_size": 400,
                "chunk_overlap": 50,
                "length_function": "len",
                "min_chunk_size": 100,
            },
        }

        rag_tool = RagTool(config=rag_config)

        # # Add MOSDAC CSV data to the RAG knowledge base
        rag_tool.add(
            data_type="csv", source=csv_path, metadata={"source": "mosdac_scraped_data"}
        )

        # Create Neo4j Knowledge Graph tool
        try:
            neo4j_tool = Neo4jAuraDBTool()
        except ValueError as e:
            print(f"Warning: Neo4j tool not available: {e}")
            neo4j_tool = None

        # Combine available tools - optimized for hybrid RAG + Neo4j workflow
        tools = [rag_tool]
        if neo4j_tool:
            tools.append(neo4j_tool)

        return Agent(
            config=self.agents_config["hybrid_data_retriever"],  # type: ignore[index]
            verbose=True,
            allow_delegation=False,  # No delegation for this agent
            tools=tools,  # Optimized RAG + Neo4j hybrid workflow
            llm=llm,  # Add the configured LLM
        )

    @agent
    def response_synthesizer(self) -> Agent:
        return Agent(
            config=self.agents_config["response_synthesizer"],  # type: ignore[index]
            verbose=True,
            allow_delegation=False,  # No delegation for this agent
            llm=llm,  # Add the configured LLM
        )

    @task
    def hybrid_data_retrieval_task(self) -> Task:
        return Task(
            config=self.tasks_config["hybrid_data_retrieval_task"],  # type: ignore[index]
        )

    @task
    def response_synthesis_task(self) -> Task:
        return Task(
            config=self.tasks_config["response_synthesis_task"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the HybridKgRagAssistant crew optimized for multi-turn conversations"""

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential, # Sequential process for multi-turn conversations
            verbose=True,
            memory=True,  # Enable memory for conversation context
            embedder={
                "provider": "google",
                "config": {
                    "api_key": GOOGLE_API_KEY,
                    "model": "models/text-embedding-004",
                },
            },
        )
