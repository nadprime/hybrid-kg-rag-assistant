from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()


class Neo4jQueryInput(BaseModel):
    """Input schema for Neo4j AuraDB Tool."""

    query: str = Field(
        ...,
        description="Natural language query or specific information to search for in the knowledge graph",
    )
    query_type: str = Field(
        default="semantic",
        description="Type of query: 'semantic' for natural language search, 'relationship' for finding connections, 'entity' for specific entity info, 'cypher' for direct Cypher queries",
    )
    limit: int = Field(default=10, description="Maximum number of results to return")


class Neo4jAuraDBTool(BaseTool):
    name: str = "Neo4j AuraDB Knowledge Graph Tool"
    description: str = (
        "A powerful tool for querying a Neo4j knowledge graph containing MOSDAC satellite data, missions, and page relationships. "
        "This tool has access to structured data about Indian satellites (KALPANA-1, INSAT-3DR, etc.), missions (IRS), "
        "MOSDAC pages, and their interconnections. It can find satellite specifications, mission details, page relationships, "
        "and contextual connections that RAG tools cannot provide. Use this when you need structured satellite data, "
        "mission relationships, or when basic text search doesn't provide enough context about satellites and space missions."
    )
    args_schema: Type[BaseModel] = Neo4jQueryInput

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Store connection details as private attributes after super().__init__()
        object.__setattr__(self, "_neo4j_uri", os.getenv("NEO4J_URI"))
        object.__setattr__(
            self, "_neo4j_username", os.getenv("NEO4J_USERNAME", "neo4j")
        )
        object.__setattr__(self, "_neo4j_password", os.getenv("NEO4J_PASSWORD"))
        object.__setattr__(
            self, "_neo4j_database", os.getenv("NEO4J_DATABASE", "neo4j")
        )

        if not all([self._neo4j_uri, self._neo4j_password]):
            raise ValueError(
                "Neo4j connection details are required. Please set NEO4J_URI and NEO4J_PASSWORD environment variables."
            )

        try:
            object.__setattr__(
                self,
                "_neo4j_driver",
                GraphDatabase.driver(
                    self._neo4j_uri, auth=(self._neo4j_username, self._neo4j_password)
                ),
            )
        except Exception as e:
            raise ValueError(f"Failed to connect to Neo4j: {str(e)}")

    def _run(self, query: str, query_type: str = "semantic", limit: int = 10) -> str:
        """Execute Neo4j query based on the query type and return formatted results."""
        try:
            with self._neo4j_driver.session(database=self._neo4j_database) as session:
                # Check query_type first to handle explicit cypher queries
                if query_type == "cypher":
                    return self._execute_cypher(session, query, limit)
                elif query_type == "relationship":
                    return self._find_relationships(session, query, limit)
                elif query_type == "entity":
                    return self._get_entity_info(session, query, limit)
                elif query_type == "semantic":
                    # For semantic queries, check for specific satellite/mission keywords
                    query_lower = query.lower()
                    if any(
                        sat in query_lower for sat in ["kalpana", "insat", "satellite"]
                    ):
                        return self._satellite_search(session, query, limit)
                    elif any(
                        mission in query_lower for mission in ["mission", "irs", "isro"]
                    ):
                        return self._mission_search(session, query, limit)
                    else:
                        return self._semantic_search(session, query, limit)
                else:
                    # Default to semantic search
                    return self._semantic_search(session, query, limit)
        except Exception as e:
            return f"Error querying Neo4j: {str(e)}"

    def _semantic_search(self, session, query: str, limit: int) -> str:
        """Perform semantic search across nodes and relationships."""
        # Try full-text search first, fall back to basic search if not available
        try:
            cypher_query = """
            CALL db.index.fulltext.queryNodes('entitySearch', $search_query)
            YIELD node, score
            OPTIONAL MATCH (node)-[r]-(connected)
            RETURN node, score, collect(DISTINCT {
                relationship: type(r),
                direction: CASE WHEN startNode(r) = node THEN 'outgoing' ELSE 'incoming' END,
                connectedNode: {
                    labels: labels(connected),
                    properties: properties(connected)
                }
            }) as relationships
            ORDER BY score DESC
            LIMIT $search_limit
            """
            result = session.run(cypher_query, search_query=query, search_limit=limit)
            return self._format_search_results(result, "Semantic Search (Full-text)")
        except:
            # Fall back to basic text search
            cypher_query = """
            MATCH (node)
            WHERE ANY(label IN labels(node) WHERE toLower(label) CONTAINS toLower($search_query))
               OR ANY(key IN keys(node) WHERE toLower(toString(node[key])) CONTAINS toLower($search_query))
            OPTIONAL MATCH (node)-[r]-(connected)
            RETURN node, 1.0 as score, collect(DISTINCT {
                relationship: type(r),
                direction: CASE WHEN startNode(r) = node THEN 'outgoing' ELSE 'incoming' END,
                connectedNode: {
                    labels: labels(connected),
                    properties: properties(connected)
                }
            }) as relationships
            LIMIT $search_limit
            """
            result = session.run(cypher_query, search_query=query, search_limit=limit)
            return self._format_search_results(result, "Semantic Search (Basic)")

    def _satellite_search(self, session, query: str, limit: int) -> str:
        """Search for satellite-specific information and related pages."""
        cypher_query = """
        MATCH (p:Page)
        WHERE toLower(p.title) CONTAINS toLower($search_query)
           OR toLower(p.text_content) CONTAINS toLower($search_query)
           OR p.has_satellite_content = true
        OPTIONAL MATCH (p)-[r]-(connected)
        RETURN p, 
               p.title as title,
               p.url as url,
               p.has_satellite_content as has_satellite_content,
               collect(DISTINCT {
                   relationship: type(r),
                   connected: labels(connected)[0],
                   properties: properties(connected)
               }) as relationships
        ORDER BY p.has_satellite_content DESC, 
                 CASE WHEN toLower(p.title) CONTAINS toLower($search_query) THEN 1 ELSE 2 END
        LIMIT $search_limit
        """
        result = session.run(cypher_query, search_query=query, search_limit=limit)
        return self._format_satellite_results(result)

    def _mission_search(self, session, query: str, limit: int) -> str:
        """Search for mission-specific information and relationships."""

        # Extract mission keywords from the query
        mission_keywords = [
            "irs",
            "insat",
            "kalpana",
            "cartosat",
            "resourcesat",
            "risat",
            "oceansat",
        ]
        query_words = query.lower().split()

        # Try to find specific mission names first
        search_terms = []
        for word in query_words:
            if (
                word in mission_keywords or len(word) >= 3
            ):  # Include mission keywords or significant words
                search_terms.append(word)

        if not search_terms:
            search_terms = [query.lower()]  # Use the whole query as fallback

        cypher_query = """
        // Search for missions by name (flexible matching)
        MATCH (m:Mission)
        WHERE ANY(term IN $search_terms WHERE toLower(m.name) CONTAINS term)
        OPTIONAL MATCH (m)-[r]-(connected)
        RETURN m, 
               m.name as mission_name,
               collect(DISTINCT {
                   relationship: type(r),
                   connected: labels(connected)[0],
                   properties: properties(connected)
               }) as relationships
        UNION
        // Search for pages that mention missions
        MATCH (p:Page)-[:MENTIONS]->(m:Mission)
        WHERE ANY(term IN $search_terms WHERE toLower(m.name) CONTAINS term OR toLower(p.title) CONTAINS term)
        RETURN p as m,
               p.title as mission_name,
               [{relationship: "MENTIONS", connected: "Mission", properties: {name: m.name}}] as relationships
        UNION
        // Search for any mission when query contains general mission terms
        MATCH (m:Mission)
        WHERE $has_mission_term = true
        OPTIONAL MATCH (m)-[r]-(connected)
        RETURN m,
               m.name as mission_name,
               collect(DISTINCT {
                   relationship: type(r),
                   connected: labels(connected)[0],
                   properties: properties(connected)
               }) as relationships
        LIMIT $search_limit
        """

        has_mission_term = any(
            term in query.lower() for term in ["mission", "program", "project"]
        )

        result = session.run(
            cypher_query,
            search_terms=search_terms,
            has_mission_term=has_mission_term,
            search_limit=limit,
        )
        return self._format_mission_results(result)

    def _find_relationships(self, session, query: str, limit: int) -> str:
        """Find relationships and connections related to the query."""
        cypher_query = """
        MATCH (n)-[r]-(m)
        WHERE ANY(label IN labels(n) WHERE toLower(label) CONTAINS toLower($search_query))
           OR ANY(label IN labels(m) WHERE toLower(label) CONTAINS toLower($search_query))
           OR toLower(type(r)) CONTAINS toLower($search_query)
           OR ANY(key IN keys(n) WHERE toLower(toString(n[key])) CONTAINS toLower($search_query))
           OR ANY(key IN keys(m) WHERE toLower(toString(m[key])) CONTAINS toLower($search_query))
        RETURN n, r, m, 
               {source: properties(n), relationship: type(r), target: properties(m)} as relationship_info
        LIMIT $search_limit
        """

        result = session.run(cypher_query, search_query=query, search_limit=limit)
        return self._format_relationship_results(result)

    def _get_entity_info(self, session, query: str, limit: int) -> str:
        """Get detailed information about specific entities."""
        cypher_query = """
        MATCH (n)
        WHERE ANY(label IN labels(n) WHERE toLower(label) CONTAINS toLower($search_query))
           OR ANY(key IN keys(n) WHERE toLower(toString(n[key])) CONTAINS toLower($search_query))
        OPTIONAL MATCH (n)-[r]-(connected)
        RETURN n, 
               labels(n) as labels,
               properties(n) as properties,
               collect(DISTINCT {
                   relationship: type(r),
                   direction: CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END,
                   connectedNode: labels(connected)[0],
                   connectedProperties: properties(connected)
               }) as connections
        LIMIT $search_limit
        """

        result = session.run(cypher_query, search_query=query, search_limit=limit)
        return self._format_entity_results(result)

    def _execute_cypher(self, session, cypher_query: str, limit: int) -> str:
        """Execute a custom Cypher query."""
        try:
            # Handle special cases for Neo4j system queries
            if "db.labels()" in cypher_query:
                cypher_query = "CALL db.labels() YIELD label RETURN label"
            elif "db.relationshipTypes()" in cypher_query:
                cypher_query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"

            # Add LIMIT if not present and not a CALL query
            if (
                "LIMIT" not in cypher_query.upper()
                and not cypher_query.strip().upper().startswith("CALL")
            ):
                cypher_query += f" LIMIT {limit}"

            result = session.run(cypher_query)
            return self._format_cypher_results(result)
        except Exception as e:
            return f"Error executing Cypher query: {str(e)}"

    def _format_search_results(self, result, search_type: str) -> str:
        """Format search results into a readable string."""
        output = [f"\n=== {search_type} Results ==="]

        for record in result:
            node = record["node"]
            score = record.get("score", 0)
            relationships = record.get("relationships", [])

            output.append(f"\n🔍 Entity (Score: {score:.2f}):")
            output.append(f"  Labels: {list(node.labels)}")
            output.append(f"  Properties: {dict(node)}")

            if relationships and any(r for r in relationships if r.get("relationship")):
                output.append("  Connected to:")
                for rel in relationships[:3]:  # Limit to first 3 relationships
                    if rel.get("relationship"):
                        output.append(
                            f"    - {rel['direction']} {rel['relationship']} -> {rel.get('connectedNode', {}).get('labels', [])} {rel.get('connectedNode', {}).get('properties', {})}"
                        )

        return "\n".join(output) if len(output) > 1 else "No results found."

    def _format_relationship_results(self, result) -> str:
        """Format relationship results into a readable string."""
        output = ["\n=== Relationship Analysis ==="]

        for record in result:
            rel_info = record["relationship_info"]
            output.append(f"\n🔗 Relationship:")
            output.append(f"  Source: {rel_info['source']}")
            output.append(f"  Relationship: {rel_info['relationship']}")
            output.append(f"  Target: {rel_info['target']}")

        return "\n".join(output) if len(output) > 1 else "No relationships found."

    def _format_entity_results(self, result) -> str:
        """Format entity results into a readable string."""
        output = ["\n=== Entity Information ==="]

        for record in result:
            output.append(f"\n📍 Entity:")
            output.append(f"  Labels: {record['labels']}")
            output.append(f"  Properties: {record['properties']}")

            connections = record.get("connections", [])
            if connections and any(c for c in connections if c.get("relationship")):
                output.append("  Connections:")
                for conn in connections[:5]:  # Limit to first 5 connections
                    if conn.get("relationship"):
                        output.append(
                            f"    - {conn['direction']} {conn['relationship']} -> {conn.get('connectedNode', 'Unknown')} {conn.get('connectedProperties', {})}"
                        )

        return "\n".join(output) if len(output) > 1 else "No entities found."

    def _format_cypher_results(self, result) -> str:
        """Format Cypher query results into a readable string."""
        output = ["\n=== Cypher Query Results ==="]

        for record in result:
            output.append(f"\nRecord: {dict(record)}")

        return "\n".join(output) if len(output) > 1 else "No results from Cypher query."

    def _format_satellite_results(self, result) -> str:
        """Format satellite-specific results into a readable string."""
        output = ["\n=== 🛰️ Satellite Information ==="]

        for record in result:
            page = record["m"] if "m" in record else record.get("p")
            title = record.get("title", "Unknown")
            url = record.get("url", "")
            has_satellite_content = record.get("has_satellite_content", False)
            relationships = record.get("relationships", [])

            output.append(f"\n🛰️ Satellite Page: {title}")
            if url:
                output.append(f"   URL: {url}")
            output.append(f"   Has Satellite Content: {has_satellite_content}")

            if hasattr(page, "get") or hasattr(page, "__getitem__"):
                props = dict(page) if hasattr(page, "__iter__") else page
                if "text_content" in props and props["text_content"]:
                    content = (
                        props["text_content"][:200] + "..."
                        if len(props["text_content"]) > 200
                        else props["text_content"]
                    )
                    output.append(f"   Content Preview: {content}")

            if relationships and any(r for r in relationships if r.get("relationship")):
                output.append("   Related Information:")
                for rel in relationships[:3]:
                    if rel.get("relationship"):
                        output.append(
                            f"     - {rel['relationship']} -> {rel.get('connected', 'Unknown')} {rel.get('properties', {})}"
                        )

        return (
            "\n".join(output) if len(output) > 1 else "No satellite information found."
        )

    def _format_mission_results(self, result) -> str:
        """Format mission-specific results into a readable string."""
        output = ["\n=== 🚀 Mission Information ==="]

        for record in result:
            mission_name = record.get("mission_name", "Unknown")
            relationships = record.get("relationships", [])

            output.append(f"\n🚀 Mission: {mission_name}")

            if relationships and any(r for r in relationships if r.get("relationship")):
                output.append("   Connections:")
                for rel in relationships[:3]:
                    if rel.get("relationship"):
                        output.append(
                            f"     - {rel['relationship']} -> {rel.get('connected', 'Unknown')} {rel.get('properties', {})}"
                        )

        return "\n".join(output) if len(output) > 1 else "No mission information found."

    def __del__(self):
        """Close the Neo4j driver when the tool is destroyed."""
        if hasattr(self, "_neo4j_driver") and self._neo4j_driver:
            self._neo4j_driver.close()
