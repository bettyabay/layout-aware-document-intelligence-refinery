"""Query Interface Agent - Stage 5 of the Document Intelligence Refinery.

This module implements the Query Interface Agent using LangGraph, providing
three tools for querying documents:
1. pageindex_navigate - Tree traversal through PageIndex
2. semantic_search - Vector retrieval from LDUs
3. structured_query - SQL queries over extracted fact tables

Every answer includes provenance: document name, page number, and bounding box.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from typing_extensions import Annotated, TypedDict

from src.agents.indexer import PageIndexBuilder
from src.models.ldu import LDU
from src.models.page_index import PageIndex, Section
from src.models.provenance import ProvenanceChain
from src.utils.fact_table import FactTable
from src.utils.vector_store import VectorStore

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the Query Agent.

    Attributes:
        messages: List of messages in the conversation.
        query: The user's query.
        doc_id: Document ID being queried.
        doc_name: Document name.
        page_index: Loaded PageIndex for the document.
        provenance_chain: List of ProvenanceChain citations.
        answer: Final answer text.
    """

    messages: Annotated[List[Any], "add_messages"]
    query: str
    doc_id: Optional[str]
    doc_name: Optional[str]
    page_index: Optional[PageIndex]
    provenance_chain: List[ProvenanceChain]
    answer: Optional[str]


class QueryAgent:
    """Query Interface Agent with three tools for document querying.

    Tools:
        - pageindex_navigate: Navigate PageIndex tree to find relevant sections
        - semantic_search: Search LDUs using vector similarity
        - structured_query: Query extracted facts using SQL
    """

    def __init__(
        self,
        vector_store: VectorStore,
        fact_table: FactTable,
        pageindex_dir: Path,
        llm_api_key: Optional[str] = None,
        llm_model: str = "mistralai/mistral-7b-instruct",
        llm_base_url: str = "https://openrouter.ai/api/v1",
    ):
        """Initialize the Query Agent.

        Args:
            vector_store: Vector store instance for semantic search.
            fact_table: Fact table instance for SQL queries.
            pageindex_dir: Directory containing PageIndex JSON files.
            llm_api_key: API key for LLM (OpenRouter).
            llm_model: LLM model to use.
            llm_base_url: Base URL for LLM API.
        """
        self.vector_store = vector_store
        self.fact_table = fact_table
        self.pageindex_dir = Path(pageindex_dir)
        self.pageindex_dir.mkdir(parents=True, exist_ok=True)
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.pageindex_builder = PageIndexBuilder()

        # Build the LangGraph agent
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph agent graph.

        Returns:
            Compiled StateGraph.
        """
        # Create tools
        tools = [
            self._create_pageindex_navigate_tool(),
            self._create_semantic_search_tool(),
            self._create_structured_query_tool(),
        ]

        # Create tool node
        tool_node = ToolNode(tools)

        # Build graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", tool_node)

        # Add edges
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END,
            },
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def _create_pageindex_navigate_tool(self):
        """Create the pageindex_navigate tool."""

        @tool
        def pageindex_navigate(
            doc_id: str, topic: str, max_sections: int = 3
        ) -> str:
            """Navigate the PageIndex tree to find sections relevant to a topic.

            This tool traverses the hierarchical PageIndex structure to locate
            sections that match a given topic, without using vector search.

            Args:
                doc_id: Document identifier.
                topic: Topic or keyword to search for.
                max_sections: Maximum number of sections to return.

            Returns:
                JSON string with relevant sections and their page ranges.
            """
            try:
                # Load PageIndex
                page_index = self.pageindex_builder.load_page_index(
                    doc_id, self.pageindex_dir
                )

                if not page_index:
                    return json.dumps({
                        "error": f"PageIndex not found for document {doc_id}",
                        "sections": [],
                    })

                # Find relevant sections
                sections = self.pageindex_builder.find_sections_by_topic(
                    page_index, topic, top_k=max_sections
                )

                # Format results
                results = {
                    "doc_id": doc_id,
                    "doc_name": page_index.doc_name,
                    "topic": topic,
                    "sections": [
                        {
                            "title": s.title,
                            "page_start": s.page_start,
                            "page_end": s.page_end,
                            "summary": s.summary,
                            "data_types": s.data_types_present,
                            "entities": s.key_entities,
                        }
                        for s in sections
                    ],
                }

                return json.dumps(results, indent=2)

            except Exception as e:
                logger.error(f"Error in pageindex_navigate: {e}", exc_info=True)
                return json.dumps({"error": str(e), "sections": []})

        return pageindex_navigate

    def _create_semantic_search_tool(self):
        """Create the semantic_search tool."""

        @tool
        def semantic_search(
            query: str, doc_id: Optional[str] = None, top_k: int = 5
        ) -> str:
            """Search for similar content using vector similarity.

            This tool performs semantic search over LDUs using embeddings
            to find the most relevant chunks for a query.

            Args:
                query: Search query text.
                doc_id: Optional document ID to filter results.
                top_k: Number of results to return.

            Returns:
                JSON string with search results including content and provenance.
            """
            try:
                # Perform vector search
                results = self.vector_store.search(
                    query=query, doc_id=doc_id, top_k=top_k
                )

                # Format results with provenance
                formatted_results = []
                for ldu, score in results:
                    provenance = ProvenanceChain(
                        document_name="",  # Will be filled from metadata
                        page_number=min(ldu.page_refs) if ldu.page_refs else 1,
                        bbox=ldu.bounding_box,
                        content_hash=ldu.content_hash,
                        verification_status=False,
                    )

                    formatted_results.append({
                        "content": ldu.content,
                        "chunk_type": ldu.chunk_type,
                        "page_number": min(ldu.page_refs) if ldu.page_refs else 1,
                        "similarity_score": score,
                        "provenance": provenance.model_dump(),
                    })

                return json.dumps({
                    "query": query,
                    "results": formatted_results,
                    "count": len(formatted_results),
                }, indent=2)

            except Exception as e:
                logger.error(f"Error in semantic_search: {e}", exc_info=True)
                return json.dumps({"error": str(e), "results": []})

        return semantic_search

    def _create_structured_query_tool(self):
        """Create the structured_query tool."""

        @tool
        def structured_query(sql_query: str, doc_id: Optional[str] = None) -> str:
            """Execute a SQL query on the extracted fact table.

            This tool allows precise queries over structured facts extracted
            from documents (e.g., financial data, key-value pairs).

            Args:
                sql_query: SQL query string (e.g., "SELECT * FROM facts WHERE fact_key = 'Revenue'").
                doc_id: Optional document ID to filter results.

            Returns:
                JSON string with query results and provenance.
            """
            try:
                # Execute SQL query
                results = self.fact_table.query(sql_query, doc_id=doc_id)

                # Add provenance for each result
                formatted_results = []
                for row in results:
                    provenance = self.fact_table.get_provenance_for_fact(row["id"])
                    formatted_results.append({
                        "fact_key": row["fact_key"],
                        "fact_value": row["fact_value"],
                        "fact_type": row["fact_type"],
                        "page_number": row["page_number"],
                        "provenance": provenance.model_dump() if provenance else None,
                    })

                return json.dumps({
                    "sql_query": sql_query,
                    "results": formatted_results,
                    "count": len(formatted_results),
                }, indent=2)

            except Exception as e:
                logger.error(f"Error in structured_query: {e}", exc_info=True)
                return json.dumps({"error": str(e), "results": []})

        return structured_query

    def _agent_node(self, state: AgentState) -> AgentState:
        """Agent node that decides which tool to use.

        Args:
            state: Current agent state.

        Returns:
            Updated state with agent response.
        """
        # Get the last user message
        messages = state["messages"]
        if not messages:
            return state

        query = state.get("query", "")
        doc_id = state.get("doc_id")

        # Simple routing logic based on query type
        # In production, this would use an LLM to decide which tool to use
        if "SELECT" in query.upper() or "FROM facts" in query.upper():
            # SQL query
            tool_name = "structured_query"
            tool_input = {"sql_query": query, "doc_id": doc_id}
        elif any(keyword in query.lower() for keyword in ["section", "chapter", "part", "where is", "find section"]):
            # Navigation query
            tool_name = "pageindex_navigate"
            tool_input = {"doc_id": doc_id or "", "topic": query, "max_sections": 3}
        else:
            # Semantic search (default)
            tool_name = "semantic_search"
            tool_input = {"query": query, "doc_id": doc_id, "top_k": 5}

        # Create tool invocation message
        # LangGraph expects tool calls in a specific format
        tool_call_id = f"call_{tool_name}_{len(messages)}"
        tool_message = AIMessage(
            content="",
            tool_calls=[{
                "name": tool_name,
                "args": tool_input,
                "id": tool_call_id,
            }]
        )

        state["messages"].append(tool_message)

        return state

    def _should_continue(self, state: AgentState) -> str:
        """Determine whether to continue or end.

        Args:
            state: Current agent state.

        Returns:
            "continue" or "end".
        """
        messages = state["messages"]
        if not messages:
            return "end"

        last_message = messages[-1]

        # If the last message has tool calls, continue to tools
        if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"

        # If the last message is from a tool, continue to generate answer
        if isinstance(last_message, dict) and last_message.get("role") == "tool":
            return "continue"

        # Otherwise, end
        return "end"

    def query(
        self,
        query: str,
        doc_id: Optional[str] = None,
        doc_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a query and return answer with provenance.

        Args:
            query: User's query string.
            doc_id: Optional document ID.
            doc_name: Optional document name.

        Returns:
            Dictionary with answer, provenance_chain, and metadata.
        """
        # Initialize state
        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "doc_id": doc_id,
            "doc_name": doc_name,
            "page_index": None,
            "provenance_chain": [],
            "answer": None,
        }

        # Run the graph
        try:
            final_state = self.graph.invoke(initial_state)
        except Exception as e:
            logger.error(f"Error executing query: {e}", exc_info=True)
            return {
                "answer": f"Error processing query: {str(e)}",
                "provenance_chain": [],
                "error": str(e),
            }

        # Extract provenance from tool results
        provenance_chain: List[ProvenanceChain] = []
        answer_parts: List[str] = []

        for message in final_state["messages"]:
            # Handle both dict and message objects
            content = None
            if isinstance(message, dict):
                content = message.get("content", "")
            elif hasattr(message, "content"):
                content = message.content

            if content:
                try:
                    tool_result = json.loads(content) if isinstance(content, str) else content
                    if isinstance(tool_result, dict):
                        if "results" in tool_result:
                            for result in tool_result["results"]:
                                if isinstance(result, dict):
                                    if "provenance" in result:
                                        prov_data = result["provenance"]
                                        if prov_data and isinstance(prov_data, dict):
                                            try:
                                                provenance_chain.append(ProvenanceChain(**prov_data))
                                            except Exception:
                                                pass
                                    if "content" in result:
                                        answer_parts.append(str(result["content"]))
                                    if "fact_value" in result:
                                        answer_parts.append(f"{result.get('fact_key', '')}: {result['fact_value']}")
                        elif "sections" in tool_result:
                            # PageIndex navigation results
                            for section in tool_result["sections"]:
                                if isinstance(section, dict):
                                    answer_parts.append(
                                        f"Section: {section.get('title', '')} "
                                        f"(pages {section.get('page_start', '')}-{section.get('page_end', '')})"
                                    )
                except (json.JSONDecodeError, Exception) as e:
                    # If not JSON, treat as plain text
                    if isinstance(content, str) and len(content) > 0:
                        answer_parts.append(content)

        # Generate final answer
        answer = " ".join(answer_parts) if answer_parts else "No results found."

        # Add document names to provenance if missing
        for prov in provenance_chain:
            if not prov.document_name and doc_name:
                prov.document_name = doc_name

        return {
            "answer": answer,
            "provenance_chain": [p.model_dump() for p in provenance_chain],
            "query": query,
            "doc_id": doc_id,
            "doc_name": doc_name,
        }


__all__ = ["QueryAgent", "AgentState"]
