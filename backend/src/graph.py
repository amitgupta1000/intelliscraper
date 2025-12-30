# graph.py
# This file defines the LangGraph workflow.
from .logging_setup import logger

# Try to import LangGraph; provide a minimal fallback for static linting/runtime without the package
try:
    from langgraph.graph import StateGraph, START, END, add_messages
except (ImportError, Exception):
    logging.warning("langgraph.graph not available; using fallback StateGraph for static checks.")
    # Minimal fallback implementation so other modules can import graph during lint/static analysis
    START = "__START__"
    END = "__END__"
    add_messages = None
    class StateGraph:
        def __init__(self, state_type=None):
            self.nodes = {}
        def add_node(self, name, func):
            self.nodes[name] = func
        def add_edge(self, a, b):
            pass
        def add_conditional_edges(self, node, route_fn, mapping):
            pass
        def compile(self):
            logger.info("Fallback StateGraph.compile() called — no-op.")
            return None

from typing import TypedDict, Optional, List, Dict, Any # Import necessary types
# Import nodes and AgentState from nodes.py
try:
    from .nodes import (
        AgentState,
        check_infrastructure,
        start_research,
        route_follow_up_question,
        create_queries,
        fast_search_results_to_final_urls,
        extract_content,
        classic_retrieve,
        fss_retrieve,
        AI_evaluate,
        write_report,
    )
except ImportError as e:
    import logging
    logger.exception("Error importing nodes: %s. Cannot define graph.", e)
    
# Initialize StateGraph
workflow = StateGraph(AgentState)

# Add nodes # Assuming all imported node functions are async
if 'check_infrastructure' in locals():
    workflow.add_node("check_infrastructure", check_infrastructure)
if 'start_research' in locals():
    workflow.add_node("start_research", start_research)
if 'route_follow_up_question' in locals():
    workflow.add_node("route_follow_up_question", route_follow_up_question)
if 'create_queries' in locals():
    workflow.add_node("create_queries", create_queries)
if 'fast_search_results_to_final_urls' in locals():
    workflow.add_node("fast_search_results_to_final_urls", fast_search_results_to_final_urls)
if 'extract_content' in locals():
    workflow.add_node("extract_content", extract_content)
if 'classic_retrieve' in locals():
    workflow.add_node("classic_retrieve", classic_retrieve) 
if 'fss_retrieve' in locals():
    workflow.add_node("fss_retrieve", fss_retrieve)
if 'AI_evaluate' in locals():
    workflow.add_node("AI_evaluate", AI_evaluate)
if 'write_report' in locals():
    workflow.add_node("write_report", write_report)

# Define the workflow logic

if all(node_name in workflow.nodes for node_name in ["check_infrastructure", "route_follow_up_question", "create_queries", "fast_search_results_to_final_urls", "extract_content", "classic_retrieve", "fss_retrieve", "write_report"]):
    # The graph now starts with a health/infrastructure check
    workflow.add_edge(START, "check_infrastructure")
    # Start research session then create queries (no prior conversation)
    workflow.add_edge("check_infrastructure", "start_research")
    workflow.add_edge("start_research", "create_queries")
    workflow.add_edge("create_queries", "fast_search_results_to_final_urls")
    workflow.add_edge("fast_search_results_to_final_urls", "extract_content")
    workflow.add_edge("extract_content", "classic_retrieve")  # Default edge; may be overridden by conditional edge
    #workflow.add_edge("classic_retrieve", "AI_evaluate") 
    #workflow.add_edge("fss_retrieve", "AI_evaluate")
    workflow.add_edge("classic_retrieve", "write_report")
    workflow.add_edge("write_report", END)
    
    # # Conditional edge: after routing, decide whether to start a new search or go straight to the report
    # def decide_search_or_report(state):
    #     if state.get("requires_new_search"):
    #         logger.info("Graph decision: Proceeding to 'create_queries' for a new search.")
    #         return "create_queries"
    #     else:
    #         logger.info("Graph decision: Bypassing search, proceeding to 'write_report'.")
    #         return "write_report"

    # workflow.add_conditional_edges(
    #     "route_follow_up_question",
    #     decide_search_or_report,
    #     {"create_queries": "create_queries", "write_report": "write_report"})

    #workflow.add_edge("extract_content", "fss_retrieve")      # Default edge; may be overridden by conditional edge
    # Conditional edge: choose retriever based on retrieval_method
    # def choose_retriever(state):
    #     method = state.get("retrieval_method", "classic")
    #     if method == "classic":
    #         return "classic_retrieve"
    #     return "fss_retrieve"

    # workflow.add_conditional_edges(
    #     "extract_content",
    #     choose_retriever,
    #     {
    #         "classic_retrieve": "classic_retrieve",
    #         "fss_retrieve": "fss_retrieve"
    #     }
    # )
    
    # # Conditional edge: After evaluation, decide to loop back for more info or finish
    # def decide_refinement(state):
    #     if state.get("proceed"):
    #         return "write_report"
    #     return "create_queries"

    # workflow.add_conditional_edges(
    #     "AI_evaluate",
    #     decide_refinement,
    #     {
    #         "write_report": "write_report",
    #         "create_queries": "create_queries"
    #     }
    # )



# Compile the workflow if nodes were successfully added
try:
    app = workflow.compile()
    logger.info("LangGraph workflow compiled successfully.")
except Exception as e:
    logger.exception("Error compiling LangGraph workflow: %s", e)
    app = None # Set app to None if compilation fails
