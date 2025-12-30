import asyncio
import logging
from typing import Dict, Any
try:
    from langchain_core.runnables import RunnableConfig
    config = RunnableConfig(recursion_limit=100)
except Exception:
    logging.debug("langchain_core.runnables not available; continuing without RunnableConfig.")
    config = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Optional storage helper for saving scraped content to GCS/Firestore
try:
    from backend.src.storage import store_scraped_content
except Exception:
    store_scraped_content = None

# Import the compiled LangGraph application
from backend.src.graph import app as workflow_app


async def run_workflow(
    initial_query: str,
    prompt_type: str
):
    """
    Runs the LangGraph workflow with an initial query and research type.
    Args:
        initial_query: The user's initial research query.
        prompt_type: Type of prompt to use.
    """
    if workflow_app is None:
        logging.error("LangGraph app is not compiled or imported. Cannot run workflow.")
        print("Workflow cannot be run due to errors in graph compilation or imports.")
        return None

    logging.info(f"Starting workflow for query: '{initial_query}'")

    # Define the initial state for the graph
    initial_state = {
        "new_query": initial_query,
        "search_queries": [],
        "rationale": None,
        "data": [],
        "relevant_contexts": {},
        "relevant_chunks": [],
        "proceed": True,
        "visited_urls": [],
        "failed_urls": [],
        "iteration_count": 0,
        "report": None,
        "report_filename": None,
        "error": None,
        "evaluation_response": None,
        "suggested_follow_up_queries": [],
        "prompt_type": prompt_type,
        "approval_iteration_count": 0,
        "search_iteration_count": 0,
        "report_type": None
    }

    # Run the compiled workflow
    try:
        if config is not None:
            astream =workflow_app.astream(initial_state, config=config)
        else:
            astream = workflow_app.astream(initial_state)

        executed_nodes = []
        last_state = None
        async for step in astream:
            for key, value in step.items():
                logging.info("Node executed: %s", key)
                executed_nodes.append(key)
            last_state = step
    
        logging.info("Workflow finished successfully.")
        if last_state:
            final_error_state = last_state.get('error')
            if final_error_state:
                logging.warning("Workflow completed with errors: %s", final_error_state)
            else:
                logging.info("Workflow completed successfully without errors.")
            return last_state
        else:
            logging.warning("No final state returned from workflow.")
            return None
    except Exception as e:
        logging.exception(f"An error occurred during workflow execution: {e}")
        return None


def simple_cli():
    """
    Simplified CLI: prompt for query and research type, generate standard report and appendix.
    """
    print("INTELLISEARCH Research Tool")
    print("=" * 50)
    user_research_query = input("Enter your Research Query: ")

    print("\nSelect research type:")
    print("1: General")
    print("2: Legal")
    print("3: Macro")
    print("4: DeepSearch")
    print("5: Person Search")
    print("6: Investment Research")
    prompt_type_choice = input("Enter the number for your desired research type: ")

    prompt_type_mapping = {
        "1": "general",
        "2": "legal",
        "3": "macro",
        "4": "deepsearch",
        "5": "person_search",
        "6": "investment"
    }
    selected_prompt_type = prompt_type_mapping.get(prompt_type_choice, "general")
    print(f"Selected research type: {selected_prompt_type}")

    print("\n🚀 Generating standard report and appendix...")
    # Run workflow and capture the final state
    final_state = asyncio.run(run_workflow(
        user_research_query,
        selected_prompt_type
    ))
    # Output summary from final state (updated for new workflow keys)
    if final_state and final_state.get("analysis_content"):
        print("\nReport generated successfully!")
        print("Analysis filename:", final_state.get("analysis_filename", "analysis.txt"))
        print("\nAnalysis preview:\n", final_state.get("analysis_content", "")[:500])
        if final_state.get("appendix_content"):
            print("\nAppendix filename:", final_state.get("appendix_filename", "appendix.txt"))
            print("\nAppendix preview:\n", final_state.get("appendix_content", "")[:500])
        if final_state.get("error"):
            print("\n[Warning] Errors during workflow:\n", final_state.get("error"))
    else:
        print("\nReport generation failed or incomplete.")



# --- Main Execution Block ---
if __name__ == "__main__":
    simple_cli()
