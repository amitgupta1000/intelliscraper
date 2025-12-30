# conditions.py
import logging

# --- Search Loop Logic ---
def should_continue_search(state: dict, max_loops: int = 3) -> bool:
    """
    Determines if search refinement should continue.
    """
    return state.get("proceed", False) and state.get("search_iteration_count", 0) < max_loops

def should_terminate_search(state: dict, max_loops: int = 3) -> bool:
    """
    Determines if search refinement should end and write report.
    """
    return not state.get("proceed", False) or state.get("search_iteration_count", 0) >= max_loops

