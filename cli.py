#!/usr/bin/env python3
"""
Minimal CLI interface for CORTEX.
Allows interactive querying with streaming output and confidence scores.
"""

from typing import Dict
from cortex.router import execute, Route, route_query
from cortex.streaming import StreamHandler


def print_header():
    """Print welcome header."""
    print("\n" + "=" * 60)
    print("  CORTEX - Interactive CLI")
    print("=" * 60)
    print("Type your query and press Enter.")
    print("Type 'exit', 'quit', or 'q' to exit.")
    print("Type 'clear' to clear the screen.")
    print("-" * 60 + "\n")


def format_confidence_bar(confidence: float, width: int = 20) -> str:
    """
    Create a visual progress bar for confidence score.
    
    Args:
        confidence: Score between 0 and 1
        width: Width of the bar in characters
    
    Returns:
        Formatted bar string
    """
    filled = int(confidence * width)
    bar = "█" * filled + "░" * (width - filled)
    return bar


def print_confidence_scores(scores: Dict[str, float], predicted_route: Route):
    """
    Print confidence scores with visual bars.
    
    Args:
        scores: Dictionary of route -> confidence score
        predicted_route: The route that was predicted/used
    """
    reset = "\033[0m"
    dim = "\033[2m"
    bold = "\033[1m"
    
    route_colors = {
        "rag": "\033[94m",      # Blue
        "meta": "\033[93m",     # Yellow
        "chat": "\033[92m",     # Green
    }
    
    print(f"\n{dim}Confidence Scores:{reset}")
    
    # Sort by confidence (highest first)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    for route_name, confidence in sorted_scores:
        color = route_colors.get(route_name, reset)
        bar = format_confidence_bar(confidence)
        
        # Highlight the predicted route
        if route_name == predicted_route.value:
            print(f"  {bold}{color}► {route_name.upper():<6}{reset} {color}{bar}{reset} {bold}{confidence:.1%}{reset}")
        else:
            print(f"  {dim}  {route_name.upper():<6} {bar} {confidence:.1%}{reset}")


def print_response_header(route: Route, confidence: float):
    """
    Print response header with route info and confidence.
    
    Args:
        route: The route being used
        confidence: Confidence score for this route (0-1)
    """
    route_colors = {
        Route.RAG: "\033[94m",      # Blue
        Route.META: "\033[93m",     # Yellow
        Route.CHAT: "\033[92m",     # Green
    }
    reset = "\033[0m"
    bold = "\033[1m"
    
    color = route_colors.get(route, reset)
    route_name = route.value.upper()
    
    print(f"\n{bold}{color}[{route_name}]{reset} {color}({confidence:.1%}){reset} ", end="", flush=True)


def stream_output(query: str, show_confidence: bool = True):
    """
    Execute query with streaming output.
    
    Args:
        query: User's question
        show_confidence: Whether to display confidence scores
    
    Returns:
        tuple: (result, route, confidence_scores)
    """
    def on_token(token: str):
        """Callback for each token."""
        print(token, end="", flush=True)
    
    handler = StreamHandler(on_token=on_token)
    
    try:
        # Get route prediction and confidence scores first
        predicted_route, confidence_scores = route_query(query)
        predicted_confidence = confidence_scores[predicted_route.value]
        
        # Show initial routing decision
        print_response_header(predicted_route, predicted_confidence)
        
        # Execute with streaming
        result, final_route, final_scores = execute(query, callbacks=[handler])
        
        # If route changed (RAG -> CHAT fallback), update display
        if final_route != predicted_route:
            fallback_confidence = final_scores[final_route.value]
            print(f"\n\033[93m[Note: Fallback to {final_route.value.upper()} mode ({fallback_confidence:.1%})]\033[0m")
        
        print()  # New line after streaming
        
        # Show confidence breakdown
        if show_confidence:
            print_confidence_scores(final_scores, final_route)
        
        return result, final_route, final_scores
        
    except Exception as e:
        print(f"\n\033[91mError: {e}\033[0m")
        import traceback
        traceback.print_exc()
        return None, None, None


def print_stats_summary(route: Route, confidence: float):
    """
    Print a summary line after the response.
    
    Args:
        route: Route that was used
        confidence: Confidence score
    """
    reset = "\033[0m"
    dim = "\033[2m"
    print(f"{dim}{'─' * 60}{reset}")


def main():
    """Main interactive loop."""
    print_header()
    
    # Configuration
    show_confidence = True  # Can be toggled via command
    
    while True:
        try:
            # Get user input
            query = input("\n> ").strip()
            
            # Handle special commands
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!\n")
                break
            
            if query.lower() == 'clear':
                import os
                os.system('clear' if os.name != 'nt' else 'cls')
                print_header()
                continue
            
            if query.lower() == 'toggle confidence':
                show_confidence = not show_confidence
                status = "enabled" if show_confidence else "disabled"
                print(f"\033[93mConfidence scores {status}\033[0m")
                continue
            
            if query.lower() in ['help', '?']:
                print("\nAvailable commands:")
                print("  exit, quit, q          - Exit the CLI")
                print("  clear                  - Clear the screen")
                print("  toggle confidence      - Toggle confidence score display")
                print("  help, ?                - Show this help message")
                continue
            
            # Execute query with streaming
            result, route, scores = stream_output(query, show_confidence=show_confidence)
            
            if result is None:
                continue
            
            # Print summary separator
            if route and scores:
                print_stats_summary(route, scores[route.value])
                
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'exit' to quit or continue querying.\n")
            continue
        except EOFError:
            print("\n\nGoodbye!\n")
            break
        except Exception as e:
            print(f"\n\033[91mUnexpected error: {e}\033[0m")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()