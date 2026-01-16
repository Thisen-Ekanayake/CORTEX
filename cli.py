#!/usr/bin/env python3
"""
Minimal CLI interface for CORTEX.
Allows interactive querying with streaming output.
"""

from cortex.router import execute, Route
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


def print_response_header(route: Route):
    """Print response header with route info."""
    route_colors = {
        Route.RAG: "\033[94m",      # Blue
        Route.META: "\033[93m",     # Yellow
        Route.CHAT: "\033[92m",     # Green
    }
    reset = "\033[0m"
    color = route_colors.get(route, reset)
    route_name = route.value.upper()
    print(f"\n{color}[{route_name}]{reset} ", end="", flush=True)


def stream_output(query: str):
    """Execute query with streaming output."""
    def on_token(token: str):
        """Callback for each token."""
        print(token, end="", flush=True)
    
    handler = StreamHandler(on_token=on_token)
    
    try:
        # First determine route to show header
        from cortex.router import route_query
        route = route_query(query)
        print_response_header(route)
        
        # Execute with streaming
        result, final_route = execute(query, callbacks=[handler])
        
        # If route changed (RAG -> CHAT fallback), update display
        if final_route != route:
            print(f"\n\033[93m[Note: No documents found, using CHAT mode]\033[0m")
        
        print()  # New line after streaming
        return result, final_route
    except Exception as e:
        print(f"\n\033[91mError: {e}\033[0m")
        return None, None


def main():
    """Main interactive loop."""
    print_header()
    
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
            
            # Execute query with streaming
            result, route = stream_output(query)
            
            if result is None:
                continue
                
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'exit' to quit or continue querying.\n")
            continue
        except EOFError:
            print("\n\nGoodbye!\n")
            break
        except Exception as e:
            print(f"\n\033[91mUnexpected error: {e}\033[0m\n")


if __name__ == "__main__":
    main()
