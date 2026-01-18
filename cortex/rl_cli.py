"""
Interactive CLI for CORTEX with Reinforcement Learning.
User selects the actual category while the model learns from feedback.
"""

from typing import Dict, Optional
from cortex.router import Route, execute
from cortex.streaming import StreamHandler
from cortex.rl_router import get_rl_router, RLRouter


# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"

ROUTE_COLORS = {
    Route.RAG: BLUE,
    Route.META: YELLOW,
    Route.CHAT: GREEN,
}


def print_header():
    """Print welcome header."""
    print("\n" + "=" * 70)
    print(f"  {BOLD}{CYAN}CORTEX - Interactive RL Training CLI{RESET}")
    print("=" * 70)
    print("The model learns from your feedback to improve routing accuracy!")
    print("\nHow it works:")
    print("  1. Type your query")
    print("  2. See the model's prediction")
    print("  3. Select the actual category")
    print("  4. The model learns and improves over time")
    print("\nCommands: 'exit', 'quit', 'stats', 'reset', 'help'")
    print("-" * 70 + "\n")


def format_confidence_bar(confidence: float, width: int = 20) -> str:
    """Create a visual progress bar for confidence score."""
    filled = int(confidence * width)
    bar = "█" * filled + "░" * (width - filled)
    return bar


def print_prediction_panel(route: Route, scores: Dict[str, float]):
    """
    Display model's prediction with confidence scores.
    
    Args:
        route: Predicted route
        scores: Confidence scores for all routes
    """
    print(f"\n{BOLD}┌─ Model Prediction ─────────────────────────────────────────┐{RESET}")
    
    # Show predicted route prominently
    color = ROUTE_COLORS.get(route, RESET)
    confidence = scores[route.value]
    print(f"│ {BOLD}{color}► Predicted: {route.value.upper():<8}{RESET} "
          f"{color}(Confidence: {confidence:.1%}){RESET}")
    print(f"│{RESET}")
    
    # Show all confidence scores
    print(f"│ {DIM}Confidence Breakdown:{RESET}")
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    for route_name, conf in sorted_scores:
        route_obj = Route(route_name)
        color = ROUTE_COLORS.get(route_obj, RESET)
        bar = format_confidence_bar(conf)
        
        if route_name == route.value:
            print(f"│   {BOLD}{color}► {route_name.upper():<6}{RESET} "
                  f"{color}{bar}{RESET} {BOLD}{conf:.1%}{RESET}")
        else:
            print(f"│   {DIM}  {route_name.upper():<6} {bar} {conf:.1%}{RESET}")
    
    print(f"└────────────────────────────────────────────────────────────┘{RESET}\n")


def get_user_category_selection() -> Optional[Route]:
    """
    Prompt user to select the actual category.
    
    Returns:
        Selected Route or None if cancelled
    """
    print(f"{BOLD}Select the ACTUAL category for this query:{RESET}")
    print(f"  {BLUE}[1]{RESET} RAG     - Retrieve from documents")
    print(f"  {YELLOW}[2]{RESET} META    - About the system")
    print(f"  {GREEN}[3]{RESET} CHAT    - General conversation")
    print(f"  {DIM}[0]{RESET} {DIM}Cancel{RESET}")
    
    while True:
        try:
            choice = input(f"\n{BOLD}Your choice [1-3, 0 to cancel]:{RESET} ").strip()
            
            if choice == "0":
                return None
            elif choice == "1":
                return Route.RAG
            elif choice == "2":
                return Route.META
            elif choice == "3":
                return Route.CHAT
            else:
                print(f"{RED}Invalid choice. Please enter 1, 2, 3, or 0.{RESET}")
        except (KeyboardInterrupt, EOFError):
            return None


def print_feedback_result(predicted: Route, actual: Route, reward: float):
    """
    Show the learning feedback result.
    
    Args:
        predicted: What model predicted
        actual: What user selected
        reward: Reward/penalty given
    """
    correct = (predicted == actual)
    
    if correct:
        icon = "✓"
        color = GREEN
        status = "CORRECT"
        emoji = "🎉"
    else:
        icon = "✗"
        color = RED
        status = "INCORRECT"
        emoji = "📚"
    
    print(f"\n{BOLD}┌─ Learning Feedback ────────────────────────────────────────┐{RESET}")
    print(f"│ {color}{BOLD}{icon} {status}{RESET} {emoji}")
    print(f"│")
    print(f"│ Predicted: {ROUTE_COLORS[predicted]}{predicted.value.upper()}{RESET}")
    print(f"│ Actual:    {ROUTE_COLORS[actual]}{actual.value.upper()}{RESET}")
    print(f"│")
    
    if reward > 0:
        reward_color = GREEN
        reward_sign = "+"
    else:
        reward_color = RED
        reward_sign = ""
    
    print(f"│ Reward: {reward_color}{reward_sign}{reward:.2f}{RESET}")
    
    if correct:
        print(f"│ {DIM}Model confidence weights adjusted (reinforced){RESET}")
    else:
        print(f"│ {DIM}Model weights updated (learning from mistake){RESET}")
    
    print(f"└────────────────────────────────────────────────────────────┘{RESET}\n")


def print_statistics(rl_router: RLRouter):
    """Print current model statistics."""
    metrics = rl_router.get_metrics()
    
    print(f"\n{BOLD}{CYAN}┌─ Model Statistics ─────────────────────────────────────────┐{RESET}")
    print(f"{CYAN}│{RESET}")
    
    # Overall accuracy
    overall = metrics["overall_accuracy"]
    total = metrics["total_predictions"]
    correct = metrics["correct_predictions"]
    
    if overall >= 0.8:
        acc_color = GREEN
    elif overall >= 0.6:
        acc_color = YELLOW
    else:
        acc_color = RED
    
    print(f"{CYAN}│{RESET} {BOLD}Overall Accuracy:{RESET} {acc_color}{overall:.1%}{RESET} "
          f"({correct}/{total} correct)")
    print(f"{CYAN}│{RESET}")
    
    # Per-route accuracy
    print(f"{CYAN}│{RESET} {BOLD}Per-Category Performance:{RESET}")
    route_metrics = metrics["route_accuracy"]
    
    for route_name in ["rag", "meta", "chat"]:
        stats = route_metrics[route_name]
        acc = stats["accuracy"]
        route = Route(route_name)
        color = ROUTE_COLORS[route]
        
        bar = format_confidence_bar(acc, width=15)
        print(f"{CYAN}│{RESET}   {color}{route_name.upper():<6}{RESET} "
              f"{bar} {acc:.1%} ({stats['correct']}/{stats['total']})")
    
    print(f"{CYAN}│{RESET}")
    
    # Confidence weights (learned adjustments)
    print(f"{CYAN}│{RESET} {BOLD}Learned Confidence Weights:{RESET}")
    weights = metrics["confidence_weights"]
    for route_name in ["rag", "meta", "chat"]:
        weight = weights[route_name]
        route = Route(route_name)
        color = ROUTE_COLORS[route]
        
        # Weight > 1.0 means model is boosting this category
        # Weight < 1.0 means model is reducing confidence
        if weight > 1.0:
            weight_indicator = f"{GREEN}↑{RESET}"
        elif weight < 1.0:
            weight_indicator = f"{RED}↓{RESET}"
        else:
            weight_indicator = "→"
        
        print(f"{CYAN}│{RESET}   {color}{route_name.upper():<6}{RESET} "
              f"{weight:.3f} {weight_indicator}")
    
    print(f"{CYAN}└────────────────────────────────────────────────────────────┘{RESET}\n")


def execute_with_streaming(query: str, route: Route) -> str:
    """Execute query with streaming output."""
    print(f"\n{BOLD}{ROUTE_COLORS[route]}[{route.value.upper()}]{RESET} ", end="", flush=True)
    
    def on_token(token: str):
        print(token, end="", flush=True)
    
    handler = StreamHandler(on_token=on_token)
    result, _, _ = execute(query, callbacks=[handler])
    
    print("\n")  # New line after streaming
    return result


def main():
    """Main interactive loop."""
    print_header()
    
    # Initialize RL router
    rl_router = get_rl_router(learning_rate=0.15)
    
    # Show initial stats if there's prior learning
    if rl_router.total_predictions > 0:
        print(f"{YELLOW}📊 Loaded existing learning data!{RESET}")
        print_statistics(rl_router)
    
    while True:
        try:
            # Get user query
            query = input(f"{BOLD}> {RESET}").strip()
            
            if not query:
                continue
            
            # Handle commands
            if query.lower() in ['exit', 'quit', 'q']:
                print(f"\n{GREEN}Goodbye! Model improvements saved.{RESET}\n")
                break
            
            if query.lower() == 'stats':
                print_statistics(rl_router)
                continue
            
            if query.lower() == 'reset':
                confirm = input(f"{RED}⚠ Reset all learning? (yes/no): {RESET}").strip().lower()
                if confirm == 'yes':
                    rl_router.reset_learning()
                    print(f"{YELLOW}✓ Learning reset. Starting fresh!{RESET}\n")
                continue
            
            if query.lower() in ['help', '?']:
                print("\nAvailable commands:")
                print(f"  {BOLD}exit, quit, q{RESET}  - Exit the CLI")
                print(f"  {BOLD}stats{RESET}          - Show model statistics")
                print(f"  {BOLD}reset{RESET}          - Reset all learning")
                print(f"  {BOLD}help, ?{RESET}        - Show this help\n")
                continue
            
            # Get model prediction (but don't use for routing yet)
            predicted_route, confidence_scores = rl_router.route_query(query)
            
            # Show prediction to user
            print_prediction_panel(predicted_route, confidence_scores)
            
            # User selects actual category
            actual_route = get_user_category_selection()
            
            if actual_route is None:
                print(f"{DIM}Cancelled. No learning applied.{RESET}\n")
                continue
            
            # Record feedback and update model
            feedback = rl_router.record_feedback(
                query=query,
                predicted_route=predicted_route,
                actual_route=actual_route,
                confidence_scores=confidence_scores
            )
            
            # Show learning result
            print_feedback_result(predicted_route, actual_route, feedback.reward)
            
            # Execute query with the ACTUAL route (user-selected)
            execute_with_streaming(query, actual_route)
            
            # Show quick stats update every 5 queries
            if rl_router.total_predictions % 5 == 0:
                metrics = rl_router.get_metrics()
                acc = metrics["overall_accuracy"]
                total = metrics["total_predictions"]
                
                if acc >= 0.8:
                    color = GREEN
                    emoji = "🎯"
                elif acc >= 0.6:
                    color = YELLOW
                    emoji = "📈"
                else:
                    color = CYAN
                    emoji = "📊"
                
                print(f"{DIM}─── {emoji} Progress: {color}{acc:.1%}{RESET}{DIM} accuracy "
                      f"({total} queries) ───{RESET}\n")
        
        except KeyboardInterrupt:
            print(f"\n\n{YELLOW}Interrupted. Type 'exit' to quit.{RESET}\n")
            continue
        except EOFError:
            print(f"\n\n{GREEN}Goodbye!{RESET}\n")
            break
        except Exception as e:
            print(f"\n{RED}Error: {e}{RESET}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()