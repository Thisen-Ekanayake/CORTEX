"""
Dashboard to visualize RL router learning progress.
Shows accuracy trends, confusion matrix, and weight evolution.
"""

import json
from collections import Counter
from typing import List
from cortex.rl_router import get_rl_router, RoutingFeedback


def print_ascii_chart(data: List[float], 
                      height: int = 10, 
                      width: int = 50,
                      title: str = "Accuracy Over Time"):
    """
    Print an ASCII chart of accuracy over time.
    
    Args:
        data: List of accuracy values (0-1)
        height: Height of chart in rows
        width: Width of chart
        title: Chart title
    """
    if not data:
        print("No data to display")
        return
    
    # Normalize data to chart width
    if len(data) > width:
        # Downsample
        step = len(data) / width
        normalized = [data[int(i * step)] for i in range(width)]
    else:
        normalized = data
    
    # Scale to chart height
    max_val = max(normalized) if normalized else 1.0
    min_val = min(normalized) if normalized else 0.0
    
    print(f"\n{title}")
    print("─" * (width + 10))
    
    # Print chart rows
    for row in range(height, 0, -1):
        threshold = min_val + (max_val - min_val) * (row / height)
        line = f"{threshold:.1%} │ "
        
        for val in normalized:
            if val >= threshold:
                line += "█"
            else:
                line += " "
        
        print(line)
    
    # Print x-axis
    print(f"     └" + "─" * len(normalized))
    print(f"       0" + " " * (len(normalized) - 10) + f"{len(data)}")
    print(f"       (Total queries: {len(data)})\n")


def print_confusion_matrix(feedbacks: List[RoutingFeedback]):
    """
    Print confusion matrix showing predicted vs actual routes.
    
    Args:
        feedbacks: List of feedback entries
    """
    if not feedbacks:
        print("No feedback data available")
        return
    
    # Count predictions
    matrix = {
        "rag": {"rag": 0, "meta": 0, "chat": 0},
        "meta": {"rag": 0, "meta": 0, "chat": 0},
        "chat": {"rag": 0, "meta": 0, "chat": 0},
    }
    
    for fb in feedbacks:
        matrix[fb.predicted_route][fb.actual_route] += 1
    
    print("\nConfusion Matrix (Predicted → Actual)")
    print("=" * 50)
    print(f"{'':8} │ {'RAG':>8} │ {'META':>8} │ {'CHAT':>8} │")
    print("─" * 50)
    
    for pred in ["rag", "meta", "chat"]:
        counts = matrix[pred]
        total = sum(counts.values())
        
        print(f"{pred.upper():8} │", end="")
        for actual in ["rag", "meta", "chat"]:
            count = counts[actual]
            
            # Color code: green for diagonal (correct), red for off-diagonal
            if pred == actual:
                color = "\033[92m"  # Green
            else:
                color = "\033[91m"  # Red
            
            if total > 0:
                pct = count / total
                print(f" {color}{count:3}{' ':1}({pct:4.0%})\033[0m │", end="")
            else:
                print(f" {count:3} (  0%) │", end="")
        print()
    
    print("=" * 50)
    print(f"{'':8}   {'RAG':^8}   {'META':^8}   {'CHAT':^8}")
    print(f"{'':8}   (Actual Categories)\n")


def print_learning_curve(rl_router):
    """Print learning curve showing accuracy progression."""
    feedbacks = rl_router.get_recent_feedback(n=1000)
    
    if not feedbacks:
        print("No learning data yet. Start chatting to build training data!")
        return
    
    # Calculate rolling accuracy
    window_size = 10
    accuracies = []
    
    for i in range(len(feedbacks)):
        start = max(0, i - window_size + 1)
        window = feedbacks[start:i+1]
        correct = sum(1 for fb in window if fb.correct)
        acc = correct / len(window)
        accuracies.append(acc)
    
    print_ascii_chart(accuracies, title="Learning Curve (10-query rolling average)")


def print_route_performance(rl_router):
    """Print detailed per-route performance."""
    metrics = rl_router.get_metrics()
    route_metrics = metrics["route_accuracy"]
    
    print("\nPer-Category Performance")
    print("=" * 60)
    
    for route_name in ["rag", "meta", "chat"]:
        stats = route_metrics[route_name]
        acc = stats["accuracy"]
        correct = stats["correct"]
        total = stats["total"]
        
        # Visual bar
        bar_length = 30
        filled = int(acc * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        # Color based on performance
        if acc >= 0.8:
            color = "\033[92m"  # Green
        elif acc >= 0.6:
            color = "\033[93m"  # Yellow
        else:
            color = "\033[91m"  # Red
        
        print(f"{route_name.upper():6} │ {color}{bar}\033[0m {acc:5.1%} "
              f"({correct}/{total})")
    
    print("=" * 60 + "\n")


def print_weight_evolution(rl_router):
    """Show how confidence weights have evolved."""
    metrics = rl_router.get_metrics()
    weights = metrics["confidence_weights"]
    
    print("\nLearned Confidence Weight Adjustments")
    print("=" * 60)
    print("(Weights > 1.0 = boosted confidence, < 1.0 = reduced)")
    print()
    
    for route_name in ["rag", "meta", "chat"]:
        weight = weights[route_name]
        
        # Visual representation
        baseline = 1.0
        diff = weight - baseline
        
        if diff > 0:
            # Boosted
            color = "\033[92m"  # Green
            bar_length = int(diff * 20)
            bar = "+" + ">" * bar_length
            indicator = "↑ BOOSTED"
        elif diff < 0:
            # Reduced
            color = "\033[91m"  # Red
            bar_length = int(abs(diff) * 20)
            bar = "<" * bar_length + "-"
            indicator = "↓ REDUCED"
        else:
            color = "\033[93m"  # Yellow
            bar = "="
            indicator = "→ NEUTRAL"
        
        print(f"{route_name.upper():6} │ {weight:.3f} {color}{bar:25}{indicator}\033[0m")
    
    print("=" * 60 + "\n")


def print_recent_mistakes(rl_router, n: int = 5):
    """Show recent incorrect predictions for analysis."""
    feedbacks = rl_router.get_recent_feedback(n=100)
    mistakes = [fb for fb in feedbacks if not fb.correct][-n:]
    
    if not mistakes:
        print("\n🎉 No recent mistakes! Model is performing well.\n")
        return
    
    print(f"\nRecent Mistakes (Last {len(mistakes)})")
    print("=" * 70)
    
    for i, fb in enumerate(mistakes, 1):
        pred_conf = fb.confidence_scores[fb.predicted_route]
        actual_conf = fb.confidence_scores[fb.actual_route]
        
        print(f"\n{i}. Query: \"{fb.query[:50]}{'...' if len(fb.query) > 50 else ''}\"")
        print(f"   Predicted: \033[91m{fb.predicted_route.upper()}\033[0m "
              f"({pred_conf:.1%} confidence)")
        print(f"   Actual:    \033[92m{fb.actual_route.upper()}\033[0m "
              f"({actual_conf:.1%} confidence)")
        print(f"   Penalty:   {fb.reward:.2f}")
    
    print("=" * 70 + "\n")


def main():
    """Main dashboard display."""
    print("\n" + "=" * 70)
    print("  CORTEX RL Router - Learning Dashboard")
    print("=" * 70 + "\n")
    
    # Load RL router
    rl_router = get_rl_router()
    metrics = rl_router.get_metrics()
    
    if metrics["total_predictions"] == 0:
        print("No training data yet. Start using the RL CLI to collect data!")
        print("\nRun: python -m cortex.rl_cli\n")
        return
    
    # Overall statistics
    print(f"Total Predictions: {metrics['total_predictions']}")
    print(f"Overall Accuracy:  {metrics['overall_accuracy']:.1%}")
    print(f"Correct: {metrics['correct_predictions']}, "
          f"Incorrect: {metrics['total_predictions'] - metrics['correct_predictions']}")
    
    # Learning curve
    print_learning_curve(rl_router)
    
    # Per-route performance
    print_route_performance(rl_router)
    
    # Confusion matrix
    feedbacks = rl_router.get_recent_feedback(n=1000)
    print_confusion_matrix(feedbacks)
    
    # Weight evolution
    print_weight_evolution(rl_router)
    
    # Recent mistakes
    print_recent_mistakes(rl_router, n=5)
    
    # Recommendations
    print("Recommendations")
    print("=" * 70)
    
    acc = metrics['overall_accuracy']
    if acc < 0.5:
        print("Model is still learning. Keep providing feedback!")
    elif acc < 0.7:
        print("Model is improving. Continue training on diverse queries.")
    elif acc < 0.85:
        print("Model is performing well. Focus on edge cases.")
    else:
        print("Excellent performance! Model has learned effectively.")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()