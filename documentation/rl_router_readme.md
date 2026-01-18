# CORTEX Reinforcement Learning Router

An interactive learning system that improves query routing accuracy through user feedback.

## 🎯 Overview

The RL Router learns from your corrections to improve routing decisions over time. Instead of relying solely on a pre-trained classifier, it adapts to your specific use patterns.

### How It Works

1. **User inputs query** → Model predicts category (RAG/META/CHAT)
2. **Shows prediction** → Displays confidence scores for each category
3. **User selects actual category** → Routes to the correct handler
4. **Model learns** → Adjusts confidence weights based on correctness
5. **Repeat** → Accuracy improves over time

### Key Features

- ✅ **Interactive Learning**: Real-time feedback loop
- 📊 **Performance Tracking**: Detailed metrics and statistics
- 🎯 **Adaptive Weights**: Learns category-specific confidence adjustments
- 💾 **Persistent Storage**: Saves learning progress across sessions
- 📈 **Progress Visualization**: Dashboard to monitor improvement

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Already included in your existing requirements
# No additional packages needed!
```

### 2. Run Interactive Training CLI

```bash
python -m cortex.rl_cli
```

### 3. Chat and Provide Feedback

```
> What documents do you have about sales?

┌─ Model Prediction ─────────────────────────────────────────┐
│ ► Predicted: RAG      (Confidence: 78.5%)
│
│ Confidence Breakdown:
│   ► RAG    ████████████████░░░░ 78.5%
│     META   ██░░░░░░░░░░░░░░░░░░ 12.3%
│     CHAT   ██░░░░░░░░░░░░░░░░░░  9.2%
└────────────────────────────────────────────────────────────┘

Select the ACTUAL category for this query:
  [1] RAG     - Retrieve from documents
  [2] META    - About the system
  [3] CHAT    - General conversation
  [0] Cancel

Your choice [1-3, 0 to cancel]: 1

┌─ Learning Feedback ────────────────────────────────────────┐
│ ✓ CORRECT 🎉
│
│ Predicted: RAG
│ Actual:    RAG
│
│ Reward: +1.39
│ Model confidence weights adjusted (reinforced)
└────────────────────────────────────────────────────────────┘

[RAG] Here are the sales documents I found...
```

### 4. View Progress Dashboard

```bash
python -m cortex.rl_dashboard
```

Output:
```
Learning Curve (10-query rolling average)
──────────────────────────────────────────────────────────
100.0% │ ███████████████████████████████████████
 90.0% │ ██████████████████████████████░░░░░░░░░
 80.0% │ ███████████████████░░░░░░░░░░░░░░░░░░░░
 70.0% │ ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░
 60.0% │ █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
       └────────────────────────────────────────
       0                                      50
       (Total queries: 50)
```

## 📊 Architecture

### Components

1. **`rl_router.py`**: Core RL logic
   - Maintains confidence weight adjustments
   - Records feedback and calculates rewards
   - Saves/loads learning state

2. **`rl_cli.py`**: Interactive training interface
   - Shows model predictions with confidence
   - Collects user feedback
   - Executes queries with correct routing

3. **`rl_dashboard.py`**: Visualization tools
   - Learning curves
   - Confusion matrices
   - Per-category performance
   - Weight evolution tracking

### Learning Algorithm

The system uses a **Q-learning inspired approach**:

```python
if correct:
    # Reinforce correct predictions
    weight[actual] += learning_rate * (1.0 + confidence_bonus)
else:
    # Penalize incorrect predictions
    weight[predicted] -= learning_rate * (1.0 + confidence_penalty)
    # Boost correct category
    weight[actual] += learning_rate * 0.5
```

**Rewards**:
- Correct prediction: `+1.0 + (confidence * 0.5)` (max +1.5)
- Incorrect prediction: `-1.0 - (confidence * 0.5)` (min -1.5)

**Confidence Weights**:
- Initialized at `1.0` (neutral)
- Range: `[0.5, 2.0]`
- Applied as multipliers to base classifier scores

## 🎮 Usage Guide

### Available Commands

In the CLI (`rl_cli.py`):

| Command | Description |
|---------|-------------|
| `stats` | Show current performance statistics |
| `reset` | Reset all learning (back to defaults) |
| `help` | Show help message |
| `exit`/`quit` | Save and exit |

### Training Tips

1. **Start with Clear Cases**: Begin with obvious RAG/META/CHAT queries
2. **Diversify**: Include edge cases and ambiguous queries
3. **Be Consistent**: Try to categorize similar queries the same way
4. **Monitor Progress**: Check stats every ~10-20 queries
5. **Target 50+ Queries**: Model starts showing improvement around 30-50 samples

### Category Selection Guide

**RAG** - Select when query:
- Asks about specific document content
- Requires information retrieval
- References "documents", "files", "reports"

**META** - Select when query:
- Asks about the system itself
- Questions like "who are you?", "what can you do?"
- Requests system information

**CHAT** - Select when query:
- General conversation
- Doesn't need documents
- Small talk, greetings, general questions

## 📈 Performance Metrics

### Overall Accuracy
```
Overall Accuracy: 85.5% (77/90 correct)
```

### Per-Category Performance
```
RAG    │ ██████████████████████████░░░░ 87.5% (28/32)
META   │ ████████████████████████░░░░░░ 80.0% (16/20)
CHAT   │ ██████████████████████████████ 86.8% (33/38)
```

### Confusion Matrix
Shows predicted vs actual distribution to identify systematic errors.

### Learned Weights
```
RAG    │ 1.154 +>>>>  ↑ BOOSTED
META   │ 0.892 <<-    ↓ REDUCED  
CHAT   │ 1.067 +>     ↑ BOOSTED
```

## 🔧 Configuration

### Learning Rate

Adjust in `rl_router.py` or when initializing:

```python
rl_router = get_rl_router(learning_rate=0.15)
```

- **Low (0.05-0.1)**: Slow, stable learning
- **Medium (0.1-0.2)**: Balanced (recommended)
- **High (0.2-0.3)**: Fast learning, may overshoot

### Memory Size

Maximum feedback entries to keep in memory:

```python
RLRouter(base_router, memory_size=1000)
```

### Confidence Threshold

Minimum confidence to trust predictions (in `router.execute`):

```python
execute(query, confidence_threshold=0.5)
```

## 💾 Data Storage

### Files Created

```
rl_feedback/
├── feedback.jsonl          # All feedback entries (append-only)
├── metrics.json           # Current performance metrics
└── (auto-created on first use)
```

### Feedback Entry Format

```json
{
  "query": "What's in the Q3 report?",
  "predicted_route": "rag",
  "actual_route": "rag",
  "confidence_scores": {
    "rag": 0.825,
    "meta": 0.102,
    "chat": 0.073
  },
  "correct": true,
  "timestamp": "2026-01-18T14:30:22.123456",
  "reward": 1.4125
}
```

## 🧪 Testing & Debugging

### Test Routing Predictions

```python
from cortex.rl_router import get_rl_router

rl_router = get_rl_router()
route, scores = rl_router.route_query("What can you do?")

print(f"Route: {route.value}")
print(f"Scores: {scores}")
```

### Simulate Feedback

```python
from cortex.router import Route

# Simulate correct prediction
feedback = rl_router.record_feedback(
    query="Tell me about the sales report",
    predicted_route=Route.RAG,
    actual_route=Route.RAG,
    confidence_scores={"rag": 0.8, "meta": 0.1, "chat": 0.1}
)

print(f"Reward: {feedback.reward}")
```

### Reset Learning

```python
rl_router.reset_learning()  # Back to default weights
```

## 🚨 Troubleshooting

### Model not improving?
- Ensure you've provided 30+ diverse training samples
- Check if you're being consistent with categorizations
- Try increasing learning rate slightly

### Weights seem stuck?
- Check `rl_feedback/metrics.json` for current weights
- Verify feedback is being saved to `feedback.jsonl`
- Consider resetting and retraining

### Performance regressed?
- Learning rate may be too high
- You may need more training data
- Check recent mistakes in dashboard

## 🎯 Best Practices

1. **Consistent Labeling**: Be consistent in how you categorize queries
2. **Balanced Training**: Try to provide examples from all three categories
3. **Regular Monitoring**: Check dashboard every 20-30 queries
4. **Edge Cases**: Include ambiguous queries to improve robustness
5. **Long-term Learning**: The more data, the better the model performs

## 📚 Integration with Existing Code

The RL router is a drop-in replacement for the base router:

```python
# Old way (base router)
from cortex.router import route_query, execute

# New way (RL router)
from cortex.rl_router import get_rl_router

rl_router = get_rl_router()
route, scores = rl_router.route_query("Your query")
```

## 🔮 Future Enhancements

- [ ] Add automated testing with labeled dataset
- [ ] Implement batch learning mode
- [ ] Add export/import for learned weights
- [ ] Create web-based training interface
- [ ] Support for custom reward functions
- [ ] Multi-user learning aggregation

<details>
<summary> Example Integration </summary>
from cortex.rl_router import get_rl_router
from cortex.router import Route, execute
from cortex.query import run_rag, run_meta, run_chat


# ============================================================================
# Example 1: Simple Prediction
# ============================================================================

def example_basic_prediction():
    """Show basic routing prediction"""
    print("=" * 60)
    print("Example 1: Basic Prediction")
    print("=" * 60)
    
    rl_router = get_rl_router()
    
    # Test queries
    queries = [
        "What's in the quarterly sales report?",
        "Who are you?",
        "Hello! How are you today?",
    ]
    
    for query in queries:
        route, scores = rl_router.route_query(query)
        
        print(f"\nQuery: {query}")
        print(f"Predicted Route: {route.value}")
        print(f"Confidence Scores:")
        for category, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {category:6}: {score:.1%}")
    
    print()


# ============================================================================
# Example 2: With User Feedback
# ============================================================================

def example_with_feedback():
    """Show how to record feedback"""
    print("=" * 60)
    print("Example 2: Recording Feedback")
    print("=" * 60)
    
    rl_router = get_rl_router()
    
    # Simulate a query with feedback
    query = "Show me documents about machine learning"
    
    # Get prediction
    predicted_route, scores = rl_router.route_query(query)
    print(f"\nQuery: {query}")
    print(f"Model predicted: {predicted_route.value} ({scores[predicted_route.value]:.1%})")
    
    # Simulate user selecting actual category
    actual_route = Route.RAG  # User confirms RAG is correct
    
    # Record feedback
    feedback = rl_router.record_feedback(
        query=query,
        predicted_route=predicted_route,
        actual_route=actual_route,
        confidence_scores=scores
    )
    
    print(f"\nFeedback recorded:")
    print(f"  Correct: {feedback.correct}")
    print(f"  Reward: {feedback.reward:.2f}")
    
    # Show updated metrics
    metrics = rl_router.get_metrics()
    print(f"\nCurrent Accuracy: {metrics['overall_accuracy']:.1%}")
    print()


# ============================================================================
# Example 3: Execute with Best Route
# ============================================================================

def example_execute_with_routing():
    """Show execution with RL routing"""
    print("=" * 60)
    print("Example 3: Execute with RL Routing")
    print("=" * 60)
    
    rl_router = get_rl_router()
    
    query = "What can CORTEX do?"
    
    # Get route prediction
    route, scores = rl_router.route_query(query)
    
    print(f"\nQuery: {query}")
    print(f"Routing to: {route.value} ({scores[route.value]:.1%})")
    
    # Execute based on route
    if route == Route.RAG:
        result = run_rag(query)
    elif route == Route.META:
        result = run_meta(query)
    else:
        result = run_chat(query)
    
    print(f"\nResult preview: {result[:100] if result else 'No result'}...")
    print()


# ============================================================================
# Example 4: Automated Testing Mode
# ============================================================================

def example_automated_testing():
    """Test with pre-labeled queries"""
    print("=" * 60)
    print("Example 4: Automated Testing with Labeled Data")
    print("=" * 60)
    
    rl_router = get_rl_router()
    
    # Pre-labeled test set
    test_set = [
        ("What's in the Q3 report?", Route.RAG),
        ("Who created you?", Route.META),
        ("Hello there!", Route.CHAT),
        ("Show me the sales data", Route.RAG),
        ("What are your capabilities?", Route.META),
        ("How's the weather?", Route.CHAT),
    ]
    
    correct = 0
    total = len(test_set)
    
    print(f"\nTesting {total} labeled queries...\n")
    
    for query, expected_route in test_set:
        predicted_route, scores = rl_router.route_query(query)
        
        is_correct = (predicted_route == expected_route)
        if is_correct:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"{status} {query[:40]:40} | "
              f"Expected: {expected_route.value:5} | "
              f"Got: {predicted_route.value:5} ({scores[predicted_route.value]:.0%})")
        
        # Record feedback (automated training)
        rl_router.record_feedback(
            query=query,
            predicted_route=predicted_route,
            actual_route=expected_route,
            confidence_scores=scores
        )
    
    accuracy = correct / total
    print(f"\nTest Accuracy: {accuracy:.1%} ({correct}/{total})")
    print()


# ============================================================================
# Example 5: Progressive Learning Simulation
# ============================================================================

def example_progressive_learning():
    """Simulate learning over time"""
    print("=" * 60)
    print("Example 5: Progressive Learning Simulation")
    print("=" * 60)
    
    rl_router = get_rl_router()
    
    # Training data (query, label)
    training_data = [
        ("Find the marketing documents", Route.RAG),
        ("What is your purpose?", Route.META),
        ("Hi!", Route.CHAT),
        ("Search for budget reports", Route.RAG),
        ("Tell me about yourself", Route.META),
        ("Good morning", Route.CHAT),
        ("Show me project files", Route.RAG),
        ("How do you work?", Route.META),
        ("Thanks!", Route.CHAT),
        ("Where are the HR documents?", Route.RAG),
    ]
    
    print("\nTraining with 10 queries...\n")
    
    accuracies = []
    
    for i, (query, actual_route) in enumerate(training_data, 1):
        # Predict
        predicted_route, scores = rl_router.route_query(query)
        
        # Check correctness
        correct = (predicted_route == actual_route)
        
        # Record feedback
        rl_router.record_feedback(
            query=query,
            predicted_route=predicted_route,
            actual_route=actual_route,
            confidence_scores=scores
        )
        
        # Track accuracy
        metrics = rl_router.get_metrics()
        acc = metrics['overall_accuracy']
        accuracies.append(acc)
        
        status = "✓" if correct else "✗"
        print(f"{i:2}. {status} Accuracy: {acc:.1%} | Query: {query[:35]}")
    
    print(f"\nFinal Accuracy: {accuracies[-1]:.1%}")
    print(f"Improvement: {accuracies[-1] - accuracies[0]:+.1%}")
    print()


# ============================================================================
# Example 6: View Learning Metrics
# ============================================================================

def example_view_metrics():
    """Display current metrics"""
    print("=" * 60)
    print("Example 6: View Learning Metrics")
    print("=" * 60)
    
    rl_router = get_rl_router()
    metrics = rl_router.get_metrics()
    
    print(f"\nOverall Performance:")
    print(f"  Total Predictions: {metrics['total_predictions']}")
    print(f"  Accuracy: {metrics['overall_accuracy']:.1%}")
    print(f"  Correct: {metrics['correct_predictions']}")
    
    print(f"\nPer-Category Accuracy:")
    for route, stats in metrics['route_accuracy'].items():
        if stats['total'] > 0:
            acc = stats['accuracy']
            print(f"  {route.upper():6}: {acc:.1%} ({stats['correct']}/{stats['total']})")
        else:
            print(f"  {route.upper():6}: No data yet")
    
    print(f"\nLearned Weights:")
    for route, weight in metrics['confidence_weights'].items():
        indicator = "↑" if weight > 1.0 else "↓" if weight < 1.0 else "→"
        print(f"  {route.upper():6}: {weight:.3f} {indicator}")
    
    print()


# ============================================================================
# Example 7: Integration with Existing CLI
# ============================================================================

def example_hybrid_mode():
    """Show hybrid approach: automatic routing with optional feedback"""
    print("=" * 60)
    print("Example 7: Hybrid Mode (Auto + Optional Feedback)")
    print("=" * 60)
    
    rl_router = get_rl_router()
    
    query = "What documents do you have?"
    
    # Automatic routing
    predicted_route, scores = rl_router.route_query(query)
    confidence = scores[predicted_route.value]
    
    print(f"\nQuery: {query}")
    print(f"Auto-routed to: {predicted_route.value} ({confidence:.1%})")
    
    # If confidence is low, ask user for feedback
    if confidence < 0.7:
        print(f"\n⚠ Low confidence! Was '{predicted_route.value}' correct?")
        print("  This would show a feedback prompt in real CLI")
        
        # Simulate user correction
        actual_route = Route.RAG  # User corrects
        
        if actual_route != predicted_route:
            print(f"  User corrected to: {actual_route.value}")
            rl_router.record_feedback(
                query=query,
                predicted_route=predicted_route,
                actual_route=actual_route,
                confidence_scores=scores
            )
            print("  ✓ Model updated from feedback")
    else:
        print("✓ High confidence - proceeding automatically")
    
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print(" CORTEX RL Router - Integration Examples")
    print("=" * 60 + "\n")
    
    examples = [
        ("Basic Prediction", example_basic_prediction),
        ("With Feedback", example_with_feedback),
        ("Execute with Routing", example_execute_with_routing),
        ("Automated Testing", example_automated_testing),
        ("Progressive Learning", example_progressive_learning),
        ("View Metrics", example_view_metrics),
        ("Hybrid Mode", example_hybrid_mode),
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        print(f"\n[Example {i}/{len(examples)}] {name}\n")
        try:
            func()
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        
        if i < len(examples):
            input("Press Enter for next example...")
    
    print("\n" + "=" * 60)
    print(" Examples Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
<details>