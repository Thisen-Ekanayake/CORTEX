# RL Router Documentation

## Overview

The `rl_router.py` module implements a Reinforcement Learning-based query routing system for the CORTEX platform. It wraps around a base TF-IDF router and learns from user feedback to improve routing accuracy over time.

## Purpose

This module serves as an adaptive layer that:
- Routes user queries to appropriate handlers (RAG, META, or CHAT)
- Learns from correct and incorrect routing decisions
- Adjusts confidence weights dynamically based on accumulated feedback
- Maintains performance metrics and feedback history
- Persists learning across sessions

## Architecture

### Core Components

#### 1. `RoutingFeedback` (Dataclass)
Represents a single routing decision and its outcome.

**Fields:**
- `query` (str): The user's input query
- `predicted_route` (str): Route predicted by the model
- `actual_route` (str): Correct route (ground truth from user)
- `confidence_scores` (Dict[str, float]): Confidence for each route
- `correct` (bool): Whether prediction matched actual route
- `timestamp` (str): ISO format timestamp
- `reward` (float): Calculated reward/penalty for this prediction

#### 2. `RLRouter` (Main Class)
The reinforcement learning router that wraps the base router.

**Key Attributes:**
- `base_router`: Underlying TF-IDF router
- `feedback_history`: Deque storing recent feedback (limited by memory_size)
- `confidence_weights`: Learned multipliers for each route
- `route_accuracy`: Per-route performance tracking
- `learning_rate`: Speed of weight adjustments (0-1)

## Installation & Setup

```python
from cortex.rl_router import get_rl_router

# Get the global RL router instance
router = get_rl_router(
    feedback_dir="rl_feedback",  # Where to store feedback
    learning_rate=0.1             # How fast to learn
)
```

## Usage

### Basic Routing

```python
# Route a query
route, confidence_scores = router.route_query("What is machine learning?")
print(f"Route: {route}")
print(f"Confidence: {confidence_scores}")
```

### Recording Feedback

```python
from cortex.router import Route

# After user confirms or corrects the route
feedback = router.record_feedback(
    query="What is machine learning?",
    predicted_route=Route.CHAT,
    actual_route=Route.RAG,  # User said it should be RAG
    confidence_scores=confidence_scores
)

print(f"Reward: {feedback.reward}")
print(f"Correct: {feedback.correct}")
```

### Checking Performance

```python
# Get current metrics
metrics = router.get_metrics()
print(f"Overall accuracy: {metrics['overall_accuracy']:.2%}")
print(f"Total predictions: {metrics['total_predictions']}")
print(f"Confidence weights: {metrics['confidence_weights']}")

# Get recent feedback
recent = router.get_recent_feedback(n=5)
for fb in recent:
    print(f"{fb.query}: {fb.predicted_route} -> {fb.actual_route}")
```

### Resetting Learning

```python
# Clear all learned weights and metrics
router.reset_learning()
```

## Hyperparameters

### Learning Rate (`learning_rate`)

**Default:** 0.1  
**Range:** 0.0 - 1.0  
**Purpose:** Controls how quickly the model adapts to feedback

**Tuning Guidelines:**
- **Low (0.01 - 0.05)**: Slow, stable learning. Use when:
  - You have noisy feedback
  - You want to avoid overfitting to recent examples
  - The base router is already fairly accurate
  
- **Medium (0.1 - 0.3)**: Balanced learning. Use when:
  - Feedback quality is moderate
  - You want reasonable adaptation speed
  - **This is the recommended default**
  
- **High (0.4 - 1.0)**: Fast learning. Use when:
  - You have high-quality, reliable feedback
  - You need quick adaptation to new patterns
  - The base router is performing poorly

**Example:**
```python
# Conservative learning
slow_router = get_rl_router(learning_rate=0.05)

# Aggressive learning
fast_router = get_rl_router(learning_rate=0.5)
```

### Memory Size (`memory_size`)

**Default:** 1000  
**Range:** 100 - 10000+  
**Purpose:** Maximum number of recent feedback entries kept in memory

**Tuning Guidelines:**
- **Small (100 - 500)**: Use when:
  - Memory is constrained
  - You only care about recent patterns
  - Quick iteration cycles
  
- **Medium (1000 - 2000)**: Use when:
  - Balanced memory usage
  - **Recommended default**
  
- **Large (5000+)**: Use when:
  - You want long-term pattern recognition
  - Memory is not a concern
  - You have diverse query patterns

**Example:**
```python
from cortex.router import get_router

base_router = get_router()
router = RLRouter(
    base_router=base_router,
    memory_size=5000  # Keep more history
)
```

### Confidence Weight Bounds

**Hard-coded bounds:**
- Minimum: 0.5 (prevents complete deactivation of a route)
- Maximum: 2.0 (prevents extreme over-weighting)

These bounds prevent the model from becoming too confident or dismissive of any particular route.

## Reward Function

The reward system is designed to encourage both correct predictions and high confidence in those predictions.

### Correct Predictions
```
reward = 1.0 + (confidence_in_correct_route × 0.5)
```
Range: 1.0 to 1.5

### Incorrect Predictions
```
reward = -1.0 - (confidence_in_wrong_route × 0.5)
```
Range: -1.5 to -1.0

**Rationale:**
- Base reward/penalty of ±1.0 for correctness
- Bonus/penalty based on confidence encourages calibrated predictions
- Being confidently wrong is penalized more heavily
- Being confidently correct is rewarded more

## Learning Algorithm

The system uses a simplified Q-learning approach:

### For Correct Predictions:
```python
weight[correct_route] = min(2.0, 
    current_weight + learning_rate × reward
)
```

### For Incorrect Predictions:
```python
# Penalize wrong route
weight[predicted_route] = max(0.5,
    current_weight + learning_rate × reward  # reward is negative
)

# Boost correct route (smaller adjustment)
weight[actual_route] = min(2.0,
    current_weight + learning_rate × |reward| × 0.5
)
```

This approach:
- Increases weights for successful routes
- Decreases weights for failed predictions
- Provides corrective boost to the route that should have been chosen

## File Storage

### Feedback Storage (`feedback.jsonl`)
- One JSON object per line
- Contains all RoutingFeedback entries
- Append-only for efficiency
- Can grow large over time (consider rotation)

**Format:**
```json
{"query": "...", "predicted_route": "rag", "actual_route": "meta", ...}
{"query": "...", "predicted_route": "chat", "actual_route": "chat", ...}
```

### Metrics Storage (`metrics.json`)
- Single JSON file
- Overwritten on each update
- Contains aggregated performance data

**Format:**
```json
{
  "overall_accuracy": 0.85,
  "total_predictions": 100,
  "correct_predictions": 85,
  "route_accuracy": {
    "rag": {"accuracy": 0.90, "total": 50, "correct": 45},
    "meta": {"accuracy": 0.80, "total": 30, "correct": 24},
    "chat": {"accuracy": 0.80, "total": 20, "correct": 16}
  },
  "confidence_weights": {
    "rag": 1.2,
    "meta": 0.9,
    "chat": 1.0
  },
  "recent_feedbacks": 100
}
```

## Alternative Approaches

### 1. Multi-Armed Bandit (MAB)
Instead of adjusting confidence weights, treat each route as an "arm" and use Thompson Sampling or UCB.

**Pros:**
- Theoretically grounded exploration-exploitation tradeoff
- Simpler than full RL
- Good for stationary problems

**Cons:**
- Doesn't leverage base router's text understanding
- Ignores query content
- May be slower to converge

**Implementation sketch:**
```python
# Thompson Sampling approach
class MABRouter:
    def __init__(self):
        self.route_successes = {route: 1 for route in routes}
        self.route_failures = {route: 1 for route in routes}
    
    def select_route(self):
        samples = {
            route: np.random.beta(
                self.route_successes[route],
                self.route_failures[route]
            )
            for route in routes
        }
        return max(samples.items(), key=lambda x: x[1])[0]
```

### 2. Contextual Bandits
Extension of MAB that considers query features.

**Pros:**
- Balances exploration and exploitation
- Uses query context
- Well-studied algorithms (LinUCB, etc.)

**Cons:**
- More complex to implement
- Requires feature engineering
- May need more data to be effective

### 3. Deep Q-Learning (DQN)
Use a neural network to learn Q-values for state-action pairs.

**Pros:**
- Can learn complex patterns
- Handles high-dimensional inputs
- Potentially higher accuracy ceiling

**Cons:**
- Requires significantly more data
- Computationally expensive
- Risk of overfitting
- Needs careful hyperparameter tuning (replay buffer, target network, etc.)

**When to use:** Only if you have 10,000+ labeled examples and computational resources.

### 4. Ensemble Approach
Combine multiple routing strategies and vote.

**Pros:**
- Robust to individual model failures
- Can incorporate diverse signals

**Cons:**
- Higher computational cost
- More complex to maintain

**Example:**
```python
class EnsembleRouter:
    def route_query(self, query):
        votes = []
        votes.append(tfidf_router.route_query(query))
        votes.append(keyword_router.route_query(query))
        votes.append(rl_router.route_query(query))
        return most_common(votes)
```

### 5. Active Learning
Intelligently select which queries to ask users for feedback on.

**Pros:**
- More efficient use of user time
- Focus on uncertain cases
- Faster improvement

**Cons:**
- Additional complexity
- May miss edge cases

**Implementation:**
```python
def should_request_feedback(confidence_scores):
    max_confidence = max(confidence_scores.values())
    # Request feedback when uncertain
    return max_confidence < 0.7
```

## Best Practices

### 1. Monitoring
Regularly check metrics to ensure the model is improving:
```python
# Log metrics periodically
metrics = router.get_metrics()
logger.info(f"Accuracy: {metrics['overall_accuracy']:.2%}")

# Alert on degradation
if metrics['overall_accuracy'] < 0.7:
    alert("Router performance degraded!")
```

### 2. Feedback Quality
Ensure feedback is accurate:
- Validate user corrections
- Consider confidence thresholds before accepting feedback
- Monitor for adversarial or noisy feedback

### 3. Cold Start Problem
The router needs time to learn. Consider:
- Starting with a well-tuned base router
- Seeding with synthetic feedback if available
- Using higher learning rate initially, then decreasing

### 4. Periodic Retraining
The base TF-IDF router doesn't update. Consider:
- Periodically retraining the base router with new examples
- Using feedback data to create training examples
- Implementing a pipeline: feedback → training data → retrained base router

### 5. A/B Testing
Test changes carefully:
```python
# Route 50% of traffic to new RL router
if random.random() < 0.5:
    route = rl_router.route_query(query)
else:
    route = base_router.route_query(query)
```

## Performance Optimization

### Memory Management
```python
# Reduce memory size for production
router = get_rl_router(memory_size=500)

# Archive old feedback periodically
def archive_feedback():
    import shutil
    timestamp = datetime.now().strftime("%Y%m%d")
    shutil.copy(
        "rl_feedback/feedback.jsonl",
        f"archives/feedback_{timestamp}.jsonl"
    )
    # Clear current file
    open("rl_feedback/feedback.jsonl", 'w').close()
```

### Batch Updates
For high-throughput scenarios:
```python
class BatchRLRouter(RLRouter):
    def __init__(self, *args, batch_size=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending_feedbacks = []
        self.batch_size = batch_size
    
    def record_feedback(self, *args, **kwargs):
        feedback = super().record_feedback(*args, **kwargs)
        self.pending_feedbacks.append(feedback)
        
        if len(self.pending_feedbacks) >= self.batch_size:
            self._batch_learn()
        
        return feedback
    
    def _batch_learn(self):
        # Process all pending feedbacks at once
        for feedback in self.pending_feedbacks:
            self._learn_from_feedback(feedback)
        self.pending_feedbacks.clear()
        self._save_metrics()
```

## Troubleshooting

### Low Accuracy
**Symptoms:** Overall accuracy < 70%

**Solutions:**
- Increase learning rate to adapt faster
- Check if base router is performing poorly
- Verify feedback quality
- Consider retraining base router

### Unstable Performance
**Symptoms:** Accuracy fluctuates wildly

**Solutions:**
- Decrease learning rate for more stability
- Increase memory size to smooth over noise
- Implement momentum or moving averages

### One Route Dominates
**Symptoms:** Most queries routed to single route

**Solutions:**
- Check if confidence weight bounds are too permissive
- Verify feedback distribution is balanced
- Consider exploration bonus (epsilon-greedy)

### Slow Learning
**Symptoms:** Accuracy improves very slowly

**Solutions:**
- Increase learning rate
- Reduce memory size (focus on recent patterns)
- Check if feedback signal is too weak

## Future Enhancements

1. **Exploration Strategy**: Add epsilon-greedy or Boltzmann exploration
2. **Feature Engineering**: Extract better features from queries for learning
3. **Temporal Patterns**: Learn time-of-day or seasonal routing preferences
4. **User Personalization**: Maintain per-user routing preferences
5. **Confidence Calibration**: Ensure predicted confidences match actual accuracy
6. **Meta-Learning**: Learn how to set hyperparameters automatically

## References

- Q-Learning: Watkins, C.J.C.H. (1989). "Learning from Delayed Rewards"
- Multi-Armed Bandits: Lai, T.L. and Robbins, H. (1985). "Asymptotically efficient adaptive allocation rules"
- Contextual Bandits: Li, L. et al. (2010). "A Contextual-Bandit Approach to Personalized News Article Recommendation"