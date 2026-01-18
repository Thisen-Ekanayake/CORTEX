# RL Dashboard Documentation

## Overview

The `rl_dashboard.py` module provides a comprehensive command-line visualization dashboard for monitoring and analyzing the Reinforcement Learning Router's performance. It displays learning curves, confusion matrices, per-route statistics, and weight evolution through ASCII-based charts and colored terminal output.

## Purpose

This dashboard serves as a monitoring and debugging tool that:
- Visualizes learning progress over time
- Identifies routing patterns and common mistakes
- Tracks confidence weight adjustments
- Provides actionable recommendations for improvement
- Helps diagnose model performance issues
- Offers insights into which routes are performing well or poorly

## Features

### 1. Learning Curve Visualization
Shows accuracy progression over time using a rolling window average.

### 2. Confusion Matrix
Displays predicted vs. actual route classifications with color-coded accuracy indicators.

### 3. Per-Route Performance
Individual accuracy metrics for RAG, META, and CHAT routes with visual progress bars.

### 4. Weight Evolution
Shows how confidence weights have been adjusted through learning.

### 5. Recent Mistakes Analysis
Lists the most recent incorrect predictions for debugging.

### 6. Performance Recommendations
Automated suggestions based on current accuracy levels.

## Installation & Setup

### Prerequisites

```bash
# Ensure CORTEX router is installed
pip install -r requirements.txt
```

### Required Dependencies

```python
# Standard library only - no external dependencies needed!
import json
from collections import Counter
from typing import List
```

The dashboard uses ANSI color codes for terminal output. Ensure your terminal supports ANSI escape sequences (most modern terminals do).

## Usage

### Basic Usage

```bash
# Run the dashboard
python rl_dashboard.py

# Or as a module
python -m cortex.rl_dashboard
```

### Integration in Code

```python
from cortex.rl_dashboard import (
    print_learning_curve,
    print_confusion_matrix,
    print_route_performance,
    print_weight_evolution,
    print_recent_mistakes
)
from cortex.rl_router import get_rl_router

# Get router
router = get_rl_router()

# Display specific visualizations
print_learning_curve(router)
print_route_performance(router)
print_recent_mistakes(router, n=10)
```

### Automated Monitoring

```python
import schedule
import time

def monitor_performance():
    """Run dashboard check every hour."""
    from cortex.rl_dashboard import main
    main()

# Schedule monitoring
schedule.every(1).hours.do(monitor_performance)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Function Reference

### 1. `print_ascii_chart()`

Renders a time-series chart in ASCII art.

**Signature:**
```python
def print_ascii_chart(
    data: List[float],
    height: int = 10,
    width: int = 50,
    title: str = "Accuracy Over Time"
) -> None
```

**Parameters:**
- `data`: List of values between 0 and 1 (e.g., accuracy percentages)
- `height`: Number of rows in the chart (default: 10)
- `width`: Number of columns in the chart (default: 50)
- `title`: Chart title to display

**Example:**
```python
accuracies = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.82, 0.85]
print_ascii_chart(accuracies, height=8, width=40, title="Training Progress")
```

**Output Example:**
```
Training Progress
──────────────────────────────────────────────────
85.0% │ ▓▓▓▓▓▓▓▓
70.0% │ ▓▓▓▓▓▓░░
55.0% │ ▓░░░░░░░
      └────────────────────
       0              8
       (Total queries: 8)
```

**Tuning:**
- **Increase `height`** for finer granularity (10-20 recommended)
- **Increase `width`** for more data points (50-100 recommended)
- Automatically downsamples if `len(data) > width`

### 2. `print_confusion_matrix()`

Displays predicted vs. actual routes in a confusion matrix format.

**Signature:**
```python
def print_confusion_matrix(feedbacks: List[RoutingFeedback]) -> None
```

**Parameters:**
- `feedbacks`: List of RoutingFeedback objects

**Example:**
```python
feedbacks = router.get_recent_feedback(n=100)
print_confusion_matrix(feedbacks)
```

**Output Example:**
```
Confusion Matrix (Predicted → Actual)
==================================================
         │      RAG │     META │     CHAT │
──────────────────────────────────────────────────
RAG      │  45 (90%) │   3 ( 6%) │   2 ( 4%) │
META     │   5 (17%) │  25 (83%) │   0 ( 0%) │
CHAT     │   0 ( 0%) │   2 (10%) │  18 (90%) │
==================================================
            RAG        META        CHAT
         (Actual Categories)
```

**Interpretation:**
- **Diagonal values (green)**: Correct predictions
- **Off-diagonal values (red)**: Misclassifications
- **Rows**: Predicted route
- **Columns**: Actual route
- **Percentages**: Row-wise percentages (how often predicted X was actually Y)

**Common Patterns:**
- **High diagonal**: Good overall performance
- **One row with low diagonal**: Specific route is being mispredicted
- **One column with scattered reds**: Specific actual route is hard to predict

### 3. `print_learning_curve()`

Shows rolling accuracy over time to visualize learning progress.

**Signature:**
```python
def print_learning_curve(rl_router) -> None
```

**Parameters:**
- `rl_router`: RLRouter instance

**Configuration:**
```python
# Hard-coded parameters (modify in source if needed)
window_size = 10  # Rolling window size
n = 1000          # Number of recent feedbacks to analyze
```

**Example:**
```python
print_learning_curve(router)
```

**Tuning the Rolling Window:**

Modify the source code to adjust:
```python
# In print_learning_curve():
window_size = 20  # Smoother curve (default: 10)
# OR
window_size = 5   # More responsive to recent changes
```

**Window Size Guidelines:**
- **Small (5)**: Responsive, shows every fluctuation
- **Medium (10)**: Balanced (default)
- **Large (20-50)**: Smooth, shows overall trend

### 4. `print_route_performance()`

Displays per-route accuracy with visual progress bars.

**Signature:**
```python
def print_route_performance(rl_router) -> None
```

**Example:**
```python
print_route_performance(router)
```

**Output Example:**
```
Per-Category Performance
============================================================
RAG    │ ██████████████████████████░░░░  90.0% (45/50)
META   │ ████████████████████░░░░░░░░░░  70.0% (21/30)
CHAT   │ ██████████████████████████████  100.0% (20/20)
============================================================
```

**Color Coding:**
- **Green (≥80%)**: Excellent performance
- **Yellow (60-79%)**: Moderate performance
- **Red (<60%)**: Poor performance, needs attention

**Customization:**
Modify `bar_length` in source code:
```python
bar_length = 40  # Longer bars (default: 30)
```

### 5. `print_weight_evolution()`

Shows how learned confidence weights have changed from baseline (1.0).

**Signature:**
```python
def print_weight_evolution(rl_router) -> None
```

**Example:**
```python
print_weight_evolution(router)
```

**Output Example:**
```
Learned Confidence Weight Adjustments
============================================================
(Weights > 1.0 = boosted confidence, < 1.0 = reduced)

RAG    │ 1.245 +>>>>>>>>>>>>          → BOOSTED
META   │ 0.876 <<<<<<<<<<<-           → REDUCED
CHAT   │ 1.000 =                      → NEUTRAL
============================================================
```

**Interpretation:**
- **Weight > 1.0**: Model has increased confidence in this route (successful predictions)
- **Weight < 1.0**: Model has decreased confidence (frequent mistakes)
- **Weight = 1.0**: No adjustment (neutral performance or insufficient data)

**Range:** 0.5 to 2.0 (hard-coded bounds in RLRouter)

### 6. `print_recent_mistakes()`

Lists recent incorrect predictions for debugging and pattern identification.

**Signature:**
```python
def print_recent_mistakes(rl_router, n: int = 5) -> None
```

**Parameters:**
- `rl_router`: RLRouter instance
- `n`: Number of recent mistakes to display (default: 5)

**Example:**
```python
print_recent_mistakes(router, n=10)
```

**Output Example:**
```
Recent Mistakes (Last 5)
======================================================================

1. Query: "What is the capital of France?"
   Predicted: RAG (75.0% confidence)
   Actual:    CHAT (15.0% confidence)
   Penalty:   -1.38

2. Query: "Show me documents about Q3 sales..."
   Predicted: META (60.0% confidence)
   Actual:    RAG (35.0% confidence)
   Penalty:   -1.30
======================================================================
```

**Use Cases:**
- **Pattern Recognition**: Identify types of queries that are frequently mispredicted
- **Edge Cases**: Find ambiguous queries that need attention
- **Feature Engineering**: Discover keywords or patterns to add to the base router
- **Feedback Quality**: Verify that user corrections make sense

### 7. `main()`

Main dashboard orchestrator that displays all metrics.

**Signature:**
```python
def main() -> None
```

**Displays:**
1. Overall statistics (total predictions, accuracy)
2. Learning curve
3. Per-route performance
4. Confusion matrix
5. Weight evolution
6. Recent mistakes
7. Automated recommendations

## Configuration Options

### Display Settings

All display functions use hard-coded defaults. To customize, modify the source:

```python
# Chart dimensions
print_ascii_chart(data, height=15, width=80)  # Larger chart

# Number of mistakes shown
print_recent_mistakes(router, n=10)  # Show more mistakes

# Confusion matrix data size
feedbacks = router.get_recent_feedback(n=500)  # Use less data
print_confusion_matrix(feedbacks)
```

### Terminal Compatibility

**ANSI Color Codes Used:**
- `\033[92m` - Green (success, high performance)
- `\033[91m` - Red (errors, low performance)
- `\033[93m` - Yellow (warnings, moderate performance)
- `\033[0m` - Reset

**Disabling Colors:**

For terminals that don't support ANSI codes:

```python
# Create a no-color version
def strip_ansi(text):
    import re
    return re.sub(r'\033\[[0-9;]+m', '', text)

# Or set environment variable
import os
os.environ['NO_COLOR'] = '1'  # Some terminals respect this
```

### Unicode Characters

The dashboard uses Unicode box-drawing characters:
- `│` (vertical line)
- `─` (horizontal line)
- `└` (bottom-left corner)
- `█` (filled block)
- `░` (light shade)

**Fallback for ASCII-only terminals:**
```python
# Replace in source code:
# │ → |
# ─ → -
# └ → +
# █ → #
# ░ → .
```

## Alternative Visualization Tools

### 1. Web-Based Dashboard (Streamlit)

For a more interactive experience:

**Implementation:**
```python
import streamlit as st
import pandas as pd
import plotly.express as px

def streamlit_dashboard():
    st.title("RL Router Dashboard")
    
    router = get_rl_router()
    metrics = router.get_metrics()
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics['overall_accuracy']:.1%}")
    col2.metric("Total Predictions", metrics['total_predictions'])
    col3.metric("Recent Feedbacks", metrics['recent_feedbacks'])
    
    # Learning curve
    feedbacks = router.get_recent_feedback(n=1000)
    accuracies = calculate_rolling_accuracy(feedbacks)
    
    fig = px.line(y=accuracies, title="Learning Curve")
    st.plotly_chart(fig)
    
    # Confusion matrix heatmap
    confusion_data = build_confusion_matrix(feedbacks)
    fig = px.imshow(confusion_data, text_auto=True, 
                    title="Confusion Matrix")
    st.plotly_chart(fig)

if __name__ == "__main__":
    streamlit_dashboard()
```

**Run:**
```bash
pip install streamlit plotly
streamlit run dashboard_web.py
```

**Pros:**
- Interactive charts
- Better visualizations
- Accessible via browser
- Easier to share

**Cons:**
- Additional dependencies
- Requires web server
- More complex setup

### 2. TensorBoard Integration

For ML practitioners familiar with TensorBoard:

```python
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    def __init__(self, log_dir="runs/rl_router"):
        self.writer = SummaryWriter(log_dir)
        self.step = 0
    
    def log_feedback(self, feedback: RoutingFeedback):
        """Log each feedback event."""
        self.writer.add_scalar(
            'Accuracy/correct',
            float(feedback.correct),
            self.step
        )
        self.writer.add_scalar(
            'Reward/value',
            feedback.reward,
            self.step
        )
        
        # Log confidences
        for route, conf in feedback.confidence_scores.items():
            self.writer.add_scalar(
                f'Confidence/{route}',
                conf,
                self.step
            )
        
        self.step += 1
    
    def log_metrics(self, metrics: dict):
        """Log aggregated metrics."""
        self.writer.add_scalar(
            'Accuracy/overall',
            metrics['overall_accuracy'],
            self.step
        )
        
        for route, stats in metrics['route_accuracy'].items():
            self.writer.add_scalar(
                f'RouteAccuracy/{route}',
                stats['accuracy'],
                self.step
            )

# Usage
logger = TensorBoardLogger()

# In your training loop
feedback = router.record_feedback(...)
logger.log_feedback(feedback)
```

**Run:**
```bash
pip install tensorboard
tensorboard --logdir=runs/rl_router
```

**Pros:**
- Industry-standard tool
- Powerful visualizations
- Good for experiments
- Supports comparison between runs

**Cons:**
- Requires PyTorch
- Learning curve
- Overkill for simple monitoring

### 3. Matplotlib/Seaborn Dashboards

For static report generation:

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_report(router, output_file="rl_report.png"):
    """Generate a comprehensive visual report."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Learning curve
    feedbacks = router.get_recent_feedback(n=1000)
    accuracies = calculate_rolling_accuracy(feedbacks, window=10)
    axes[0, 0].plot(accuracies)
    axes[0, 0].set_title('Learning Curve')
    axes[0, 0].set_xlabel('Query Number')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True)
    
    # Confusion matrix
    confusion = build_confusion_matrix_array(feedbacks)
    sns.heatmap(confusion, annot=True, fmt='d', 
                xticklabels=['RAG', 'META', 'CHAT'],
                yticklabels=['RAG', 'META', 'CHAT'],
                ax=axes[0, 1])
    axes[0, 1].set_title('Confusion Matrix')
    
    # Per-route accuracy
    metrics = router.get_metrics()
    routes = list(metrics['route_accuracy'].keys())
    accuracies = [metrics['route_accuracy'][r]['accuracy'] 
                  for r in routes]
    axes[1, 0].bar(routes, accuracies)
    axes[1, 0].set_title('Per-Route Accuracy')
    axes[1, 0].set_ylim([0, 1])
    
    # Weight evolution
    weights = [metrics['confidence_weights'][r] for r in routes]
    axes[1, 1].bar(routes, weights)
    axes[1, 1].axhline(y=1.0, color='r', linestyle='--', 
                       label='Baseline')
    axes[1, 1].set_title('Confidence Weights')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Report saved to {output_file}")

# Usage
generate_report(router)
```

**Pros:**
- High-quality static images
- Good for reports/papers
- Highly customizable
- Familiar to data scientists

**Cons:**
- Not interactive
- Requires manual refresh
- More code to maintain

### 4. Logging to File + Grafana

For production monitoring:

```python
import logging
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    filename='rl_router_metrics.log',
    format='%(asctime)s %(message)s',
    level=logging.INFO
)

def log_metrics_for_grafana(router):
    """Log metrics in Grafana-compatible format."""
    metrics = router.get_metrics()
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'overall_accuracy': metrics['overall_accuracy'],
        'total_predictions': metrics['total_predictions'],
        'rag_accuracy': metrics['route_accuracy']['rag']['accuracy'],
        'meta_accuracy': metrics['route_accuracy']['meta']['accuracy'],
        'chat_accuracy': metrics['route_accuracy']['chat']['accuracy'],
        'rag_weight': metrics['confidence_weights']['rag'],
        'meta_weight': metrics['confidence_weights']['meta'],
        'chat_weight': metrics['confidence_weights']['chat'],
    }
    
    logging.info(json.dumps(log_entry))

# Schedule periodic logging
import schedule
schedule.every(1).minutes.do(lambda: log_metrics_for_grafana(router))
```

**Configure Grafana to read from logs or push to Prometheus.**

**Pros:**
- Production-ready
- Real-time monitoring
- Alerting capabilities
- Industry standard

**Cons:**
- Complex setup
- Requires infrastructure
- Overkill for development

## Best Practices

### 1. Regular Monitoring

```bash
# Run daily
0 9 * * * cd /path/to/project && python rl_dashboard.py > daily_report.txt

# Or integrate into CI/CD
python rl_dashboard.py && \
  if [ $accuracy -lt 70 ]; then \
    echo "Warning: Model performance degraded!" | mail -s "Alert" team@example.com; \
  fi
```

### 2. Interpretation Guidelines

**Overall Accuracy:**
- **< 50%**: Model is worse than random, check for bugs
- **50-70%**: Learning in progress, continue training
- **70-85%**: Good performance, focus on edge cases
- **> 85%**: Excellent, monitor for overfitting

**Confusion Matrix:**
- Look for systematic patterns (e.g., META always predicted as RAG)
- Focus improvement on most common confusions
- If one route dominates, check data balance

**Weight Evolution:**
- Weights should stabilize after ~100-200 predictions
- Extreme weights (near 0.5 or 2.0) indicate strong patterns
- Neutral weights (near 1.0) suggest base router is already good

### 3. Debug Workflows

**High Mistakes on One Route:**
```python
# Analyze specific route
feedbacks = router.get_recent_feedback(n=1000)
rag_mistakes = [
    fb for fb in feedbacks 
    if fb.actual_route == 'rag' and not fb.correct
]

print(f"RAG mispredicted {len(rag_mistakes)} times")
for mistake in rag_mistakes[:10]:
    print(f"  Query: {mistake.query}")
    print(f"  Predicted as: {mistake.predicted_route}")
```

**Tracking Improvement:**
```python
# Compare two time periods
old_feedbacks = load_feedbacks_from_file("backup_2024_01.jsonl")
new_feedbacks = router.get_recent_feedback(n=1000)

old_acc = calculate_accuracy(old_feedbacks)
new_acc = calculate_accuracy(new_feedbacks)

print(f"Improvement: {(new_acc - old_acc) * 100:.1f} percentage points")
```

### 4. Automated Alerts

```python
def check_performance_and_alert(router, threshold=0.7):
    """Send alert if performance drops."""
    metrics = router.get_metrics()
    
    if metrics['overall_accuracy'] < threshold:
        send_alert(
            f"⚠️ Router accuracy dropped to "
            f"{metrics['overall_accuracy']:.1%}"
        )
    
    # Check individual routes
    for route, stats in metrics['route_accuracy'].items():
        if stats['total'] > 10 and stats['accuracy'] < 0.5:
            send_alert(
                f"⚠️ {route.upper()} route accuracy is very low: "
                f"{stats['accuracy']:.1%}"
            )

# Run periodically
schedule.every(1).hours.do(
    lambda: check_performance_and_alert(router)
)
```

## Performance Considerations

### Memory Usage

The dashboard loads feedback into memory:

```python
# Current implementation
feedbacks = router.get_recent_feedback(n=1000)  # ~1-5 MB

# For large histories (10,000+), consider sampling
feedbacks = router.get_recent_feedback(n=100)  # Smaller sample
```

### Speed Optimization

```python
# Cache metrics to avoid recalculation
import functools
from datetime import datetime, timedelta

@functools.lru_cache(maxsize=1)
def get_cached_metrics(timestamp_hour):
    """Cache metrics for 1 hour."""
    return router.get_metrics()

# Use in dashboard
current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
metrics = get_cached_metrics(current_hour)
```

### File I/O Optimization

```python
# For very large feedback files, use streaming
def stream_large_feedback_file(filename, n=1000):
    """Read last n lines without loading entire file."""
    with open(filename, 'rb') as f:
        # Seek to end
        f.seek(0, 2)
        file_size = f.tell()
        
        # Read backwards
        lines = []
        position = file_size
        
        while len(lines) < n and position > 0:
            # Read in chunks
            chunk_size = min(4096, position)
            position -= chunk_size
            f.seek(position)
            chunk = f.read(chunk_size).decode('utf-8')
            lines = chunk.splitlines() + lines
        
        return [json.loads(line) for line in lines[-n:]]
```

## Troubleshooting

### Issue: "No training data yet"

**Cause:** Router hasn't recorded any feedback

**Solution:**
```bash
# Run the RL CLI to generate training data
python -m cortex.rl_cli

# Or programmatically
from cortex.rl_router import get_rl_router
from cortex.router import Route

router = get_rl_router()
router.record_feedback(
    query="test query",
    predicted_route=Route.RAG,
    actual_route=Route.RAG,
    confidence_scores={'rag': 0.9, 'meta': 0.05, 'chat': 0.05}
)
```

### Issue: Colors not displaying

**Cause:** Terminal doesn't support ANSI codes

**Solution:**
```bash
# Use a compatible terminal (bash, zsh, etc.)
# Or disable colors in source code

# Quick fix: pipe through cat
python rl_dashboard.py | cat  # Strips ANSI codes
```

### Issue: Unicode characters broken

**Cause:** Terminal encoding not set to UTF-8

**Solution:**
```bash
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
python rl_dashboard.py
```

### Issue: Learning curve is flat

**Cause:** Not enough variance in data or window too large

**Solution:**
```python
# In print_learning_curve(), reduce window size
window_size = 5  # More sensitive (default: 10)
```

## Future Enhancements

1. **Export to CSV/Excel**: Save metrics for external analysis
2. **Comparative Views**: Compare multiple models side-by-side
3. **Query Clustering**: Group similar queries to identify patterns
4. **Temporal Analysis**: Show performance by time of day/week
5. **Automated Insights**: AI-generated explanations of performance issues
6. **Integration Tests**: Automated dashboard rendering tests
7. **Theme Support**: Light/dark theme options

## Example Workflows

### Daily Standup Report

```bash
#!/bin/bash
# daily_report.sh

echo "=== RL Router Daily Report ===" > report.txt
echo "Generated: $(date)" >> report.txt
echo "" >> report.txt

python rl_dashboard.py >> report.txt 2>&1

# Email to team
mail -s "Daily RL Router Report" team@example.com < report.txt
```

### A/B Testing Comparison

```python
# Compare two learning rates
router_a = RLRouter(base_router, learning_rate=0.1)
router_b = RLRouter(base_router, learning_rate=0.3)

# Train both on same data
for query, true_route in test_data:
    route_a, scores_a = router_a.route_query(query)
    route_b, scores_b = router_b.route_query(query)
    
    router_a.record_feedback(query, route_a, true_route, scores_a)
    router_b.record_feedback(query, route_b, true_route, scores_b)

# Compare
print("Router A (lr=0.1):")
print_route_performance(router_a)

print("\nRouter B (lr=0.3):")
print_route_performance(router_b)
```

### Performance Regression Testing

```python
import unittest

class DashboardTests(unittest.TestCase):
    def test_accuracy_threshold(self):
        """Ensure accuracy doesn't drop below threshold."""
        router = get_rl_router()
        metrics = router.get_metrics()
        
        self.assertGreater(
            metrics['overall_accuracy'],
            0.7,
            "Router accuracy below acceptable threshold"
        )
    
    def test_no_route_dominance(self):
        """Ensure no single route dominates."""
        router = get_rl_router()
        feedbacks = router.get_recent_feedback(n=100)
        
        route_counts = Counter(fb.predicted_route for fb in feedbacks)
        max_count = max(route_counts.values())
        
        self.assertLess(
            max_count / len(feedbacks),
            0.8,
            "One route is dominating predictions"
        )

if __name__ == '__main__':
    unittest.main()
```

## References

- ANSI Color Codes: https://en.wikipedia.org/wiki/ANSI_escape_code
- Box Drawing Characters: https://en.wikipedia.org/wiki/Box-drawing_character
- Terminal Visualization Best Practices: https://blog.devgenius.io/