# RL CLI Documentation

## Overview

The `rl_cli.py` module provides an interactive command-line interface for training and using the CORTEX Reinforcement Learning Router. It allows users to query the system, see model predictions, provide feedback, and watch the model improve in real-time through an intuitive visual interface.

## Purpose

This CLI serves as both a training tool and an interactive demonstration that:
- Enables hands-on reinforcement learning through user feedback
- Visualizes model predictions with confidence scores
- Shows learning progress in real-time
- Provides an engaging way to collect training data
- Demonstrates the RL router's learning capabilities
- Offers a user-friendly interface for model improvement

## Features

### 1. **Interactive Query Interface**
Users type queries and receive predictions with confidence visualizations.

### 2. **Visual Feedback Panel**
Color-coded display of predicted routes with confidence breakdowns.

### 3. **User Correction System**
Users select the actual correct category, teaching the model.

### 4. **Real-Time Learning**
Model updates weights immediately after each feedback.

### 5. **Progress Tracking**
Periodic accuracy updates and comprehensive statistics.

### 6. **Streaming Responses**
Live execution of queries with streaming output.

### 7. **Command System**
Built-in commands for stats, reset, help, and exit.

## Installation & Setup

### Prerequisites

```bash
# Ensure CORTEX is installed with dependencies
pip install cortex

# Or install from requirements
pip install -r requirements.txt
```

### Required Modules

```python
from cortex.router import Route, execute
from cortex.streaming import StreamHandler
from cortex.rl_router import get_rl_router, RLRouter
```

### Terminal Requirements

- **ANSI color support**: Most modern terminals (bash, zsh, PowerShell Core)
- **UTF-8 encoding**: For Unicode box-drawing characters
- **Minimum width**: 70 characters recommended

## Usage

### Basic Usage

```bash
# Run the CLI directly
python rl_cli.py

# Or as a module
python -m cortex.rl_cli

# Make executable (Unix/Linux/Mac)
chmod +x rl_cli.py
./rl_cli.py
```

### First Run Experience

```
======================================================================
  CORTEX - Interactive RL Training CLI
======================================================================
The model learns from your feedback to improve routing accuracy!

How it works:
  1. Type your query
  2. See the model's prediction
  3. Select the actual category
  4. The model learns and improves over time

Commands: 'exit', 'quit', 'stats', 'reset', 'help'
----------------------------------------------------------------------

> What is machine learning?

┌─ Model Prediction ──────────────────────────┐
│ ► Predicted: CHAT     (Confidence: 65.0%)
│
│ Confidence Breakdown:
│   ► CHAT   ████████████████████ 65.0%
│     RAG    ████████░░░░░░░░░░░░ 25.0%
│     META   ██░░░░░░░░░░░░░░░░░░ 10.0%
└─────────────────────────────────────────────┘

Select the ACTUAL category for this query:
  [1] RAG     - Retrieve from documents
  [2] META    - About the system
  [3] CHAT    - General conversation
  [0] Cancel

Your choice [1-3, 0 to cancel]: 3
```

### Interactive Session

```bash
# Query the system
> What documents do we have about Q3 sales?

# Model shows prediction
# You confirm or correct the category
# Model learns and executes query

# Check progress periodically
> stats

# Continue querying
> Tell me about the CORTEX system

# Reset if needed
> reset

# Exit when done
> exit
```

### Programmatic Usage

```python
from cortex.rl_cli import (
    print_prediction_panel,
    print_feedback_result,
    print_statistics,
    get_user_category_selection
)
from cortex.rl_router import get_rl_router
from cortex.router import Route

# Initialize router
router = get_rl_router(learning_rate=0.15)

# Get prediction
route, scores = router.route_query("sample query")

# Display prediction
print_prediction_panel(route, scores)

# Get user feedback
actual = get_user_category_selection()

# Record and learn
feedback = router.record_feedback(
    query="sample query",
    predicted_route=route,
    actual_route=actual,
    confidence_scores=scores
)

# Show result
print_feedback_result(route, actual, feedback.reward)
```

## Configuration

### Learning Rate

Set when initializing the router:

```python
# Default: 0.15 (balanced learning)
rl_router = get_rl_router(learning_rate=0.15)

# Conservative learning
rl_router = get_rl_router(learning_rate=0.05)

# Aggressive learning
rl_router = get_rl_router(learning_rate=0.3)
```

**Recommendations:**
- **0.05-0.10**: Slow, stable learning for production use
- **0.15-0.20**: Balanced learning for training sessions (default)
- **0.25-0.40**: Fast learning for quick experiments

### Display Configuration

Modify constants at the top of the file:

```python
# Color scheme (ANSI codes)
RESET = "\033[0m"
GREEN = "\033[92m"  # Success
YELLOW = "\033[93m" # Warning
BLUE = "\033[94m"   # RAG route
RED = "\033[91m"    # Error/META route
CYAN = "\033[96m"   # Headers
MAGENTA = "\033[95m"

# Route colors
ROUTE_COLORS = {
    Route.RAG: BLUE,
    Route.META: YELLOW,
    Route.CHAT: GREEN,
}
```

### Progress Update Frequency

```python
# In main() function, modify:
if rl_router.total_predictions % 5 == 0:  # Every 5 queries (default)
    # Show progress update

# Change to:
if rl_router.total_predictions % 10 == 0:  # Every 10 queries
```

## Commands Reference

### Interactive Commands

| Command | Aliases | Description |
|---------|---------|-------------|
| `exit` | `quit`, `q` | Exit the CLI and save progress |
| `stats` | - | Display comprehensive model statistics |
| `reset` | - | Reset all learning (requires confirmation) |
| `help` | `?` | Show available commands |

### Usage Examples

```bash
# View current performance
> stats

# Reset learning (careful!)
> reset
⚠ Reset all learning? (yes/no): yes
✓ Learning reset. Starting fresh!

# Get help
> help

# Exit
> exit
Goodbye! Model improvements saved.
```

## Function Reference

### 1. `print_header()`

Displays the welcome banner and instructions.

**Signature:**
```python
def print_header() -> None
```

**Example:**
```python
print_header()
```

**Output:**
```
======================================================================
  CORTEX - Interactive RL Training CLI
======================================================================
The model learns from your feedback to improve routing accuracy!

How it works:
  1. Type your query
  2. See the model's prediction
  3. Select the actual category
  4. The model learns and improves over time

Commands: 'exit', 'quit', 'stats', 'reset', 'help'
----------------------------------------------------------------------
```

**Customization:**
```python
def print_header():
    """Custom header with company branding."""
    print("\n" + "=" * 70)
    print(f"  {BOLD}{CYAN}YourCompany AI Assistant - Training Mode{RESET}")
    print("=" * 70)
    # ... rest of header
```

### 2. `format_confidence_bar()`

Creates an ASCII progress bar for confidence visualization.

**Signature:**
```python
def format_confidence_bar(confidence: float, width: int = 20) -> str
```

**Parameters:**
- `confidence` (float): Value between 0 and 1
- `width` (int): Total width of bar in characters

**Returns:**
- String with filled and empty blocks

**Example:**
```python
bar = format_confidence_bar(0.75, width=20)
print(bar)  # "███████████████░░░░░"

bar = format_confidence_bar(0.50, width=30)
print(bar)  # "███████████████░░░░░░░░░░░░░░░"
```

**Customization:**
```python
# Use different characters
def format_confidence_bar(confidence: float, width: int = 20) -> str:
    filled = int(confidence * width)
    bar = "#" * filled + "." * (width - filled)  # ASCII-safe
    return bar

# Use gradient
def format_confidence_bar(confidence: float, width: int = 20) -> str:
    filled = int(confidence * width)
    if confidence > 0.8:
        char = "█"
    elif confidence > 0.5:
        char = "▓"
    else:
        char = "▒"
    bar = char * filled + "░" * (width - filled)
    return bar
```

### 3. `print_prediction_panel()`

Displays the model's prediction with confidence breakdown.

**Signature:**
```python
def print_prediction_panel(route: Route, scores: Dict[str, float]) -> None
```

**Parameters:**
- `route` (Route): Predicted route
- `scores` (Dict[str, float]): Confidence scores for all routes

**Example:**
```python
route = Route.RAG
scores = {"rag": 0.7, "meta": 0.2, "chat": 0.1}
print_prediction_panel(route, scores)
```

**Output:**
```
┌─ Model Prediction ──────────────────────────┐
│ ► Predicted: RAG      (Confidence: 70.0%)
│
│ Confidence Breakdown:
│   ► RAG    ██████████████░░░░░░ 70.0%
│     META   ████░░░░░░░░░░░░░░░░ 20.0%
│     CHAT   ██░░░░░░░░░░░░░░░░░░ 10.0%
└─────────────────────────────────────────────┘
```

**Customization:**
```python
# Simpler version without box drawing
def print_prediction_panel_simple(route: Route, scores: Dict[str, float]):
    print(f"\n--- Model Prediction ---")
    print(f"Predicted: {route.value.upper()} ({scores[route.value]:.1%})")
    print(f"\nAll Scores:")
    for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        bar = "#" * int(score * 20)
        print(f"  {name.upper():<6} {bar:<20} {score:.1%}")
    print()
```

### 4. `get_user_category_selection()`

Prompts user to select the correct category.

**Signature:**
```python
def get_user_category_selection() -> Optional[Route]
```

**Returns:**
- `Route` object if user makes a selection
- `None` if user cancels

**Example:**
```python
actual = get_user_category_selection()
if actual is None:
    print("User cancelled")
else:
    print(f"User selected: {actual.value}")
```

**Behavior:**
- Validates input (1-3 or 0)
- Handles KeyboardInterrupt (Ctrl+C)
- Handles EOFError (Ctrl+D)
- Loops until valid input

**Customization:**
```python
# Add keyboard shortcuts
def get_user_category_selection() -> Optional[Route]:
    print("Select category: [r]ag, [m]eta, [c]hat, [0] cancel")
    
    choice = input("Your choice: ").strip().lower()
    
    mapping = {
        'r': Route.RAG, '1': Route.RAG,
        'm': Route.META, '2': Route.META,
        'c': Route.CHAT, '3': Route.CHAT,
        '0': None
    }
    
    return mapping.get(choice)
```

### 5. `print_feedback_result()`

Shows the learning outcome after user feedback.

**Signature:**
```python
def print_feedback_result(predicted: Route, actual: Route, reward: float) -> None
```

**Parameters:**
- `predicted` (Route): Model's prediction
- `actual` (Route): User's correction
- `reward` (float): Calculated reward/penalty

**Example:**
```python
print_feedback_result(
    predicted=Route.CHAT,
    actual=Route.RAG,
    reward=-1.25
)
```

**Output:**
```
┌─ Learning Feedback ────────────────────────┐
│ ✗ INCORRECT 📚
│
│ Predicted: CHAT
│ Actual:    RAG
│
│ Reward: -1.25
│ Model weights updated (learning from mistake)
└────────────────────────────────────────────┘
```

### 6. `print_statistics()`

Displays comprehensive model performance metrics.

**Signature:**
```python
def print_statistics(rl_router: RLRouter) -> None
```

**Parameters:**
- `rl_router` (RLRouter): The RL router instance

**Example:**
```python
router = get_rl_router()
print_statistics(router)
```

**Output:**
```
┌─ Model Statistics ──────────────────────────┐
│
│ Overall Accuracy: 85.0% (85/100 correct)
│
│ Per-Category Performance:
│   RAG    ███████████████ 90.0% (45/50)
│   META   ████████████░░░ 75.0% (15/20)
│   CHAT   ██████████████░ 83.3% (25/30)
│
│ Learned Confidence Weights:
│   RAG    1.250 ↑
│   META   0.850 ↓
│   CHAT   1.100 ↑
└─────────────────────────────────────────────┘
```

### 7. `execute_with_streaming()`

Executes the query with live streaming output.

**Signature:**
```python
def execute_with_streaming(query: str, route: Route) -> str
```

**Parameters:**
- `query` (str): The user's query
- `route` (Route): Route to execute (actual/corrected route)

**Returns:**
- Complete response string

**Example:**
```python
response = execute_with_streaming(
    "What is machine learning?",
    Route.CHAT
)
```

**How it works:**
1. Prints route indicator: `[CHAT] `
2. Streams tokens one at a time as they arrive
3. Returns complete response when finished

**Customization:**
```python
# Add typing delay for effect
import time

def execute_with_streaming_slow(query: str, route: Route) -> str:
    print(f"\n[{route.value.upper()}] ", end="", flush=True)
    
    def on_token(token: str):
        print(token, end="", flush=True)
        time.sleep(0.01)  # Slight delay
    
    handler = StreamHandler(on_token=on_token)
    result, _, _ = execute(query, callbacks=[handler])
    print("\n")
    return result
```

### 8. `main()`

Main interactive loop orchestrating the entire CLI.

**Signature:**
```python
def main() -> None
```

**Flow:**
1. Print header
2. Initialize RL router
3. Show existing stats (if any)
4. Enter interactive loop:
   - Get user query
   - Handle commands
   - Get model prediction
   - Show prediction panel
   - Get user category selection
   - Record feedback
   - Show learning result
   - Execute query
   - Show periodic progress
5. Handle graceful exit

**Customization:**
```python
# Add logging
import logging

def main():
    logging.basicConfig(
        filename='rl_cli_session.log',
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    print_header()
    rl_router = get_rl_router(learning_rate=0.15)
    
    while True:
        try:
            query = input(f"{BOLD}> {RESET}").strip()
            logging.info(f"User query: {query}")
            # ... rest of main loop
```

## Training Workflows

### 1. Quick Training Session (30 queries)

```bash
python rl_cli.py

# Ask 30 diverse queries across all categories
> What is the capital of France?     # CHAT
> Show me the Q3 report              # RAG
> What can you do?                   # META
> Explain quantum computing          # CHAT
> Find documents about sales         # RAG
# ... continue for 30 total queries

> stats  # Check progress
> exit   # Save and exit
```

### 2. Category-Focused Training

```bash
# Train specifically on RAG queries
python rl_cli.py

# Only ask document-retrieval questions
> Find all contracts from 2024
> Show me the employee handbook
> What's in the latest quarterly report?
# ... 20-30 RAG queries

> stats  # Verify RAG accuracy improved
```

### 3. Edge Case Collection

```bash
# Focus on ambiguous queries
python rl_cli.py

# Ask queries that could fit multiple categories
> What documents explain how you work?  # RAG or META?
> Tell me about our sales strategy      # RAG or CHAT?
> What are you capable of?              # META or CHAT?

# Your corrections teach the model these edge cases
```

### 4. Batch Training from File

```python
# train_from_file.py
from cortex.rl_cli import main
from cortex.rl_router import get_rl_router
from cortex.router import Route

def batch_train(training_data):
    """Train from labeled dataset."""
    router = get_rl_router(learning_rate=0.2)
    
    for query, correct_route in training_data:
        # Get prediction
        predicted, scores = router.route_query(query)
        
        # Record feedback
        feedback = router.record_feedback(
            query=query,
            predicted_route=predicted,
            actual_route=Route(correct_route),
            confidence_scores=scores
        )
        
        print(f"Query: {query[:50]}")
        print(f"  Predicted: {predicted.value}, Actual: {correct_route}")
        print(f"  Reward: {feedback.reward:.2f}\n")
    
    # Show final stats
    metrics = router.get_metrics()
    print(f"Final accuracy: {metrics['overall_accuracy']:.1%}")

# Load training data
training_data = [
    ("What is machine learning?", "chat"),
    ("Show me the sales report", "rag"),
    ("What can you do?", "meta"),
    # ... more examples
]

batch_train(training_data)
```

### 5. Continuous Improvement Loop

```python
# continuous_training.py
import schedule
import time
from cortex.rl_router import get_rl_router

def daily_review():
    """Review and improve model daily."""
    router = get_rl_router()
    metrics = router.get_metrics()
    
    if metrics['overall_accuracy'] < 0.7:
        print("⚠️ Accuracy below target. Schedule training session.")
        # Send notification, create task, etc.
    else:
        print("✓ Model performing well.")

schedule.every().day.at("09:00").do(daily_review)

while True:
    schedule.run_pending()
    time.sleep(3600)
```

## Best Practices

### 1. Training Data Quality

**Good Practices:**
- Provide diverse queries across all categories
- Include edge cases and ambiguous queries
- Be consistent with your categorizations
- Regularly review and correct your own mistakes

**Bad Practices:**
- Always selecting the same category
- Random categorizations
- Ignoring the prediction completely
- Not reading queries carefully

### 2. Session Management

```bash
# Start of day: Check current performance
> stats

# Train for 20-30 minutes
# ... query and correct

# End of session: Review progress
> stats

# Exit and save
> exit
```

### 3. Learning Rate Selection

| Use Case | Learning Rate | Rationale |
|----------|---------------|-----------|
| Initial training (0-50 queries) | 0.20-0.30 | Fast learning needed |
| Ongoing improvement (50-200) | 0.10-0.15 | Balanced approach |
| Fine-tuning (200+) | 0.05-0.10 | Stable, incremental |
| Production use | 0.05 | Conservative updates |

### 4. Progress Monitoring

```python
# Add custom progress tracking
def track_session_progress():
    router = get_rl_router()
    start_metrics = router.get_metrics()
    start_acc = start_metrics['overall_accuracy']
    start_total = start_metrics['total_predictions']
    
    # ... training session happens ...
    
    end_metrics = router.get_metrics()
    end_acc = end_metrics['overall_accuracy']
    queries_added = end_metrics['total_predictions'] - start_total
    
    improvement = (end_acc - start_acc) * 100
    print(f"\nSession Summary:")
    print(f"  Queries: {queries_added}")
    print(f"  Accuracy change: {improvement:+.1f}pp")
    print(f"  Final accuracy: {end_acc:.1%}")
```

### 5. Error Recovery

The CLI handles errors gracefully:

```python
try:
    # Main loop
except KeyboardInterrupt:
    # Ctrl+C → Show message, continue
    print("\nInterrupted. Type 'exit' to quit.")
except EOFError:
    # Ctrl+D → Exit gracefully
    print("\nGoodbye!")
except Exception as e:
    # Unexpected errors → Show traceback
    print(f"Error: {e}")
    traceback.print_exc()
```

## Alternative Interfaces

### 1. GUI Version (Tkinter)

```python
import tkinter as tk
from tkinter import ttk

class RLTrainerGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("CORTEX RL Trainer")
        self.router = get_rl_router(learning_rate=0.15)
        
        # Query input
        self.query_entry = tk.Entry(self.window, width=50)
        self.query_entry.pack(pady=10)
        
        # Predict button
        tk.Button(
            self.window,
            text="Get Prediction",
            command=self.predict
        ).pack()
        
        # Prediction display
        self.pred_label = tk.Label(self.window, text="")
        self.pred_label.pack(pady=10)
        
        # Category selection
        self.category_var = tk.StringVar()
        for cat in ["RAG", "META", "CHAT"]:
            tk.Radiobutton(
                self.window,
                text=cat,
                variable=self.category_var,
                value=cat.lower()
            ).pack()
        
        # Submit button
        tk.Button(
            self.window,
            text="Submit Feedback",
            command=self.submit_feedback
        ).pack(pady=10)
        
        # Stats display
        self.stats_text = tk.Text(self.window, height=10, width=50)
        self.stats_text.pack()
        
    def predict(self):
        query = self.query_entry.get()
        self.predicted_route, self.scores = self.router.route_query(query)
        
        self.pred_label.config(
            text=f"Predicted: {self.predicted_route.value.upper()} "
                 f"({self.scores[self.predicted_route.value]:.1%})"
        )
    
    def submit_feedback(self):
        actual = Route(self.category_var.get())
        query = self.query_entry.get()
        
        feedback = self.router.record_feedback(
            query=query,
            predicted_route=self.predicted_route,
            actual_route=actual,
            confidence_scores=self.scores
        )
        
        # Update stats
        metrics = self.router.get_metrics()
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, 
            f"Accuracy: {metrics['overall_accuracy']:.1%}\n"
            f"Total: {metrics['total_predictions']}\n"
            f"Reward: {feedback.reward:.2f}"
        )
        
        # Clear input
        self.query_entry.delete(0, tk.END)
    
    def run(self):
        self.window.mainloop()

# Run
if __name__ == "__main__":
    app = RLTrainerGUI()
    app.run()
```

### 2. Web Interface (Flask)

```python
from flask import Flask, render_template, request, jsonify
from cortex.rl_router import get_rl_router
from cortex.router import Route

app = Flask(__name__)
router = get_rl_router(learning_rate=0.15)

@app.route('/')
def index():
    return render_template('trainer.html')

@app.route('/predict', methods=['POST'])
def predict():
    query = request.json['query']
    predicted, scores = router.route_query(query)
    
    return jsonify({
        'predicted': predicted.value,
        'scores': scores
    })

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    
    feedback = router.record_feedback(
        query=data['query'],
        predicted_route=Route(data['predicted']),
        actual_route=Route(data['actual']),
        confidence_scores=data['scores']
    )
    
    return jsonify({
        'reward': feedback.reward,
        'correct': feedback.correct
    })

@app.route('/stats')
def stats():
    metrics = router.get_metrics()
    return jsonify(metrics)

if __name__ == '__main__':
    app.run(debug=True)
```

### 3. Telegram Bot

```python
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

router = get_rl_router(learning_rate=0.15)
current_prediction = {}

async def predict(update: Update, context):
    """Handle user query and show prediction."""
    query = update.message.text
    user_id = update.effective_user.id
    
    predicted, scores = router.route_query(query)
    current_prediction[user_id] = (query, predicted, scores)
    
    await update.message.reply_text(
        f"Predicted: {predicted.value.upper()}\n"
        f"Confidence: {scores[predicted.value]:.1%}\n\n"
        "Is this correct? Reply with:\n"
        "/rag, /meta, or /chat"
    )

async def handle_feedback(update: Update, context, route_name):
    """Record user feedback."""
    user_id = update.effective_user.id
    
    if user_id not in current_prediction:
        await update.message.reply_text("No pending prediction.")
        return
    
    query, predicted, scores = current_prediction[user_id]
    actual = Route(route_name)
    
    feedback = router.record_feedback(
        query=query,
        predicted_route=predicted,
        actual_route=actual,
        confidence_scores=scores
    )
    
    icon = "✅" if feedback.correct else "❌"
    await update.message.reply_text(
        f"{icon} Feedback recorded!\n"
        f"Reward: {feedback.reward:.2f}"
    )
    
    del current_prediction[user_id]

# Set up handlers
app = Application.builder().token("YOUR_TOKEN").build()
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, predict))
app.add_handler(CommandHandler("rag", lambda u, c: handle_feedback(u, c, "rag")))
app.add_handler(CommandHandler("meta", lambda u, c: handle_feedback(u, c, "meta")))
app.add_handler(CommandHandler("chat", lambda u, c: handle_feedback(u, c, "chat")))

app.run_polling()
```

## Troubleshooting

### Issue: Colors not displaying

**Symptoms:** ANSI codes visible as text: `\033[92m`

**Solutions:**
```bash
# Use a compatible terminal
# Windows: Use PowerShell Core or Windows Terminal
# Mac/Linux: Most terminals work by default

# Or disable colors in code
# At top of file:
RESET = ""
GREEN = ""
# ... set all colors to empty strings
```

### Issue: Unicode characters broken

**Symptoms:** Box drawing shows as `?` or weird characters

**Solutions:**
```bash
# Set UTF-8 encoding
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# Or use ASCII fallback:
# Replace box characters with ASCII
# ┌ → +
# │ → |
# └ → +
# ─ → -
```

### Issue: "No module named cortex"

**Symptoms:** Import errors

**Solutions:**
```bash
# Ensure you're in the right directory
cd /path/to/cortex/project

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/cortex"

# Or install in development mode
pip install -e .
```

### Issue: Predictions don't improve

**Symptoms:** Accuracy stays flat after many queries

**Possible Causes:**
1. **Learning rate too low**: Increase to 0.2-0.3
2. **Inconsistent feedback**: Review your categorizations
3. **Base router already optimal**: Check base router accuracy
4. **Too few examples**: Need at least 50-100 queries per category

**Solutions:**
```python
# Increase learning rate
router = get_rl_router(learning_rate=0.3)

# Reset and retrain
> reset
> yes

# Verify base router performance
from cortex.router import get_router
base = get_router()
route, scores = base.route_query("test query")
print(scores)
```

### Issue: Session data not persisting

**Symptoms:** Stats reset every time you run CLI

**Solutions:**
```bash
# Check feedback directory exists
ls rl_feedback/

# Verify file permissions
chmod 755 rl_feedback/
chmod 644 rl_feedback/*

# Check file contents
cat rl_feedback/metrics.json
```

## Performance Optimization

### Faster Startup

```python
# Cache router instance
_cached_router = None

def get_cached_router():
    global _cached_router
    if _cached_router is None:
        _cached_router = get_rl_router(learning_rate=0.15)
    return _cached_router
```

### Reduced Memory Usage

```python
# Limit feedback history in memory
router = RLRouter(
    base_router=base_router,
    memory_size=500  # Default: 1000
)
```

### Batch Operations

```python
# Process multiple queries before updating display
batch = []
for query in queries:
    batch.append((query, actual_route))
    
    if len(batch) >= 10:
        for q, actual in batch:
            predicted, scores = router.route_query(q)
            router.record_feedback(q, predicted, actual, scores)
        batch.clear()
        print_statistics(router)
```

## Security Considerations

### Input Validation

The CLI accepts arbitrary user input. For production use:

```python
def sanitize_query(query: str) -> str:
    """Sanitize user input."""
    # Limit length
    max_length = 1000
    query = query[:max_length]
    
    # Remove control characters
    query = ''.join(char for char in query if char.isprintable() or char.isspace())
    
    return query.strip()

# Use in main loop
query = sanitize_query(input(f"{BOLD}> {RESET}"))
```

### File Permissions

```bash
# Restrict feedback directory access
chmod 700 rl_feedback/
chmod 600 rl_feedback/*.json*

# Prevent unauthorized modifications
sudo chown -R your_user:your_group rl_feedback/
```

### Rate Limiting

```python
import time

class RateLimiter:
    def __init__(self, max_requests=10, time_window=60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    def allow_request(self):
        now = time.time()
        # Remove old requests
        self.requests = [t for t in self.requests if now - t < self.time_window]
        
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False

# Use in main loop
limiter = RateLimiter(max_requests=20, time_window=60)

while True:
    if not limiter.allow_request():
        print(f"{RED}Rate limit exceeded. Please slow down.{RESET}")
        time.sleep(5)
        continue
    
    query = input(f"{BOLD}> {RESET}")
    # ... rest of loop
```

## Testing

### Unit Tests

```python
import unittest
from unittest.mock import patch, MagicMock
from cortex.rl_cli import (
    format_confidence_bar,
    get_user_category_selection
)

class TestRLCLI(unittest.TestCase):
    def test_confidence_bar(self):
        """Test confidence bar formatting."""
        bar = format_confidence_bar(0.5, width=10)
        self.assertEqual(len(bar), 10)
        self.assertEqual(bar.count('█'), 5)
        self.assertEqual(bar.count('░'), 5)
    
    def test_confidence_bar_extremes(self):
        """Test edge cases."""
        bar_zero = format_confidence_bar(0.0, width=10)
        self.assertEqual(bar_zero.count('█'), 0)
        
        bar_one = format_confidence_bar(1.0, width=10)
        self.assertEqual(bar_one.count('█'), 10)
    
    @patch('builtins.input', return_value='1')
    def test_user_selection_rag(self, mock_input):
        """Test RAG selection."""
        route = get_user_category_selection()
        self.assertEqual(route, Route.RAG)
    
    @patch('builtins.input', return_value='0')
    def test_user_selection_cancel(self, mock_input):
        """Test cancellation."""
        route = get_user_category_selection()
        self.assertIsNone(route)

if __name__ == '__main__':
    unittest.main()
```

### Integration Tests

```python
def test_full_training_session():
    """Test complete training workflow."""
    from cortex.rl_router import get_rl_router
    from cortex.router import Route
    
    router = get_rl_router(learning_rate=0.2)
    initial_acc = router.get_metrics()['overall_accuracy']
    
    # Simulate training
    training_data = [
        ("What is AI?", Route.CHAT),
        ("Show me docs", Route.RAG),
        ("What can you do?", Route.META),
    ] * 20  # 60 queries total
    
    for query, actual in training_data:
        predicted, scores = router.route_query(query)
        router.record_feedback(query, predicted, actual, scores)
    
    final_acc = router.get_metrics()['overall_accuracy']
    
    # Accuracy should improve
    assert final_acc > initial_acc, "Model did not learn"
    print(f"✓ Accuracy improved: {initial_acc:.1%} → {final_acc:.1%}")
```

### Smoke Tests

```bash
#!/bin/bash
# smoke_test.sh

echo "Running RL CLI smoke tests..."

# Test 1: CLI starts
timeout 5 python rl_cli.py <<EOF
exit
EOF

if [ $? -eq 0 ]; then
    echo "✓ CLI starts successfully"
else
    echo "✗ CLI failed to start"
    exit 1
fi

# Test 2: Stats command
timeout 5 python rl_cli.py <<EOF
stats
exit
EOF

if [ $? -eq 0 ]; then
    echo "✓ Stats command works"
else
    echo "✗ Stats command failed"
    exit 1
fi

echo "All smoke tests passed!"
```

## FAQ

### Q: How many training examples do I need?

**A:** Minimum 20-30 per category (60-90 total) for basic performance. For production-quality routing, aim for 100-200 per category (300-600 total).

### Q: Can I undo incorrect feedback?

**A:** Not directly through the CLI. You can:
1. Use `reset` to start fresh (loses all learning)
2. Manually edit `rl_feedback/feedback.jsonl` to remove entries
3. Add corrective examples to counterbalance mistakes

### Q: Why doesn't the model use my feedback for routing?

**A:** The model DOES use feedback, but through adjusted confidence weights, not direct routing. It modifies the base router's confidence scores, so you might not see immediate route changes for individual queries.

### Q: Can multiple people train the same model?

**A:** Yes, but be careful:
- Shared file system: Works, but may have race conditions
- Version control: Commit `rl_feedback/` directory
- Database backend: Better for multi-user (requires custom implementation)

### Q: How do I export my training data?

**A:**
```python
import json

# Read feedback file
with open('rl_feedback/feedback.jsonl', 'r') as f:
    feedbacks = [json.loads(line) for line in f]

# Export to CSV
import csv
with open('training_export.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=feedbacks[0].keys())
    writer.writeheader()
    writer.writerows(feedbacks)
```

### Q: Can I use this in production?

**A:** The CLI is for training. For production routing, use the `RLRouter` directly in your application:

```python
from cortex.rl_router import get_rl_router

router = get_rl_router(learning_rate=0.05)  # Conservative rate
route, scores = router.route_query(user_query)

# Use route for actual routing
response = execute(user_query, route)
```

## Future Enhancements

1. **Multi-User Support**: Collaborative training sessions
2. **Query Suggestions**: AI-generated training queries
3. **Export/Import**: Backup and restore training data
4. **Annotation Mode**: Review and re-label historical queries
5. **A/B Comparison**: Compare two models side-by-side
6. **Voice Input**: Speech-to-text for queries
7. **Auto-Save**: Checkpoint every N queries
8. **Undo/Redo**: Navigate feedback history
9. **Themes**: Light/dark mode customization
10. **Localization**: Multi-language support

## References

- ANSI Escape Codes: https://en.wikipedia.org/wiki/ANSI_escape_code
- Terminal User Interfaces: https://github.com/Textualize/rich
- Reinforcement Learning: Sutton & Barto, "Reinforcement Learning: An Introduction"
- Interactive ML: https://pair.withgoogle.com/guidebook/