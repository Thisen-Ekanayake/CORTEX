# Run Script Documentation

## Overview

The `run.py` module serves as the main entry point and command-line interface (CLI) for the CORTEX document Q&A assistant. It provides a simple, user-friendly interface for both document ingestion and querying operations.

## Purpose

This script acts as the orchestration layer by:

1. **Providing CLI interface** for user interaction
2. **Routing commands** to appropriate subsystems (ingest or query)
3. **Validating prerequisites** before executing operations
4. **Enabling logging** for query operations
5. **Simplifying workflow** for end users

## Architecture

### Command-Line Arguments

The script uses Python's `argparse` module to handle two main operations:

#### `--ingest`

Triggers document ingestion into the vector database.

**Usage**:
```bash
python run.py --ingest
```

**Behavior**:
- Calls `ingest()` from `cortex.ingest`
- Processes documents from configured directory
- Builds/updates vector database (ChromaDB)
- No return value to user (completion is implicit)

**Use Cases**:
- Initial setup of document database
- Adding new documents
- Rebuilding index after document updates
- Periodic re-indexing

#### `--ask`

Asks a question to CORTEX.

**Usage**:
```bash
python run.py --ask "What's in the Q3 report?"
```

**Behavior**:
1. Checks if vector database exists (`chroma_db/chroma.sqlite3`)
2. If missing, prompts user to run `--ingest` first
3. If present:
   - Configures logging to `query.log`
   - Logs the query
   - Calls `ask()` from `cortex.query`
   - Displays response to user

**Prerequisites**:
- Vector database must exist (created via `--ingest`)

## Workflow

### 1. First-Time Setup

```bash
# Step 1: Ingest documents
python run.py --ingest

# Step 2: Ask questions
python run.py --ask "What is the revenue for Q3?"
```

### 2. Adding New Documents

```bash
# Re-run ingest to include new documents
python run.py --ingest

# Query updated database
python run.py --ask "What's in the latest report?"
```

### 3. Error Handling

```bash
# Attempting to query without ingestion
python run.py --ask "Hello"
# Output: "Vector DB not found. Run with --ingest first"
```

## Implementation Details

### Logging Configuration

```python
logging.basicConfig(filename="query.log", level=logging.INFO)
logging.info(f"Query asked: {args.ask}")
```

**Configuration**:
- **File**: `query.log` (created in current directory)
- **Level**: `INFO` (logs informational messages and above)
- **Format**: Default format (timestamp, level, message)

**Logged Information**:
- User queries (for audit trail)
- Timestamp (automatic from logging module)

**Log File Location**:
- Same directory as `run.py`
- Created automatically if doesn't exist
- Appends to existing file

### Database Check

```python
if not os.path.exists("chroma_db/chroma.sqlite3"):
    print("Vector DB not found. Run with --ingest first")
```

**Purpose**:
- Prevents errors from querying non-existent database
- Provides clear user guidance
- Validates prerequisites before expensive operations

**Database Path**:
- Hardcoded: `chroma_db/chroma.sqlite3`
- ChromaDB's SQLite storage format
- Located relative to script execution directory

## Usage Examples

### 1. Basic Usage

```bash
# Ingest documents
python run.py --ingest

# Ask single question
python run.py --ask "What are the main conclusions?"

# Ask another question
python run.py --ask "Who are the authors?"
```

### 2. With Virtual Environment

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Run commands
python run.py --ingest
python run.py --ask "Your question here"
```

### 3. Script Automation

```bash
#!/bin/bash
# ingest_and_query.sh

# Ingest documents
echo "Ingesting documents..."
python run.py --ingest

# Ask multiple questions
questions=(
    "What is the total revenue?"
    "Who are the key stakeholders?"
    "What are the risks mentioned?"
)

for question in "${questions[@]}"; do
    echo "Asking: $question"
    python run.py --ask "$question"
    echo "---"
done
```

### 4. Scheduled Re-indexing

```bash
# cron job for daily re-indexing (Linux/Mac)
# Add to crontab -e:
0 2 * * * cd /path/to/cortex && python run.py --ingest
```

### 5. Logging Inspection

```bash
# View all queries
cat query.log

# View recent queries
tail -n 20 query.log

# Search for specific query
grep "revenue" query.log

# Follow log in real-time
tail -f query.log
```

## Integration Examples

### 1. Enhanced with Router System

```python
# run_enhanced.py
import os
import argparse
import logging
from cortex.ingest import ingest
from cortex.router import execute

def main():
    parser = argparse.ArgumentParser(description="CORTEX - AI Assistant")
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--ask", type=str)
    parser.add_argument("--verbose", action="store_true", help="Show routing info")
    
    args = parser.parse_args()
    
    if args.ingest:
        print("Ingesting documents...")
        ingest()
        print("✓ Ingestion complete")
    
    if args.ask:
        if not os.path.exists("chroma_db/chroma.sqlite3"):
            print("Vector DB not found. Run with --ingest first")
            return
        
        logging.basicConfig(filename="query.log", level=logging.INFO)
        logging.info(f"Query: {args.ask}")
        
        # Use router instead of direct ask
        result, route, scores = execute(args.ask)
        
        print(f"\nAnswer: {result}")
        
        if args.verbose:
            print(f"\nRoute used: {route.value}")
            print(f"Confidence: {scores[route.value]:.2%}")
        
        logging.info(f"Route: {route.value}, Confidence: {scores[route.value]:.2f}")

if __name__ == "__main__":
    main()
```

### 2. Interactive Mode

```python
# run_interactive.py
import os
from cortex.ingest import ingest
from cortex.router import execute
from memory import ConversationMemory

def interactive_mode():
    """Run CORTEX in interactive mode"""
    memory = ConversationMemory()
    
    print("CORTEX Interactive Mode")
    print("Commands: 'exit', 'quit', 'help', 'history'")
    print("-" * 50)
    
    while True:
        try:
            query = input("\nYou: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            if query.lower() == 'help':
                print("Available commands:")
                print("  - Ask any question")
                print("  - 'history' - Show conversation history")
                print("  - 'exit' - Quit interactive mode")
                continue
            
            if query.lower() == 'history':
                queries = memory.all_queries()
                print(f"\nConversation history ({len(queries)} items):")
                for i, q in enumerate(queries, 1):
                    print(f"  {i}. {q}")
                continue
            
            # Execute query
            result, route, scores = execute(query)
            memory.add(query, result)
            
            print(f"\nCORTEX: {result}")
            
        except KeyboardInterrupt:
            print("\nUse 'exit' to quit")
        except Exception as e:
            print(f"Error: {e}")

def main():
    if not os.path.exists("chroma_db/chroma.sqlite3"):
        response = input("Vector DB not found. Run ingestion now? (y/n): ")
        if response.lower() == 'y':
            ingest()
        else:
            print("Cannot proceed without vector database.")
            return
    
    interactive_mode()

if __name__ == "__main__":
    main()
```

### 3. Web API Wrapper

```python
# api_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from cortex.ingest import ingest
from cortex.router import execute

app = FastAPI(title="CORTEX API")

class Query(BaseModel):
    question: str

class IngestRequest(BaseModel):
    force: bool = False

@app.post("/ingest")
async def api_ingest(request: IngestRequest):
    """Trigger document ingestion"""
    try:
        ingest()
        return {"status": "success", "message": "Documents ingested"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def api_ask(query: Query):
    """Ask question to CORTEX"""
    if not os.path.exists("chroma_db/chroma.sqlite3"):
        raise HTTPException(
            status_code=400,
            detail="Vector DB not found. Run /ingest first"
        )
    
    try:
        result, route, scores = execute(query.question)
        return {
            "answer": result,
            "route": route.value,
            "confidence": scores[route.value]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check system health"""
    db_exists = os.path.exists("chroma_db/chroma.sqlite3")
    return {
        "status": "healthy",
        "database_ready": db_exists
    }

# Run with: uvicorn api_server:app --reload
```

### 4. Batch Processing

```python
# run_batch.py
import os
import argparse
import json
from cortex.router import execute

def batch_process(questions_file: str, output_file: str):
    """Process multiple questions from file"""
    
    if not os.path.exists("chroma_db/chroma.sqlite3"):
        print("Vector DB not found. Run --ingest first")
        return
    
    # Load questions
    with open(questions_file, 'r') as f:
        questions = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(questions)} questions...")
    
    results = []
    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {question}")
        
        answer, route, scores = execute(question)
        
        results.append({
            'question': question,
            'answer': answer,
            'route': route.value,
            'confidence': scores[route.value]
        })
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True, help="File with questions")
    parser.add_argument("--output", default="results.json", help="Output file")
    
    args = parser.parse_args()
    batch_process(args.questions, args.output)

if __name__ == "__main__":
    main()
```

### 5. GUI Wrapper

```python
# run_gui.py
import tkinter as tk
from tkinter import scrolledtext, messagebox
import os
from cortex.ingest import ingest
from cortex.router import execute

class CortexGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CORTEX Assistant")
        self.root.geometry("600x500")
        
        # Question input
        tk.Label(root, text="Ask CORTEX:", font=("Arial", 12)).pack(pady=5)
        self.question_entry = tk.Entry(root, width=70)
        self.question_entry.pack(pady=5)
        self.question_entry.bind("<Return>", lambda e: self.ask_question())
        
        # Buttons
        button_frame = tk.Frame(root)
        button_frame.pack(pady=5)
        
        tk.Button(button_frame, text="Ask", command=self.ask_question, width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Ingest", command=self.run_ingest, width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Clear", command=self.clear_output, width=10).pack(side=tk.LEFT, padx=5)
        
        # Output area
        tk.Label(root, text="Response:", font=("Arial", 12)).pack(pady=5)
        self.output_text = scrolledtext.ScrolledText(root, width=70, height=20)
        self.output_text.pack(pady=5)
    
    def ask_question(self):
        question = self.question_entry.get().strip()
        if not question:
            return
        
        if not os.path.exists("chroma_db/chroma.sqlite3"):
            messagebox.showwarning("Warning", "Vector DB not found. Run Ingest first.")
            return
        
        try:
            self.output_text.insert(tk.END, f"\nYou: {question}\n")
            self.output_text.insert(tk.END, "CORTEX: ")
            self.root.update()
            
            result, route, scores = execute(question)
            
            self.output_text.insert(tk.END, f"{result}\n")
            self.output_text.insert(tk.END, f"[Route: {route.value}, Confidence: {scores[route.value]:.2%}]\n")
            self.output_text.insert(tk.END, "-" * 70 + "\n")
            self.output_text.see(tk.END)
            
            self.question_entry.delete(0, tk.END)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def run_ingest(self):
        response = messagebox.askyesno("Confirm", "Run document ingestion?")
        if response:
            try:
                self.output_text.insert(tk.END, "\nRunning ingestion...\n")
                self.root.update()
                ingest()
                self.output_text.insert(tk.END, "✓ Ingestion complete\n")
                messagebox.showinfo("Success", "Documents ingested successfully!")
            except Exception as e:
                messagebox.showerror("Error", str(e))
    
    def clear_output(self):
        self.output_text.delete(1.0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = CortexGUI(root)
    root.mainloop()
```

## Configuration & Customization

### 1. Configurable Paths

```python
# run_configurable.py
import os
import argparse
import yaml

# Load configuration
def load_config(config_file="config.yaml"):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--ask", type=str)
    parser.add_argument("--config", default="config.yaml")
    
    args = parser.parse_args()
    
    # Use paths from config
    db_path = config['database']['path']
    log_file = config['logging']['query_log']
    
    if args.ask:
        db_file = os.path.join(db_path, "chroma.sqlite3")
        if not os.path.exists(db_file):
            print(f"Vector DB not found at {db_file}")
            return
        
        # ... rest of logic
```

**config.yaml**:
```yaml
database:
  path: "chroma_db"
  collection: "documents"

logging:
  query_log: "logs/query.log"
  level: "INFO"

ingest:
  documents_dir: "data/documents"
  chunk_size: 1000
  chunk_overlap: 200
```

### 2. Enhanced Error Handling

```python
# run_robust.py
import sys
import traceback

def main():
    try:
        parser = argparse.ArgumentParser()
        # ... argument setup
        
        args = parser.parse_args()
        
        if args.ingest:
            try:
                ingest()
                print("✓ Ingestion successful")
            except FileNotFoundError as e:
                print(f"Error: Document directory not found - {e}")
                sys.exit(1)
            except Exception as e:
                print(f"Error during ingestion: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        if args.ask:
            if not os.path.exists("chroma_db/chroma.sqlite3"):
                print("Vector DB not found. Run with --ingest first")
                sys.exit(1)
            
            try:
                ask(args.ask)
            except Exception as e:
                print(f"Error during query: {e}")
                traceback.print_exc()
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
```

### 3. Progress Indicators

```python
# run_with_progress.py
from rich.console import Console
from rich.progress import track
import time

console = Console()

def main():
    # ... argument parsing
    
    if args.ingest:
        with console.status("[bold green]Ingesting documents...") as status:
            ingest()
        console.print("✓ [bold green]Ingestion complete!")
    
    if args.ask:
        if not os.path.exists("chroma_db/chroma.sqlite3"):
            console.print("[bold red]Vector DB not found. Run with --ingest first")
            return
        
        console.print(f"[bold cyan]Question:[/bold cyan] {args.ask}")
        
        with console.status("[bold yellow]Thinking..."):
            result, route, scores = execute(args.ask)
        
        console.print(f"[bold green]Answer:[/bold green] {result}")
        console.print(f"[dim]Route: {route.value} | Confidence: {scores[route.value]:.2%}[/dim]")
```

## Alternative Approaches

### 1. Click-Based CLI

**Pros**:
- More intuitive command structure
- Better help messages
- Easier to extend
- Supports command groups

**Cons**:
- Additional dependency
- Slightly more code

**Implementation**:
```python
import click
from cortex.ingest import ingest
from cortex.query import ask

@click.group()
def cli():
    """CORTEX - Document Q&A Assistant"""
    pass

@cli.command()
def ingest_docs():
    """Ingest documents into vector database"""
    click.echo("Ingesting documents...")
    ingest()
    click.echo("✓ Complete!")

@cli.command()
@click.argument('question')
def ask_question(question):
    """Ask a question to CORTEX"""
    if not os.path.exists("chroma_db/chroma.sqlite3"):
        click.echo("Vector DB not found. Run 'ingest-docs' first")
        return
    
    ask(question)

if __name__ == "__main__":
    cli()

# Usage:
# python run_click.py ingest-docs
# python run_click.py ask-question "What is the revenue?"
```

### 2. Typer-Based CLI

**Pros**:
- Type hints for automatic validation
- Modern Python syntax
- Great auto-completion
- Less boilerplate

**Cons**:
- Additional dependency
- Learning curve for advanced features

**Implementation**:
```python
import typer
from typing import Optional

app = typer.Typer()

@app.command()
def ingest(
    force: bool = typer.Option(False, "--force", "-f", help="Force re-ingestion")
):
    """Ingest documents into vector database"""
    typer.echo("Ingesting documents...")
    from cortex.ingest import ingest as do_ingest
    do_ingest()
    typer.secho("✓ Complete!", fg=typer.colors.GREEN)

@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    verbose: bool = typer.Option(False, "--verbose", "-v")
):
    """Ask a question to CORTEX"""
    if not os.path.exists("chroma_db/chroma.sqlite3"):
        typer.secho("Vector DB not found. Run 'ingest' first", fg=typer.colors.RED)
        raise typer.Exit(1)
    
    from cortex.router import execute
    result, route, scores = execute(question)
    
    typer.echo(result)
    if verbose:
        typer.echo(f"\nRoute: {route.value}")
        typer.echo(f"Confidence: {scores[route.value]:.2%}")

if __name__ == "__main__":
    app()
```

### 3. REST API Only

**Pros**:
- Language-agnostic access
- Easy to integrate with web/mobile
- Scalable
- Can serve multiple clients

**Cons**:
- Requires server setup
- More complex deployment
- Additional dependencies

Already shown in "Web API Wrapper" example above.

### 4. Config File Driven

**Pros**:
- No command-line arguments needed
- Easy to version control settings
- Good for scheduled tasks
- Repeatable workflows

**Cons**:
- Less flexible for one-off queries
- Requires config file management

**Implementation**:
```python
# run_config.py
import yaml

def main():
    with open('cortex_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    if config.get('ingest', {}).get('enabled'):
        ingest()
    
    for question in config.get('queries', []):
        result, route, scores = execute(question)
        print(f"Q: {question}")
        print(f"A: {result}\n")
```

**cortex_config.yaml**:
```yaml
ingest:
  enabled: false

queries:
  - "What is the revenue?"
  - "Who are the authors?"
  - "What are the conclusions?"
```

## Testing

### 1. Unit Tests

```python
# tests/test_run.py
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

class TestRunScript(unittest.TestCase):
    @patch('cortex.ingest.ingest')
    def test_ingest_called(self, mock_ingest):
        """Test that ingest is called with --ingest flag"""
        sys.argv = ['run.py', '--ingest']
        from run import main
        main()
        mock_ingest.assert_called_once()
    
    @patch('cortex.query.ask')
    @patch('os.path.exists')
    def test_ask_with_db(self, mock_exists, mock_ask):
        """Test asking when DB exists"""
        mock_exists.return_value = True
        sys.argv = ['run.py', '--ask', 'test question']
        from run import main
        main()
        mock_ask.assert_called_once_with('test question')
    
    @patch('os.path.exists')
    def test_ask_without_db(self, mock_exists):
        """Test error message when DB doesn't exist"""
        mock_exists.return_value = False
        sys.argv = ['run.py', '--ask', 'test question']
        # Should print error, not call ask()
```

### 2. Integration Tests

```python
# tests/test_integration.py
import subprocess
import os

def test_full_workflow():
    """Test complete ingest and query workflow"""
    
    # Clean up any existing DB
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")
    
    # Run ingestion
    result = subprocess.run(
        ["python", "run.py", "--ingest"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert os.path.exists("chroma_db/chroma.sqlite3")
    
    # Run query
    result = subprocess.run(
        ["python", "run.py", "--ask", "test question"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert len(result.stdout) > 0
```

## Best Practices

### 1. Error Messages

✅ **Good**:
```python
if not os.path.exists("chroma_db/chroma.sqlite3"):
    print("Vector DB not found. Run with --ingest first")
    print("Example: python run.py --ingest")
    sys.exit(1)
```

❌ **Bad**:
```python
if not os.path.exists("chroma_db/chroma.sqlite3"):
    print("Error")  # Vague, not helpful
```

### 2. Logging

✅ **Good**:
```python
logging.basicConfig(
    filename="query.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info(f"Query: {args.ask}")
logging.info(f"Route: {route.value}, Confidence: {scores[route.value]}")
```

### 3. Exit Codes

```python
# Success
sys.exit(0)

# User error (wrong arguments, missing DB)
sys.exit(1)

# System error (exceptions, crashes)
sys.exit(2)
```

## Conclusion

The `run.py` script provides a simple, effective CLI for CORTEX. It's ideal for:

- **Quick testing and development**
- **Simple deployments**
- **Script automation**
- **Learning the system**

### When to Extend

Consider more sophisticated alternatives when you need:
- **Web interface** → FastAPI/Flask REST API
- **Rich CLI** → Click or Typer
- **GUI** → Tkinter or web-based UI
- **Multi-user** → Web application with authentication
- **Scheduled tasks** → Config-driven automation

The simple argparse-based approach is perfect for getting started and can easily be extended as requirements grow.