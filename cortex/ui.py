import os
import sys
import logging
import queue
import threading
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.markdown import Markdown
from rich.table import Table
from rich.align import Align
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML

from cortex.ingest import ingest
from cortex.query import load_qa_chain
from cortex.router import execute
from cortex.memory import ConversationMemory
from cortex.streaming import StreamHandler
from cortex.logo import get_logo

# Initialize console
console = Console()

# Initialize memory
memory = ConversationMemory()

# Command completer
COMMANDS = ['help', 'exit', 'quit', 'clear', 'history', 'ingest', 'status']
completer = WordCompleter(COMMANDS, ignore_case=True)

# Prompt style - using prompt_toolkit's style format
prompt_style = Style.from_dict({
    'prompt': '#00ffff bold',  # Cyan bold for CORTEX
    'arrow': '#00ff00',        # Green for arrow
})


def print_logo(animate=False):
    """Display the CORTEX logo with styling"""
    from cortex.logo import CORTEX_LOGO_TEXT
    
    # Work with the original logo text to preserve structure
    # Don't use strip() as it removes the initial newline which affects line numbering
    logo_lines = CORTEX_LOGO_TEXT.split('\n')
    # Filter out empty lines at start/end but preserve structure
    # Remove leading empty lines
    while logo_lines and not logo_lines[0].strip():
        logo_lines.pop(0)
    # Remove trailing empty lines
    while logo_lines and not logo_lines[-1].strip():
        logo_lines.pop()
    
    # Find the leftmost character position (minimum leading whitespace)
    min_leading = float('inf')
    for line in logo_lines:
        if line.strip():  # Only consider non-empty lines
            leading = len(line) - len(line.lstrip())
            min_leading = min(min_leading, leading)
    
    if min_leading == float('inf'):
        min_leading = 0
    
    # Normalize: remove minimum leading whitespace from all lines
    # This aligns all lines to the leftmost character while preserving relative positions
    normalized_lines = []
    for line in logo_lines:
        if line.strip():
            # Remove min_leading spaces from the start
            normalized_line = line[min_leading:] if len(line) > min_leading else line.lstrip()
            normalized_lines.append(normalized_line)
        else:
            normalized_lines.append('')
    
    # Find the maximum width of normalized lines
    max_width = max(len(line) for line in normalized_lines) if normalized_lines else 0
    
    # Get available width in Panel (terminal width - borders - padding)
    # Panel: 2 char borders + 4 char horizontal padding = 6 chars
    terminal_width = console.width if hasattr(console, 'width') and console.width else 80
    available_width = terminal_width - 6
    
    # Calculate padding to center the block
    padding = max(0, (available_width - max_width) // 2)
    
    # Add consistent padding to all lines to center the block
    padded_lines = [' ' * padding + line for line in normalized_lines]
    centered_logo_text = '\n'.join(padded_lines)
    
    # Now apply Rich styling to the centered text
    # Rebuild the logo with proper styling
    styled_logo = Text()
    centered_lines = centered_logo_text.split('\n')
    
    for i, line in enumerate(centered_lines):
        if '██' in line:
            # Main CORTEX text - bright cyan bold
            styled_logo.append(line, style="bold bright_cyan")
        elif 'Think. Retrieve. Answer.' in line:
            # Tagline - special gradient
            parts = line.split('Think')
            styled_logo.append(parts[0], style="cyan")
            styled_logo.append('Think', style="bold bright_green")
            
            remaining = 'Think'.join(parts[1:])
            parts2 = remaining.split('Retrieve')
            styled_logo.append(parts2[0], style="cyan")
            styled_logo.append('Retrieve', style="bold bright_blue")
            
            remaining2 = 'Retrieve'.join(parts2[1:])
            parts3 = remaining2.split('Answer')
            styled_logo.append(parts3[0], style="cyan")
            styled_logo.append('Answer', style="bold bright_magenta")
            styled_logo.append('Answer'.join(parts3[1:]), style="cyan")
        else:
            # Empty or separator lines
            styled_logo.append(line, style="cyan")
        
        if i < len(centered_lines) - 1:
            styled_logo.append('\n')
    
    if animate:
        # Animate logo appearance
        for i in range(1, len(centered_lines) + 1):
            partial_lines = centered_lines[:i]
            partial_text = '\n'.join(partial_lines)
            
            # Apply styling to partial text
            partial_styled = Text()
            for j, line in enumerate(partial_lines):
                if '██' in line:
                    partial_styled.append(line, style="bold bright_cyan")
                elif 'Think. Retrieve. Answer.' in line:
                    parts = line.split('Think')
                    partial_styled.append(parts[0], style="cyan")
                    partial_styled.append('Think', style="bold bright_green")
                    remaining = 'Think'.join(parts[1:])
                    parts2 = remaining.split('Retrieve')
                    partial_styled.append(parts2[0], style="cyan")
                    partial_styled.append('Retrieve', style="bold bright_blue")
                    remaining2 = 'Retrieve'.join(parts2[1:])
                    parts3 = remaining2.split('Answer')
                    partial_styled.append(parts3[0], style="cyan")
                    partial_styled.append('Answer', style="bold bright_magenta")
                    partial_styled.append('Answer'.join(parts3[1:]), style="cyan")
                else:
                    partial_styled.append(line, style="cyan")
                if j < len(partial_lines) - 1:
                    partial_styled.append('\n')
            
            console.print(Panel(
                partial_styled,
                border_style="bright_cyan",
                padding=(1, 2),
                title="[bold bright_cyan]CORTEX v1.0[/bold bright_cyan]",
                subtitle="[dim bright_cyan]Neural Interface | Local AI Knowledge Assistant[/dim bright_cyan]"
            ))
            time.sleep(0.05)
            if i < len(centered_lines):
                console.clear()
    else:
        console.print(Panel(
            styled_logo,
            border_style="bright_cyan",
            padding=(1, 2),
            title="[bold bright_cyan]CORTEX v1.0[/bold bright_cyan]",
            subtitle="[dim bright_cyan]Neural Interface | Local AI Knowledge Assistant[/dim bright_cyan]"
        ))
    console.print()


def print_welcome():
    """Display welcome message"""
    welcome_text = Text()
    welcome_text.append("Welcome to ", style="white")
    welcome_text.append("CORTEX", style="bold cyan")
    welcome_text.append(" - Your local AI assistant", style="white")
    
    console.print(Panel(
        welcome_text,
        border_style="green",
        padding=(0, 1)
    ))
    console.print()


def print_help():
    """Display help information"""
    help_table = Table(title="[bold cyan]CORTEX Commands[/bold cyan]", show_header=True, header_style="bold magenta")
    help_table.add_column("Command", style="cyan", no_wrap=True)
    help_table.add_column("Description", style="white")
    
    help_table.add_row("help", "Show this help message")
    help_table.add_row("exit / quit", "Exit CORTEX")
    help_table.add_row("clear", "Clear the screen")
    help_table.add_row("history", "Show conversation history")
    help_table.add_row("ingest", "Ingest documents into knowledge base")
    help_table.add_row("status", "Show system status")
    help_table.add_row("<question>", "Ask CORTEX a question")
    
    console.print(help_table)
    console.print()


def check_db_status():
    """Check if vector database exists"""
    db_path = Path("chroma_db/chroma.sqlite3")
    return db_path.exists()


def show_status():
    """Display system status"""
    status_table = Table(title="[bold cyan]System Status[/bold cyan]", show_header=True, header_style="bold magenta")
    status_table.add_column("Component", style="cyan", no_wrap=True)
    status_table.add_column("Status", style="white")
    
    db_exists = check_db_status()
    status_table.add_row(
        "Vector Database",
        "[green]✓ Ready[/green]" if db_exists else "[red]✗ Not Found[/red]"
    )
    
    status_table.add_row(
        "Conversation History",
        f"[green]✓ {len(memory.history)} items[/green]"
    )
    
    status_table.add_row(
        "Memory",
        f"[green]✓ {memory.max_items} max items[/green]"
    )
    
    console.print(status_table)
    console.print()


def show_history():
    """Display conversation history"""
    if not memory.history:
        console.print("[yellow]No conversation history yet.[/yellow]")
        return
    
    history_table = Table(title="[bold cyan]Conversation History[/bold cyan]", show_header=True, header_style="bold magenta")
    history_table.add_column("#", style="cyan", no_wrap=True, width=4)
    history_table.add_column("Time", style="dim white", no_wrap=True, width=10)
    history_table.add_column("Query", style="white", overflow="fold")
    
    for idx, item in enumerate(memory.history[-10:], 1):  # Show last 10
        history_table.add_row(
            str(idx),
            item.timestamp,
            item.query[:60] + "..." if len(item.query) > 60 else item.query
        )
    
    console.print(history_table)
    console.print()


def ingest_documents():
    """Ingest documents with progress display"""
    console.print("[cyan]Starting document ingestion...[/cyan]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Ingesting documents...", total=None)
            ingest()
            progress.update(task, completed=True)
        
        console.print("[green]✓ Document ingestion completed successfully![/green]")
    except Exception as e:
        console.print(f"[red]✗ Error during ingestion: {str(e)}[/red]")
    
    console.print()


def stream_response(query: str):
    """Stream response with live updates"""
    response_buffer = ""
    token_queue = queue.Queue()
    stream_complete = threading.Event()
    
    def on_token(token: str):
        token_queue.put(token)
    
    stream_handler = StreamHandler(on_token)
    
    # Create a renderable for live updates
    class StreamingResponse:
        def __init__(self):
            self.buffer = ""
        
        def __rich__(self):
            if not self.buffer.strip():
                return Spinner("dots", text="[cyan]CORTEX is thinking...")
            return Markdown(self.buffer)
        
        def update(self, text: str):
            self.buffer = text
    
    streaming_response = StreamingResponse()
    
    # Execute query in background thread
    def execute_query():
        try:
            result = execute(query, callbacks=[stream_handler])
            # Put final result if streaming didn't capture everything
            if result:
                token_queue.put(("final", result))
        except Exception as e:
            token_queue.put(("error", str(e)))
        finally:
            stream_complete.set()
    
    query_thread = threading.Thread(target=execute_query, daemon=True)
    query_thread.start()
    
    # Display streaming response
    try:
        with Live(
            Panel(streaming_response, title="[bold green]CORTEX Response[/bold green]", border_style="green", padding=(1, 2)),
            console=console,
            refresh_per_second=20,
            transient=False
        ) as live:
            while not stream_complete.is_set() or not token_queue.empty():
                try:
                    item = token_queue.get(timeout=0.1)
                    if isinstance(item, tuple):
                        if item[0] == "error":
                            raise Exception(item[1])
                        elif item[0] == "final":
                            response_buffer = item[1]
                    else:
                        response_buffer += item
                    streaming_response.update(response_buffer)
                    live.update(Panel(streaming_response, title="[bold green]CORTEX Response[/bold green]", border_style="green", padding=(1, 2)))
                except queue.Empty:
                    continue
        
        query_thread.join(timeout=2)
        
        # Display final answer with sources
        if response_buffer:
            # Final answer already shown during streaming; keep sources/history only
            # console.print(Panel(
            #     Markdown(response_buffer),
            #     title="[bold green]CORTEX Response[/bold green]",
            #     border_style="green",
            #     padding=(1, 2)
            # ))
            
            # Show sources if available
            try:
                chain, retriever = load_qa_chain()
                docs = retriever._get_relevant_documents(query, run_manager=None)
                if docs:
                    sources_text = "\n".join([
                        f"• {doc.metadata.get('source', 'unknown')} (page {doc.metadata.get('page', 'N/A')})"
                        for doc in docs[:5]
                    ])
                    console.print(Panel(
                        sources_text,
                        title="[bold cyan]Sources[/bold cyan]",
                        border_style="cyan",
                        padding=(0, 1)
                    ))
            except:
                pass
            
            memory.add(query, response_buffer)
    
    except Exception as e:
        console.print(f"[red]✗ Error: {str(e)}[/red]")
        console.print_exception()
    
    console.print()


def process_query(query: str):
    """Process a user query"""
    query = query.strip()
    
    if not query:
        return
    
    # Handle commands
    if query.lower() in ['exit', 'quit']:
        console.print("[yellow]Goodbye![/yellow]")
        sys.exit(0)
    
    elif query.lower() == 'help':
        print_help()
        return
    
    elif query.lower() == 'clear':
        console.clear()
        print_logo()
        return
    
    elif query.lower() == 'history':
        show_history()
        return
    
    elif query.lower() == 'ingest':
        ingest_documents()
        return
    
    elif query.lower() == 'status':
        show_status()
        return
    
    # Check if database exists for RAG queries
    if not check_db_status():
        console.print("[yellow]⚠ Vector database not found. Some queries may not work properly.[/yellow]")
        console.print("[dim]Run 'ingest' command to create the knowledge base.[/dim]")
        console.print()
    
    # Process as a question
    stream_response(query)


def interactive_mode():
    """Run interactive CLI mode"""
    print_logo(animate=False)  # Set to True for animated logo
    print_welcome()
    
    # Check initial status
    if not check_db_status():
        console.print("[yellow]⚠ Warning: Vector database not found.[/yellow]")
        console.print("[dim]Run 'ingest' command to create the knowledge base.[/dim]")
        console.print()
    
    # Setup logging
    logging.basicConfig(filename="query.log", level=logging.INFO)
    
    # Create prompt session with history
    history_file = Path.home() / ".cortex_history"
    session = PromptSession(
        history=FileHistory(str(history_file)),
        auto_suggest=AutoSuggestFromHistory(),
        completer=completer,
        style=prompt_style
    )
    
    # Main loop
    while True:
        try:
            # Get user input with custom prompt using HTML formatting for prompt_toolkit
            prompt_text = HTML('<prompt>CORTEX</prompt><arrow> > </arrow>')
            
            query = session.prompt(prompt_text)
            
            if query.strip():
                logging.info(f"Query asked: {query}")
                process_query(query)
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
            continue
        except EOFError:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]✗ Unexpected error: {str(e)}[/red]")
            console.print_exception()


if __name__ == "__main__":
    interactive_mode()