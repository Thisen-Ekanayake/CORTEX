# Logo Module Documentation

## Overview

The `logo.py` module provides ASCII art branding and visual identity for the CORTEX AI assistant. It offers multiple styled variations of the CORTEX logo with rich terminal formatting, colors, and effects using the `rich` library.

## Purpose

This module serves as the visual branding component by:

1. **Establishing brand identity** through consistent ASCII art
2. **Providing visual appeal** in terminal/CLI interfaces
3. **Supporting multiple display styles** for different contexts
4. **Enhancing user experience** with colorful, professional output
5. **Creating memorable first impressions** when the application launches

## Dependencies

```python
from rich.text import Text      # For styled text rendering
from rich.console import Console  # For terminal output (demo only)
from rich.panel import Panel      # For bordered displays (demo only)
from rich.align import Align      # For centering content (demo only)
```

**Required Package**:
```bash
pip install rich
```

## Components

### 1. ASCII Art Logo: `CORTEX_LOGO_TEXT`

The base ASCII art design using box-drawing characters.

```
        ██████╗ ██████╗ ██████╗ ████████╗███████╗██╗  ██╗
       ██╔════╝██╔═══██╗██╔══██╗╚══██╔══╝██╔════╝╚██╗██╔╝
       ██║     ██║   ██║██████╔╝   ██║   █████╗   ╚███╔╝ 
       ██║     ██║   ██║██╔══██╗   ██║   ██╔══╝   ██╔██╗ 
       ╚██████╗╚██████╔╝██║  ██║   ██║   ███████╗██╔╝ ██╗
        ╚═════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝

              ════  Think. Retrieve. Answer.  ════
```

**Design Elements**:
- **Block characters (██)**: Bold, prominent lettering
- **Box-drawing characters (╔═╗║╚╝)**: Clean, technical aesthetic
- **Tagline**: "Think. Retrieve. Answer." - encapsulates core functionality
- **Symmetry**: Centered design for visual balance

### 2. Main Functions

#### `get_logo(compact=False, futuristic=True, colored=True)`

Returns the CORTEX logo with customizable styling.

**Parameters**:
- `compact` (bool): Use compact version with tagline
  - Default: `False`
  - Currently, both versions are the same
  - Reserved for future compact variant
- `futuristic` (bool): Use futuristic style
  - Default: `True`
  - Currently unused (placeholder for future styles)
- `colored` (bool): Return colored Rich Text object vs plain string
  - Default: `True`
  - `True`: Returns `rich.text.Text` object with colors
  - `False`: Returns plain string

**Returns**:
- `rich.text.Text` if `colored=True`
- `str` if `colored=False`

**Color Scheme** (when `colored=True`):
- **Main CORTEX text**: Bright cyan, bold
- **Tagline keywords**:
  - "Think": Bright green, bold
  - "Retrieve": Bright blue, bold
  - "Answer": Bright magenta, bold
  - Decorative elements: Cyan

**Usage Examples**:

```python
from logo import get_logo
from rich.console import Console

console = Console()

# Basic colored logo
logo = get_logo()
console.print(logo)

# Plain text version (no colors)
logo_plain = get_logo(colored=False)
print(logo_plain)

# With panel border
from rich.panel import Panel
logo = get_logo(colored=True)
console.print(Panel(logo, border_style="cyan"))
```

#### `get_logo_gradient(compact=False)`

Returns logo with a vertical color gradient effect (cyan to blue).

**Parameters**:
- `compact` (bool): Use compact version (currently unused)

**Returns**:
- `rich.text.Text` object with gradient styling

**Gradient Colors** (top to bottom):
1. Bright cyan
2. Cyan
3. Bright blue
4. Blue

**Usage**:
```python
from logo import get_logo_gradient
from rich.console import Console

console = Console()
logo = get_logo_gradient()
console.print(logo)
```

#### `get_logo_neon(compact=False)`

Returns logo with neon-style coloring for a vibrant, futuristic appearance.

**Parameters**:
- `compact` (bool): Use compact version (currently unused)

**Returns**:
- `rich.text.Text` object with neon styling

**Color Scheme**:
- **Main CORTEX text**: Bright cyan, bold (neon effect)
- **Tagline**: Bright green, bold
- **Other elements**: Dim cyan

**Usage**:
```python
from logo import get_logo_neon
from rich.console import Console

console = Console()
logo = get_logo_neon()
console.print(logo)
```

## Integration Examples

### 1. CLI Application Startup

```python
# main.py or cli.py
from logo import get_logo
from rich.console import Console
from rich.panel import Panel
from rich.align import Align

def show_welcome_screen():
    """Display welcome screen with CORTEX logo"""
    console = Console()
    
    logo = get_logo(colored=True)
    
    console.print(Panel(
        Align.center(logo),
        border_style="bright_cyan",
        padding=(1, 2),
        title="[bold bright_cyan]CORTEX v1.0[/bold bright_cyan]",
        subtitle="[bright_cyan]Privacy-First AI Assistant[/bright_cyan]"
    ))
    
    console.print("\n[bold cyan]Welcome to CORTEX![/bold cyan]")
    console.print("[dim]Type 'help' for available commands[/dim]\n")

if __name__ == "__main__":
    show_welcome_screen()
    # Start application...
```

### 2. Help Screen Integration

```python
from logo import get_logo
from rich.console import Console
from rich.table import Table

def show_help():
    """Display help screen with logo header"""
    console = Console()
    
    # Show compact logo
    logo = get_logo(compact=True)
    console.print(logo)
    console.print()
    
    # Create help table
    table = Table(title="Available Commands", border_style="cyan")
    table.add_column("Command", style="bright_cyan", no_wrap=True)
    table.add_column("Description", style="white")
    
    table.add_row("ask [query]", "Ask CORTEX a question")
    table.add_row("search [query]", "Search documents")
    table.add_row("help", "Show this help message")
    table.add_row("exit", "Exit CORTEX")
    
    console.print(table)
```

### 3. Loading Screen

```python
from logo import get_logo_neon
from rich.console import Console
from rich.spinner import Spinner
from rich.live import Live
import time

def show_loading_screen(task_name="Initializing CORTEX"):
    """Display animated loading screen with logo"""
    console = Console()
    
    # Show neon logo
    logo = get_logo_neon()
    console.print(logo)
    console.print()
    
    # Animated spinner
    with console.status(f"[bold bright_cyan]{task_name}...", spinner="dots"):
        time.sleep(2)  # Simulate loading
    
    console.print("[bold bright_green]✓[/bold bright_green] Ready!\n")
```

### 4. Error Screen

```python
from logo import get_logo
from rich.console import Console
from rich.panel import Panel

def show_error_screen(error_message):
    """Display error with minimal logo"""
    console = Console()
    
    # Plain logo (no colors for error context)
    logo = get_logo(colored=False)
    console.print(logo)
    
    # Error panel
    console.print(Panel(
        f"[bold red]Error:[/bold red] {error_message}",
        border_style="red",
        padding=(1, 2)
    ))
```

### 5. Interactive REPL

```python
from logo import get_logo_gradient
from rich.console import Console
from rich.prompt import Prompt

def start_repl():
    """Start interactive REPL with logo"""
    console = Console()
    
    # Show gradient logo on startup
    logo = get_logo_gradient()
    console.print(logo)
    console.print()
    
    while True:
        try:
            query = Prompt.ask("[bold cyan]CORTEX[/bold cyan]")
            
            if query.lower() in ['exit', 'quit']:
                console.print("[dim]Goodbye![/dim]")
                break
            
            # Process query...
            console.print(f"[green]Processing:[/green] {query}\n")
            
        except KeyboardInterrupt:
            console.print("\n[dim]Use 'exit' to quit[/dim]")
        except EOFError:
            break
```

## Customization Guide

### 1. Creating Custom Color Schemes

```python
from rich.text import Text
from logo import CORTEX_LOGO_TEXT

def get_logo_custom_colors(primary_color="yellow", accent_color="red"):
    """Create logo with custom colors"""
    logo_text = CORTEX_LOGO_TEXT
    lines = logo_text.strip().split('\n')
    
    text = Text()
    for i, line in enumerate(lines):
        if '██' in line:
            # Main text
            text.append(line, style=f"bold {primary_color}")
        elif 'Think. Retrieve. Answer.' in line:
            # Tagline
            text.append(line, style=f"bold {accent_color}")
        else:
            # Other lines
            text.append(line, style=primary_color)
        
        if i < len(lines) - 1:
            text.append('\n')
    
    return text

# Usage
logo = get_logo_custom_colors(primary_color="magenta", accent_color="yellow")
```

### 2. Adding Animation

```python
from rich.console import Console
from rich.live import Live
from logo import get_logo
import time

def animate_logo_fade_in():
    """Animate logo with fade-in effect"""
    console = Console()
    logo_lines = get_logo(colored=False).split('\n')
    
    with Live(console=console, refresh_per_second=10) as live:
        for i in range(len(logo_lines)):
            # Show lines progressively
            partial_logo = '\n'.join(logo_lines[:i+1])
            live.update(f"[cyan]{partial_logo}[/cyan]")
            time.sleep(0.1)
```

### 3. Compact Version Design

```python
# Add to logo.py
CORTEX_LOGO_COMPACT = """
   ╔═══╗╔═══╗╔═══╗╔════╗╔═══╗╔╗ ╔╗
   ║╔═╗║║╔═╗║║╔═╗║╚═╗╔═╝║╔══╝╚╗╔╝║
   ║║ ╚╝║║ ║║║╚═╝║  ║║  ║╚══╗ ╚╝ ║
   ║║ ╔╗║║ ║║║╔╗╔╝  ║║  ║╔══╝ ╔╗ ║
   ║╚═╝║║╚═╝║║║║╚╗  ║║  ║╚══╗╔╝╚╗║
   ╚═══╝╚═══╝╚╝╚═╝  ╚╝  ╚═══╝╚══╝║
"""

def get_logo(compact=False, futuristic=True, colored=True):
    """Updated to support compact version"""
    logo_text = CORTEX_LOGO_COMPACT if compact else CORTEX_LOGO_TEXT
    # ... rest of implementation
```

### 4. Theme-Based Selection

```python
# logo_themes.py
from logo import get_logo, get_logo_gradient, get_logo_neon

THEMES = {
    'default': lambda: get_logo(),
    'gradient': lambda: get_logo_gradient(),
    'neon': lambda: get_logo_neon(),
    'minimal': lambda: get_logo(colored=False),
}

def get_themed_logo(theme_name='default'):
    """Get logo based on theme"""
    theme_func = THEMES.get(theme_name, THEMES['default'])
    return theme_func()

# Usage
logo = get_themed_logo('neon')
```

### 5. Seasonal/Event Variants

```python
from datetime import datetime
from logo import get_logo
from rich.text import Text

def get_seasonal_logo():
    """Return logo styled for current season/holiday"""
    today = datetime.now()
    month = today.month
    
    # Holiday themes
    if month == 12:  # December - Winter/Holiday
        return get_logo_custom_colors("bright_white", "bright_red")
    elif month in [10, 11]:  # Fall
        return get_logo_custom_colors("yellow", "red")
    elif month in [3, 4, 5]:  # Spring
        return get_logo_custom_colors("bright_green", "bright_yellow")
    elif month in [6, 7, 8]:  # Summer
        return get_logo_custom_colors("bright_yellow", "bright_cyan")
    else:  # Default
        return get_logo()
```

## Alternative Approaches

### 1. FIGlet-Based Logo

**Pros**:
- Large library of fonts
- Easy to generate different styles
- No manual ASCII art required

**Cons**:
- Additional dependency
- Less control over exact appearance
- May not render well in all terminals

**Implementation**:
```python
from pyfiglet import Figlet

def get_logo_figlet(font='slant'):
    """Generate logo using FIGlet"""
    fig = Figlet(font=font)
    logo_text = fig.renderText('CORTEX')
    
    # Add tagline
    logo_with_tagline = logo_text + '\n    Think. Retrieve. Answer.\n'
    
    return logo_with_tagline

# Popular fonts: slant, banner, big, block, digital
```

### 2. Image-Based ASCII Art

**Pros**:
- Can convert any image to ASCII
- Highly detailed possibilities
- Supports gradients and shading

**Cons**:
- Requires image processing libraries
- Larger file size
- More complex to maintain

**Implementation**:
```python
from PIL import Image
import ascii_magic

def get_logo_from_image(image_path='cortex_logo.png'):
    """Convert image to ASCII art"""
    output = ascii_magic.from_image_file(
        image_path,
        columns=80,
        mode=ascii_magic.Modes.ASCII
    )
    return output
```

### 3. Dynamic Text-Based Logo

**Pros**:
- Smaller, cleaner
- Faster to render
- Works in any terminal

**Cons**:
- Less visually impressive
- Harder to brand
- Limited styling options

**Implementation**:
```python
from rich.text import Text

def get_logo_minimal():
    """Minimal text-based logo"""
    text = Text()
    text.append("C", style="bold bright_cyan")
    text.append("O", style="bold bright_blue")
    text.append("R", style="bold bright_green")
    text.append("T", style="bold bright_yellow")
    text.append("E", style="bold bright_magenta")
    text.append("X", style="bold bright_red")
    text.append("\n")
    text.append("Think. Retrieve. Answer.", style="italic dim cyan")
    return text
```

### 4. SVG/Unicode Box Drawing

**Pros**:
- Cleaner appearance
- Better terminal compatibility
- Smaller character footprint

**Cons**:
- Limited font support
- May not render in all terminals
- Less dramatic visual impact

**Implementation**:
```python
CORTEX_LOGO_UNICODE = """
┌─┐┌─┐┌─┐┌┬┐┌─┐─┐ ┬
│  │ │├┬┘ │ ├┤ ┌┴┬┘
└─┘└─┘┴└─ ┴ └─┘┴ └─
"""
```

### 5. Rich Console Markup Only

**Pros**:
- No ASCII art needed
- Always renders correctly
- Easy to maintain

**Cons**:
- Less distinctive
- Minimal visual impact
- Not memorable

**Implementation**:
```python
from rich.console import Console

def show_logo_markup():
    """Logo using Rich markup only"""
    console = Console()
    console.print("\n[bold bright_cyan]╔════════════════════╗[/bold bright_cyan]")
    console.print("[bold bright_cyan]║[/bold bright_cyan]   [bold bright_white on bright_cyan] CORTEX [/bold bright_white on bright_cyan]   [bold bright_cyan]║[/bold bright_cyan]")
    console.print("[bold bright_cyan]╚════════════════════╝[/bold bright_cyan]")
    console.print("[dim cyan]Think. Retrieve. Answer.[/dim cyan]\n")
```

## Performance Considerations

### 1. Lazy Loading

```python
# logo.py
_cached_logos = {}

def get_logo_cached(style='default'):
    """Cache logos to avoid regeneration"""
    if style not in _cached_logos:
        if style == 'gradient':
            _cached_logos[style] = get_logo_gradient()
        elif style == 'neon':
            _cached_logos[style] = get_logo_neon()
        else:
            _cached_logos[style] = get_logo()
    
    return _cached_logos[style]
```

### 2. Conditional Display

```python
import os

def should_show_logo():
    """Determine if logo should be displayed"""
    # Don't show logo if:
    # - Not in TTY (piped output)
    # - Terminal too narrow
    # - Explicitly disabled
    
    if not os.isatty(1):  # Not a terminal
        return False
    
    try:
        columns = os.get_terminal_size().columns
        if columns < 70:  # Logo needs at least 70 chars
            return False
    except:
        return False
    
    if os.environ.get('CORTEX_NO_LOGO'):
        return False
    
    return True

# Usage
if should_show_logo():
    console.print(get_logo())
```

### 3. Fast Rendering

```python
from rich.console import Console

def print_logo_fast():
    """Optimized logo rendering"""
    console = Console()
    
    # Use no_wrap for faster rendering
    console.print(get_logo(), no_wrap=True)
```

## Testing

### 1. Visual Testing

```python
# tests/test_logo_visual.py
from logo import get_logo, get_logo_gradient, get_logo_neon
from rich.console import Console

def test_all_logo_variants():
    """Visual test of all logo variants"""
    console = Console()
    
    variants = [
        ("Standard", get_logo()),
        ("Gradient", get_logo_gradient()),
        ("Neon", get_logo_neon()),
        ("Plain", get_logo(colored=False)),
    ]
    
    for name, logo in variants:
        console.print(f"\n[bold]{name}:[/bold]")
        console.print(logo)
        console.print()
```

### 2. Unit Testing

```python
import unittest
from logo import get_logo, CORTEX_LOGO_TEXT

class TestLogo(unittest.TestCase):
    def test_logo_contains_cortex(self):
        """Verify logo contains CORTEX text"""
        logo = get_logo(colored=False)
        self.assertIn('██', logo)
    
    def test_logo_has_tagline(self):
        """Verify tagline is present"""
        logo = get_logo(colored=False)
        self.assertIn('Think. Retrieve. Answer.', logo)
    
    def test_colored_logo_returns_text_object(self):
        """Verify colored logo returns Rich Text object"""
        from rich.text import Text
        logo = get_logo(colored=True)
        self.assertIsInstance(logo, Text)
    
    def test_plain_logo_returns_string(self):
        """Verify plain logo returns string"""
        logo = get_logo(colored=False)
        self.assertIsInstance(logo, str)
```

### 3. Terminal Compatibility Test

```python
import os
from rich.console import Console

def test_terminal_rendering():
    """Test logo renders in different terminal modes"""
    console = Console()
    
    # Test with different color depths
    for color_system in ['auto', 'standard', 'truecolor', None]:
        test_console = Console(color_system=color_system)
        logo = get_logo()
        
        print(f"\nTesting with color_system='{color_system}':")
        test_console.print(logo)
```

## Best Practices

### 1. Logo Display Guidelines

✅ **DO**:
- Show logo on application startup
- Use appropriate variant for context (neon for splash, plain for errors)
- Center logo in panels or screens
- Include version number in panel title
- Use consistent border styling

❌ **DON'T**:
- Show logo before every command (too repetitive)
- Use colored logo in log files
- Display logo in piped/non-TTY output
- Show logo on narrow terminals (< 70 columns)

### 2. Color Consistency

```python
# Define color palette
CORTEX_COLORS = {
    'primary': 'bright_cyan',
    'secondary': 'cyan',
    'accent1': 'bright_green',
    'accent2': 'bright_blue',
    'accent3': 'bright_magenta',
    'text': 'white',
    'dim': 'dim cyan'
}

# Use throughout application
from rich.panel import Panel

Panel(logo, border_style=CORTEX_COLORS['primary'])
```

### 3. Accessibility

```python
def get_accessible_logo():
    """Logo with high contrast for accessibility"""
    from rich.text import Text
    
    text = Text()
    logo_lines = CORTEX_LOGO_TEXT.strip().split('\n')
    
    for i, line in enumerate(lines):
        # Use high-contrast colors
        if '██' in line:
            text.append(line, style="bold white on blue")
        else:
            text.append(line, style="white")
        
        if i < len(lines) - 1:
            text.append('\n')
    
    return text
```

## Monitoring & Analytics

### Track Logo Display Events

```python
import logging

logger = logging.getLogger(__name__)

def show_logo_with_analytics(variant='default'):
    """Show logo and log analytics"""
    logger.info(f"Logo displayed: variant={variant}")
    
    # Track metrics
    from analytics import track_event
    track_event('logo_shown', {'variant': variant})
    
    # Display logo
    console = Console()
    if variant == 'gradient':
        logo = get_logo_gradient()
    elif variant == 'neon':
        logo = get_logo_neon()
    else:
        logo = get_logo()
    
    console.print(logo)
```

## Conclusion

The `logo.py` module provides a professional, visually appealing branding element for CORTEX. It's ideal for:

- **CLI applications** requiring startup screens
- **Terminal UIs** with rich formatting
- **Interactive shells** (REPL interfaces)
- **Help/documentation** displays
- **Loading/splash screens**

### When to Use Different Variants

| Variant | Best For | Use Case |
|---------|----------|----------|
| **Standard** | General purpose | Default startup, help screens |
| **Gradient** | Visual appeal | Splash screens, demos |
| **Neon** | High energy | Interactive mode, live demos |
| **Plain** | Logging/files | Error logs, documentation |

The module's simple API and rich styling options make it easy to maintain consistent branding across your application while providing flexibility for different contexts and preferences.