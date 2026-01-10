"""
ASCII art logo for CORTEX - Futuristic Design with Colors
"""

from rich.text import Text

# Logo without the border box (Panel will provide the border)

CORTEX_LOGO_TEXT = """
        ██████╗ ██████╗ ██████╗ ████████╗███████╗██╗  ██╗
       ██╔════╝██╔═══██╗██╔══██╗╚══██╔══╝██╔════╝╚██╗██╔╝
       ██║     ██║   ██║██████╔╝   ██║   █████╗   ╚███╔╝ 
       ██║     ██║   ██║██╔══██╗   ██║   ██╔══╝   ██╔██╗ 
       ╚██████╗╚██████╔╝██║  ██║   ██║   ███████╗██╔╝ ██╗
        ╚═════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝

              ════  Think. Retrieve. Answer.  ════
"""


def get_logo(compact=False, futuristic=True, colored=True):
    """
    Return the CORTEX logo
    
    Args:
        compact: Use compact version with tagline
        futuristic: Use futuristic style (currently same as regular)
        colored: Return as Rich Text object with colors (default: True)
    
    Returns:
        Rich Text object if colored=True, otherwise plain string
    """
    # Select the logo version
    logo_text =  CORTEX_LOGO_TEXT
    
    # Return plain string if colors not requested
    if not colored:
        return logo_text
    
    # Create colored version using Rich Text
    text = Text()
    lines = logo_text.strip().split('\n')
    
    for i, line in enumerate(lines):
        if '██' in line:
            # Main CORTEX text - bright cyan bold
            text.append(line, style="bold bright_cyan")
        elif 'Think. Retrieve. Answer.' in line:
            # Tagline - special gradient
            parts = line.split('Think')
            text.append(parts[0], style="cyan")
            text.append('Think', style="bold bright_green")
            
            remaining = 'Think'.join(parts[1:])
            parts2 = remaining.split('Retrieve')
            text.append(parts2[0], style="cyan")
            text.append('Retrieve', style="bold bright_blue")
            
            remaining2 = 'Retrieve'.join(parts2[1:])
            parts3 = remaining2.split('Answer')
            text.append(parts3[0], style="cyan")
            text.append('Answer', style="bold bright_magenta")
            text.append('Answer'.join(parts3[1:]), style="cyan")
        else:
            # Empty or separator lines
            text.append(line, style="cyan")
        
        if i < len(lines) - 1:
            text.append('\n')
    
    return text


def get_logo_gradient(compact=False):
    """
    Return logo with a vertical gradient effect (cyan -> blue)
    """
    logo_text = CORTEX_LOGO_TEXT
    lines = logo_text.strip().split('\n')
    
    # Define gradient colors from cyan to blue
    colors = ["bright_cyan", "cyan", "bright_blue", "blue"]
    
    text = Text()
    for i, line in enumerate(lines):
        # Calculate color based on position
        color_index = int((i / len(lines)) * len(colors))
        color_index = min(color_index, len(colors) - 1)
        color = colors[color_index]
        
        if i < len(lines) - 1:
            text.append(line + '\n', style=f"bold {color}")
        else:
            text.append(line, style=f"bold {color}")
    
    return text


def get_logo_neon(compact=False):
    """
    Return logo with neon-style coloring (bright cyan with bright effects)
    """
    logo_text = CORTEX_LOGO_TEXT
    lines = logo_text.strip().split('\n')
    
    text = Text()
    for i, line in enumerate(lines):
        if '██' in line:
            # Main text - bright neon cyan
            style = "bold bright_cyan"
        elif 'Think. Retrieve. Answer.' in line:
            # Tagline - neon green
            style = "bold bright_green"
        else:
            # Other lines
            style = "dim cyan"
        
        if i < len(lines) - 1:
            text.append(line + '\n', style=style)
        else:
            text.append(line, style=style)
    
    return text


# Example usage
if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.align import Align
    
    console = Console()
    
    console.print("\n[bold white]Standard Colored Logo (Centered in Panel):[/bold white]")
    logo1 = get_logo(compact=True, colored=True)
    console.print(Panel(
        Align.center(logo1),
        border_style="bright_cyan",
        padding=(1, 2),
        title="[bold bright_cyan]CORTEX v1.0[/bold bright_cyan]",
        subtitle="[dim bright_cyan]Neural Interface | Local AI Knowledge Assistant[/dim bright_cyan]"
    ))
    
    console.print("\n[bold white]Gradient Logo:[/bold white]")
    logo2 = get_logo_gradient(compact=True)
    console.print(Panel(
        Align.center(logo2),
        border_style="bright_cyan",
        padding=(1, 2),
        title="[bold bright_cyan]CORTEX v1.0[/bold bright_cyan]"
    ))
    
    console.print("\n[bold white]Neon Logo:[/bold white]")
    logo3 = get_logo_neon(compact=True)
    console.print(Panel(
        Align.center(logo3),
        border_style="bright_cyan",
        padding=(1, 2),
        title="[bold bright_cyan]CORTEX v1.0[/bold bright_cyan]"
    ))