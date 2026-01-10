from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Static, ListView, ListItem, Label
from textual.containers import Vertical
from textual.reactive import reactive
from textual.events import Key
from rich.markdown import Markdown

from cortex.query import load_qa_chain
from cortex.memory import ConversationMemory
from cortex.streaming import SreamHandler
from cortex.router import execute
import asyncio

class AnswerBox(Static):
    def update_text(self, text: str):
        self.update(Markdown(text))

class CortexUI(App):
    CSS = """
    Screen {
        background: black;
    }

    Input {
        border: round #00ffff;
    }

    Static {
        color: #00ffcc;
    }
    """

    memory = ConversationMemory()
    show_history = reactive(False)

    answer_text = reactive("")
    sources_text = reactive("")

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Vertical(
            Static("CORTEX - Local Knowledge Interface", id="title"),
            Input(placeholder="Ask CORTEX something...", id="query"),
            Static("Answer:", id="answer_label"),
            AnswerBox(id="answer"),
            Static("Source:", id="sources_label"),
            Static("", id="sources"),
        )
        yield Footer()

    async def on_input_submitted(self, event: Input.Submitted):
        query = event.value.strip()
        event.input.value = ""

        if not query:
            return

        self.memory.add_query(query)

        self.query_one("#answer").update_text("Thinking...")
        self.query_one("#sources").update("")

        # run through router (single brain entry)
        result = await asyncio.to_thread(lambda: execute(query))

        # display result
        self.query_one("#answer").update_text(result)

    async def on_key(self, event: Key):
        if event.key == "ctrl+r":
            self.show_history = not self.show_history
            if self.show_history:
                self.open_history()
            else:
                self.close_history()

    def open_history(self):
        queries = self.memory.all_queries()
        items = [ListItem(Label(q)) for q in queries[::-1]]

        self.mount(ListView(*items, id="history"))

    def close_history(self):
        if self.query("#history"):
            self.query_one("#history").remove()

    def stream_token(self, token):
        current = self.query_one("#answer").renderable
        self.query_one("#answer").update_text(str(current) + token)

if __name__ == "__main__":
    CortexUI().run()