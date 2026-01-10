from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Static, ListView, ListItem, Label
from textual.containers import Vertical
from textual.reactive import reactive
from textual.events import Key
from rich.markdown import Markdown

from cortex.query import load_qa_chain
from cortex.memory import ConversationMemory
from cortex.streaming import StreamHandler
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
    _streaming_buffer = ""  # buffer for accumulating streaming tokens

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

        self.query_one("#answer").update_text("Thinking...")
        self.query_one("#sources").update("")
        
        # reset streaming buffer
        self._streaming_buffer = ""

        # create streaming handler
        stream_handler = StreamHandler(self.stream_token)

        # run through router (single brain entry)
        result = await asyncio.to_thread(lambda: execute(query, callbacks=[stream_handler]))

        # display final result (in case streaming didn't capture everything)
        if result:
            self.query_one("#answer").update_text(result)
            self.memory.add(query, result)

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
        """Callback for streaming tokens - safe to call from threads"""
        self.call_from_thread(self._update_stream_token, token)
    
    def _update_stream_token(self, token):
        """Actually update the UI with the token"""
        try:
            # accumulate tokens in buffer
            self._streaming_buffer += token
            # update ui with accumulated text
            self.query_one("#answer").update_text(self._streaming_buffer)
        except Exception as e:
            # in case the widget doesn't exist, just ignore
            pass

if __name__ == "__main__":
    CortexUI().run()