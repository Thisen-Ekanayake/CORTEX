from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Static, ListView, ListItem, Label
from textual.containers import Vertical
from textual.reactive import reactive
from textual.events import Key
from rich.markdown import Markdown

from cortex.query import load_qa_chain
from cortex.memory import ConversationMemory
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
        query = event.value
        event.input.value = ""

        self.query_one("#answer").update_text("Retrieving Knowledge...")
        self.query_one("#sources").update("")

        chain, retriever = load_qa_chain()

        # run retrieval
        answer = await asyncio.to_thread(chain.invoke, query)
        docs = retriever._get_relevant_documents(query, run_manager=None)

        sources = []
        for doc in docs:
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "N/A")
            sources.append(f"- {src} (page {page})")

        self.query_one("#answer").update_text(answer)
        self.query_one("#sources").update("\n".join(sources))

    async def on_key(self, event: Key):
        if event.ctrl and event.key == "r":
            self.show_history = not self.show_history
            if self.show_history:
                self.open_history()

    def open_history(self):
        queries = self.memory.all_queries()
        items = [ListItem(Label(q)) for q in queries[::-1]]

        self.mount(
            ListView(*items, id="history")
        )

if __name__ == "__main__":
    CortexUI().run()