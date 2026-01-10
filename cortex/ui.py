from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Static
from textual.containers import Vertical
from textual.reactive import reactive
from rich.markdown import Markdown

from cortex.query import load_qa_chain
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

if __name__ == "__main__":
    CortexUI().run()