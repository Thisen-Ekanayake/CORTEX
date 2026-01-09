import argparse
from cortex.ingest import ingest
from cortex.query import ask


def main():
    parser = argparse.ArgumentParser(description="CORTEX - Document Q&A Assistant")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest documents into the vector database"
    )
    parser.add_argument(
        "--ask",
        type=str,
        help="Ask a question to CORTEX"
    )

    args = parser.parse_args()

    if args.ingest:
        ingest()

    if args.ask:
        ask(args.ask)


if __name__ == "__main__":
    main()
