import os
import argparse
import logging
from cortex.ingest import ingest
from cortex.query import ask


def main():
    """
    Main entry point for CORTEX command-line interface.
    
    Supports two modes:
    - --ingest: Ingest documents into the vector database
    - --ask: Ask a question and get an answer from CORTEX
    """
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
        if not os.path.exists("chroma_db/chroma.sqlite3"):
            print("Vector DB not found. Run with --ingest first")
        
        else:
            logging.basicConfig(filename="query.log", level=logging.INFO)
            logging.info(f"Query asked: {args.ask}")
            ask(args.ask)
            

if __name__ == "__main__":
    main()
