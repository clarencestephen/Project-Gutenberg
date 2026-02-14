#!/usr/bin/env python3
"""
Project-Gutenberg: LLM-Powered Book Recommendations (Sentiment & Theme Analysis)

Uses an LLM to analyze a book's narrative arc, themes, emotional tone, and topic
dynamics — then finds books with similar literary DNA. Inspired by the NMF topic
modeling and sentiment analysis pipeline applied to 3,524 Project Gutenberg books.

Usage:
    python recommend.py "The Yellow Wallpaper"
    python recommend.py "Moby Dick" --n 5
"""

import argparse
import json
import re
import sys

try:
    from openai import OpenAI
except ImportError:
    print("Install openai: pip install openai")
    sys.exit(1)


def get_llm_client():
    """Connect to available LLM backend (vLLM → Ollama → OpenAI)."""
    # Try vLLM
    try:
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
        client.models.list()
        return client, "vLLM"
    except Exception:
        pass

    # Try Ollama
    try:
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        client.models.list()
        return client, "Ollama"
    except Exception:
        pass

    # Try OpenAI
    import os
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return OpenAI(api_key=api_key), "OpenAI"

    print("No LLM backend found. Start vLLM, Ollama, or set OPENAI_API_KEY.")
    sys.exit(1)


def get_recommendations(book_title, n=10):
    """
    Get book recommendations based on sentiment and theme analysis.

    Analyzes the book's narrative arc, themes, emotional tone, and topic dynamics
    to find books with similar literary DNA — similar themes, emotional arcs,
    writing style, or genre conventions.

    Args:
        book_title: Title of the book to analyze.
        n: Number of recommendations to return.

    Returns:
        list[dict]: Recommendations as [{"title": ..., "author": ...}, ...]
    """
    client, backend = get_llm_client()
    print(f"Using {backend} backend")

    prompt = (
        f'Analyze the narrative themes, emotional tone, genre, sentiment arc, and '
        f'literary style of "{book_title}". Consider its topic modeling profile — '
        f'the distribution of themes across the narrative, how sentiment shifts from '
        f'exposition through rising action, climax, falling action, and resolution.\n\n'
        f'Then recommend {n} books that share similar literary DNA — similar themes, '
        f'emotional arcs, writing style, topic complexity, or genre conventions.\n\n'
        f'Return ONLY a JSON array of objects with "title" and "author" keys. '
        f'No explanations, no markdown, no extra text. Example:\n'
        f'[{{"title": "Example Book", "author": "Example Author"}}]'
    )

    models = client.models.list()
    model = models.data[0].id if models.data else "gpt-3.5-turbo"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a literary analysis engine. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=800,
    )

    text = response.choices[0].message.content.strip()

    # Parse JSON from response
    cleaned = re.sub(r'```(?:json)?\s*', '', text)
    cleaned = re.sub(r'```', '', cleaned)
    match = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if not match:
        return []

    try:
        items = json.loads(match.group())
        return [
            {"title": item["title"].strip(), "author": item.get("author", "Unknown").strip()}
            for item in items
            if isinstance(item, dict) and "title" in item
        ][:n]
    except (json.JSONDecodeError, TypeError):
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Project-Gutenberg: LLM-powered book recommendations (sentiment & theme analysis)"
    )
    parser.add_argument("book_title", help="Title of the book to analyze")
    parser.add_argument("--n", type=int, default=10, help="Number of recommendations (default: 10)")
    args = parser.parse_args()

    print(f'\nAnalyzing themes and sentiment for "{args.book_title}"...\n')
    recs = get_recommendations(args.book_title, args.n)

    if not recs:
        print("No recommendations found.")
        return

    print(f"Books with similar literary DNA to \"{args.book_title}\":\n")
    for i, rec in enumerate(recs, 1):
        print(f"  {i:2d}. {rec['title']} — {rec['author']}")
    print()


if __name__ == "__main__":
    main()
