"""
mitigation/retriever.py
Retrieves relevant context from Wikipedia to ground LLM reasoning.
Used by the re-grounding module when hallucination is detected.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDoc:
    source: str
    title: str
    content: str
    relevance_score: float = 1.0


class WikipediaRetriever:
    """
    Retrieves relevant Wikipedia passages for a given query.

    Uses langchain_community's WikipediaAPIWrapper under the hood.
    Falls back to a simple keyword approach if langchain is unavailable.

    Usage:
        retriever = WikipediaRetriever(top_k=3, max_chars=1200)
        docs = retriever.retrieve("What is transformer architecture in ML?")
    """

    def __init__(
        self,
        top_k: int = 3,
        max_chars_per_doc: int = 1500,
        lang: str = "en",
    ) -> None:
        self._top_k = top_k
        self._max_chars = max_chars_per_doc
        self._lang = lang
        self._wrapper = None
        self._init_wrapper()

    def _init_wrapper(self) -> None:
        try:
            from langchain_community.utilities import WikipediaAPIWrapper
            self._wrapper = WikipediaAPIWrapper(
                top_k_results=self._top_k,
                doc_content_chars_max=self._max_chars,
                lang=self._lang,
            )
            logger.debug("WikipediaAPIWrapper initialized")
        except ImportError:
            logger.warning(
                "langchain_community not installed. "
                "Run: pip install langchain-community wikipedia. "
                "Retriever will return empty results."
            )

    def retrieve(self, query: str) -> list[RetrievedDoc]:
        """
        Retrieve Wikipedia passages relevant to the query.

        Args:
            query: Search query (should be a concise factual question or claim)

        Returns:
            List of RetrievedDoc sorted by relevance
        """
        if not self._wrapper:
            return []

        # Clean query — remove question words that confuse Wikipedia search
        clean_query = re.sub(r"^(what is|who is|when did|where is|how does)\s+", "", query, flags=re.I).strip()
        clean_query = clean_query.rstrip("?.,!")[:200]

        try:
            raw = self._wrapper.run(clean_query)
            return self._parse_raw(raw)
        except Exception as e:
            logger.warning(f"Wikipedia retrieval error for '{clean_query}': {e}")
            return []

    def retrieve_for_claims(self, suspicious_claims: list[str]) -> list[RetrievedDoc]:
        """
        Retrieve context for a list of specific suspicious claims.
        Queries each claim separately and deduplicates.
        """
        seen_titles: set[str] = set()
        docs = []
        for claim in suspicious_claims[:3]:  # cap at 3 claims to save calls
            retrieved = self.retrieve(claim)
            for doc in retrieved:
                if doc.title not in seen_titles:
                    seen_titles.add(doc.title)
                    docs.append(doc)
        return docs[:self._top_k]

    def _parse_raw(self, raw: str) -> list[RetrievedDoc]:
        """
        Parse WikipediaAPIWrapper's raw string output into RetrievedDoc objects.
        The wrapper returns pages separated by blank lines, each starting with
        "Page: <title>\nSummary: ..."
        """
        if not raw or not raw.strip():
            return []

        docs = []
        # Split on "Page: " boundaries
        chunks = re.split(r"\nPage:\s+", raw)
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            lines = chunk.split("\n", 1)
            title = lines[0].replace("Page:", "").strip() if len(lines) > 0 else "Unknown"
            content = lines[1].strip() if len(lines) > 1 else chunk

            # Remove "Summary: " prefix if present
            content = re.sub(r"^Summary:\s*", "", content)
            # Truncate
            content = content[: self._max_chars]

            docs.append(
                RetrievedDoc(
                    source="wikipedia",
                    title=title,
                    content=content,
                )
            )

        return docs[: self._top_k]

    def format_for_prompt(self, docs: list[RetrievedDoc]) -> str:
        """Format retrieved docs as a context block for inclusion in prompts."""
        if not docs:
            return ""
        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(f"[Source {i}: {doc.title}]\n{doc.content}")
        return "\n\n".join(parts)
