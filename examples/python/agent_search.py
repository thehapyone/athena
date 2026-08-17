"""Retrieve cited Athena context for a Python AI agent.

Usage:
    ATHENA_URL=http://127.0.0.1:8080 ATHENA_API_TOKEN=... \
        python examples/python/agent_search.py "How do returns work?" support
"""

import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def search_athena(query: str, collection_id: str) -> list[dict]:
    """Return cited passages ready to add to an agent's context."""
    base_url = os.environ.get("ATHENA_URL", "http://127.0.0.1:8080").rstrip("/")
    token = os.environ["ATHENA_API_TOKEN"]
    body = json.dumps({"query": query, "collection_ids": [collection_id]}).encode()
    request = Request(
        f"{base_url}/v1/search",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=15) as response:
        return json.load(response).get("items", [])


def agent_context(query: str, collection_id: str) -> str:
    """Format Athena passages and citations for an agent prompt."""
    passages = search_athena(query, collection_id)
    if not passages:
        return "No relevant Athena sources were found. Do not make up an answer."

    lines = ["Use only the cited context below. If it is insufficient, say so."]
    for passage in passages:
        citations = passage.get("citations", [])
        citation = citations[0] if citations else {}
        location = citation.get("locator") or citation.get("section") or "source"
        source = citation.get("source_uri") or passage.get("title") or "unknown source"
        lines.append(f"\n[{source} — {location}]\n{passage['text']}")
    return "\n".join(lines)


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "What information is available?"
    collection = sys.argv[2] if len(sys.argv) > 2 else "support"
    try:
        print(agent_context(query, collection))
    except KeyError:
        raise SystemExit("Set ATHENA_API_TOKEN before running this example.") from None
    except HTTPError as error:
        raise SystemExit(f"Athena returned HTTP {error.code}.") from None
    except URLError as error:
        raise SystemExit(f"Could not reach Athena: {error.reason}") from None
