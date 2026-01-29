import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def web_search(query: str) -> str:
    try:
        response = client.search(query=query, search_depth="basic", max_results=3)

        if not response or "results" not in response:
            return None

        answers = [r["content"] for r in response["results"]]
        return "\n\n".join(answers[:2])

    except Exception:
        return None
