import requests
from langchain.agents import tool

def fetch_paper_metadata(title: str) -> dict:
    """
    Searches for paper metadata (title, URL) using the Semantic Scholar API.
    """
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={title}&limit=1&fields=title,url"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data["data"]:
            paper = data["data"][0]
            return {"title": paper["title"], "url": paper.get("url")}
    return None

def fetch_arxiv_pdf(title: str) -> str:
    """
    Searches for the PDF link of a paper using the ArXiv API.
    """
    base_url = "http://export.arxiv.org/api/query"
    params = {"search_query": f"all:{title}", "start": 0, "max_results": 1}
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        if "arxiv.org" in response.text:
            pdf_link = response.text.split("<link title=\"pdf\" href=\"")[1].split("\"")[0]
            return pdf_link
    return None

@tool
def get_paper_url(title: str) -> str:
    """
    Searches for the URL or PDF link of a paper with the following priorities:
    1. Search for the URL using the Semantic Scholar API.
    2. Search for the PDF link using the ArXiv API.
    """
    # Search in Semantic Scholar
    metadata = fetch_paper_metadata(title)
    if metadata and metadata.get("url"):
        return metadata.get("url")

    # Search in ArXiv
    pdf_link = fetch_arxiv_pdf(title)
    if pdf_link:
        return pdf_link

    return "No paper found for the given title."
