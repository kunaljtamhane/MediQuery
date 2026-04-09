"""
Person C — Research Agent (Weeks 7-8)
Queries arXiv and PubMed live for papers not in the local corpus.
Owned jointly by Person C (integration) and Person D (data patterns).
"""
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

log = logging.getLogger(__name__)
ARXIV_API = "http://export.arxiv.org/api/query"
PUBMED_SEARCH_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
NS = "{http://www.w3.org/2005/Atom}"


def _open_url(url: str, timeout: int = 10) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def _clean_text(value: str | None) -> str:
    return " ".join((value or "").split())


def _normalize_pubmed_month(value: str) -> str:
    month = _clean_text(value).lower()
    if not month:
        return "01"
    if month.isdigit():
        return month.zfill(2)

    month_lookup = {
        "jan": "01",
        "january": "01",
        "feb": "02",
        "february": "02",
        "mar": "03",
        "march": "03",
        "apr": "04",
        "april": "04",
        "may": "05",
        "jun": "06",
        "june": "06",
        "jul": "07",
        "july": "07",
        "aug": "08",
        "august": "08",
        "sep": "09",
        "sept": "09",
        "september": "09",
        "oct": "10",
        "october": "10",
        "nov": "11",
        "november": "11",
        "dec": "12",
        "december": "12",
    }
    return month_lookup.get(month, "01")


def _extract_pubmed_date(article: ET.Element) -> str:
    article_date = article.find(".//Article/ArticleDate")
    if article_date is not None:
        year = _clean_text(article_date.findtext("Year"))
        month = _normalize_pubmed_month(article_date.findtext("Month") or "")
        day = _clean_text(article_date.findtext("Day")) or "01"
        return f"{year}-{month}-{day.zfill(2)}" if year else "Unknown date"

    pub_date = article.find(".//JournalIssue/PubDate")
    if pub_date is not None:
        year = _clean_text(pub_date.findtext("Year"))
        month = _normalize_pubmed_month(pub_date.findtext("Month") or "")
        day = _clean_text(pub_date.findtext("Day")) or "01"
        return f"{year}-{month}-{day.zfill(2)}" if year else "Unknown date"

    return "Unknown date"


def _extract_pubmed_abstract(article: ET.Element) -> str:
    sections = []
    for abstract_text in article.findall(".//Article/Abstract/AbstractText"):
        label = _clean_text(abstract_text.attrib.get("Label"))
        text = _clean_text("".join(abstract_text.itertext()))
        if not text:
            continue
        sections.append(f"{label}: {text}" if label else text)
    return "\n".join(sections) or "No abstract available from PubMed."


def _extract_pubmed_authors(article: ET.Element) -> list[str]:
    authors = []
    for author in article.findall(".//Article/AuthorList/Author"):
        collective_name = _clean_text(author.findtext("CollectiveName"))
        if collective_name:
            authors.append(collective_name)
            continue

        fore_name = _clean_text(author.findtext("ForeName"))
        last_name = _clean_text(author.findtext("LastName"))
        if fore_name and last_name:
            authors.append(f"{fore_name} {last_name}")
        elif last_name:
            authors.append(last_name)
    return authors


def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """Query the arXiv API and return structured paper metadata."""
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = f"{ARXIV_API}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            xml_data = resp.read()
    except Exception as e:
        log.error(f"arXiv API error: {e}")
        return []

    root = ET.fromstring(xml_data)
    papers = []
    for entry in root.findall(f"{NS}entry"):
        arxiv_id = entry.find(f"{NS}id").text.split("/abs/")[-1]
        papers.append({
            "source": "arxiv",
            "source_id": arxiv_id,
            "doc_id": f"arxiv:{arxiv_id}",
            "title": entry.find(f"{NS}title").text.strip(),
            "summary": entry.find(f"{NS}summary").text.strip(),
            "authors": [a.find(f"{NS}name").text for a in entry.findall(f"{NS}author")],
            "published": entry.find(f"{NS}published").text[:10],
            "url": entry.find(f"{NS}id").text.strip(),
        })
    return papers


def search_pubmed(query: str, max_results: int = 5) -> list[dict]:
    """Query PubMed E-utilities and return structured paper metadata."""
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results,
        "sort": "pub date",
        "tool": "capstone-research-agent",
    }
    email = os.getenv("NCBI_EMAIL")
    api_key = os.getenv("NCBI_API_KEY")
    if email:
        search_params["email"] = email
    if api_key:
        search_params["api_key"] = api_key

    search_url = f"{PUBMED_SEARCH_API}?{urllib.parse.urlencode(search_params)}"
    try:
        search_payload = json.loads(_open_url(search_url, timeout=10).decode("utf-8"))
        ids = search_payload.get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        log.error(f"PubMed search error: {e}")
        return []

    if not ids:
        return []

    fetch_params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "xml",
        "rettype": "abstract",
        "tool": "capstone-research-agent",
    }
    if email:
        fetch_params["email"] = email
    if api_key:
        fetch_params["api_key"] = api_key

    fetch_url = f"{PUBMED_FETCH_API}?{urllib.parse.urlencode(fetch_params)}"
    try:
        root = ET.fromstring(_open_url(fetch_url, timeout=15))
    except Exception as e:
        log.error(f"PubMed fetch error: {e}")
        return []

    papers = []
    for article in root.findall(".//PubmedArticle"):
        pmid = _clean_text(article.findtext(".//MedlineCitation/PMID"))
        title_node = article.find(".//Article/ArticleTitle")
        title = _clean_text("".join(title_node.itertext())) if title_node is not None else ""
        if not pmid or not title:
            continue

        journal = _clean_text(article.findtext(".//Article/Journal/Title"))
        papers.append({
            "source": "pubmed",
            "source_id": pmid,
            "doc_id": f"pubmed:{pmid}",
            "title": title,
            "summary": _extract_pubmed_abstract(article),
            "authors": _extract_pubmed_authors(article),
            "published": _extract_pubmed_date(article),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "journal": journal,
        })
    return papers


def _format_paper_line(paper: dict) -> str:
    authors = ", ".join(paper.get("authors", [])[:3])
    if len(paper.get("authors", [])) > 3:
        authors += ", et al."
    author_line = f"\nAuthors: {authors}" if authors else ""
    journal = paper.get("journal")
    venue_line = f" | {journal}" if journal else ""
    return (
        f"**{paper['title']}** ({paper['published']}, {paper['source'].title()}{venue_line})"
        f"{author_line}\n{paper['summary'][:200]}...\n[{paper['url']}]\n"
    )


def research_agent_node(state: dict) -> dict:
    """LangGraph node: fetches live arXiv and PubMed results and summarises them."""
    query = state["query"]
    with ThreadPoolExecutor(max_workers=2) as executor:
        arxiv_future = executor.submit(search_arxiv, query, 5)
        pubmed_future = executor.submit(search_pubmed, query, 5)
        arxiv_papers = arxiv_future.result()
        pubmed_papers = pubmed_future.result()

    papers = arxiv_papers + pubmed_papers

    if not papers:
        return {
            **state,
            "final_answer": "No papers found on arXiv or PubMed for this query.",
            "sources": [],
        }

    summary_lines = [f"Found {len(papers)} recent papers across arXiv and PubMed.\n"]
    if arxiv_papers:
        summary_lines.append("### arXiv\n")
        for paper in arxiv_papers:
            summary_lines.append(_format_paper_line(paper))
    if pubmed_papers:
        summary_lines.append("### PubMed\n")
        for paper in pubmed_papers:
            summary_lines.append(_format_paper_line(paper))

    sources = [
        {
            "doc_id": paper["doc_id"],
            "title": paper["title"],
            "score": 1.0,
            "source": paper["source"],
            "url": paper["url"],
        }
        for paper in papers
    ]

    return {
        **state,
        "final_answer": "\n".join(summary_lines),
        "sources": sources,
        "research_results": papers,
    }
