"""
Knowledge Graph Module — Jaya (Week 6)
Extracts medical entities from text using SciSpaCy and builds a NetworkX
co-occurrence graph. At query time, returns first- and second-degree neighbor
relationships for query entities as structured context for the LLM.

Entity types recognized: DISEASE, CHEMICAL, GENE_OR_GENE_PRODUCT,
CELL_TYPE, ORGANISM, CANCER (via en_core_sci_lg NER model).

Low-confidence guard: nodes with 0 or 1 co-occurrence edges are flagged
and excluded from query-time context to prevent speculative reasoning.
"""
import logging
from collections import defaultdict
from typing import Optional

import networkx as nx

log = logging.getLogger(__name__)

# SciSpaCy model — loaded lazily to avoid import-time cost
_nlp = None
SCISPACY_MODEL = "en_core_sci_lg"

MEDICAL_ENTITY_LABELS = {
    "DISEASE", "CHEMICAL", "GENE_OR_GENE_PRODUCT",
    "CELL_TYPE", "ORGANISM", "CANCER",
}

# Global graph singleton — populated by index_document calls
_graph: nx.Graph = nx.Graph()


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load(SCISPACY_MODEL)
            log.info(f"Loaded SciSpaCy model: {SCISPACY_MODEL}")
        except Exception as e:
            log.warning(f"SciSpaCy model '{SCISPACY_MODEL}' unavailable: {e}. NER disabled.")
            _nlp = False
    return _nlp if _nlp is not False else None


def extract_entities(text: str) -> list[str]:
    """Extract medical entity surface forms from text using SciSpaCy."""
    nlp = _get_nlp()
    if nlp is None:
        return []
    doc = nlp(text[:50000])  # cap to avoid OOM on very long docs
    return list({
        ent.text.lower().strip()
        for ent in doc.ents
        if ent.label_ in MEDICAL_ENTITY_LABELS and len(ent.text.strip()) > 2
    })


def index_document(doc_id: str, text: str) -> int:
    """
    Extract entities from a document and add co-occurrence edges to the graph.
    Returns number of entities found.
    """
    entities = extract_entities(text)
    for i, e1 in enumerate(entities):
        _graph.add_node(e1)
        for e2 in entities[i + 1:]:
            if _graph.has_edge(e1, e2):
                _graph[e1][e2]["weight"] += 1
                _graph[e1][e2]["docs"].add(doc_id)
            else:
                _graph.add_edge(e1, e2, weight=1, docs={doc_id})
    return len(entities)


def get_query_context(query: str, max_hops: int = 2, min_edge_weight: int = 2) -> str:
    """
    Return structured graph context for query entities.
    Only includes nodes with at least min_edge_weight co-occurrences
    (low-confidence guard per proposal spec).

    Returns a formatted string to prepend to the LLM prompt.
    """
    if _graph.number_of_nodes() == 0:
        return ""

    query_entities = extract_entities(query)
    if not query_entities:
        return ""

    context_lines = []
    seen_pairs: set = set()

    for entity in query_entities:
        if entity not in _graph:
            continue

        # First-degree neighbors above confidence threshold
        neighbors_1 = [
            (n, _graph[entity][n]["weight"])
            for n in _graph.neighbors(entity)
            if _graph[entity][n]["weight"] >= min_edge_weight
        ]
        if not neighbors_1:
            continue

        context_lines.append(f"[KG] '{entity}' co-occurs with:")
        for neighbor, weight in sorted(neighbors_1, key=lambda x: -x[1])[:5]:
            pair = tuple(sorted([entity, neighbor]))
            if pair not in seen_pairs:
                context_lines.append(f"  → {neighbor} (co-occurrences: {weight})")
                seen_pairs.add(pair)

            if max_hops >= 2:
                neighbors_2 = [
                    (n2, _graph[neighbor][n2]["weight"])
                    for n2 in _graph.neighbors(neighbor)
                    if n2 != entity and _graph[neighbor][n2]["weight"] >= min_edge_weight
                ]
                for n2, w2 in sorted(neighbors_2, key=lambda x: -x[1])[:3]:
                    pair2 = tuple(sorted([neighbor, n2]))
                    if pair2 not in seen_pairs:
                        context_lines.append(f"      → {n2} (via {neighbor}, co-occurrences: {w2})")
                        seen_pairs.add(pair2)

    if not context_lines:
        return ""

    header = f"[Knowledge Graph Context — {_graph.number_of_nodes()} entities, {_graph.number_of_edges()} relationships]"
    return header + "\n" + "\n".join(context_lines) + "\n"


def graph_stats() -> dict:
    return {
        "nodes": _graph.number_of_nodes(),
        "edges": _graph.number_of_edges(),
        "low_confidence_nodes": sum(
            1 for n in _graph.nodes
            if sum(d["weight"] for _, _, d in _graph.edges(n, data=True)) <= 1
        ),
    }
