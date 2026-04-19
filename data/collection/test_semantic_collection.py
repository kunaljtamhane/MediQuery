import unittest

from semantic_collection import (
    JSONL_FIELDS,
    SemanticRanker,
    collect_keyword_hits,
    finalize_records,
    make_record,
    select_source_queries,
    tokenize_text,
)


class SemanticCollectionTests(unittest.TestCase):
    def test_record_schema_contains_all_fields(self):
        record = make_record()
        self.assertEqual(set(record.keys()), set(JSONL_FIELDS))

    def test_tokenize_text_keeps_short_biomedical_terms(self):
        tokens = tokenize_text("Vitamin D supplementation improves asthma control")
        self.assertIn("vitamin", tokens)
        self.assertIn("d", tokens)

    def test_finalize_records_scores_and_orders_documents(self):
        query_bank = ["vitamin d supplementation", "asthma control"]
        records = [
            make_record(
                source="pubmed",
                source_id="1",
                title="Vitamin D supplementation improves asthma control",
                abstract="A randomized trial of vitamin d supplementation in asthma.",
            ),
            make_record(
                source="pubmed",
                source_id="2",
                title="Unrelated cardiology paper",
                abstract="Heart rhythm analysis in postoperative care.",
            ),
        ]
        ranked = finalize_records(records, query_bank=query_bank, ranker=SemanticRanker(), limit=2)
        self.assertEqual(ranked[0]["source_id"], "1")
        self.assertIsNotNone(ranked[0]["semantic_score"])
        self.assertIn("vitamin d supplementation", ranked[0]["matched_keywords"])

    def test_collect_keyword_hits_uses_phrases(self):
        record = make_record(
            source="arxiv",
            source_id="x",
            title="Semantic search for vitamin d supplementation in asthma",
            abstract="The paper studies asthma control and supplementation response.",
        )
        hits = collect_keyword_hits(record, ["vitamin d supplementation", "asthma control", "oncology"])
        self.assertEqual(hits, ["vitamin d supplementation", "asthma control"])

    def test_select_source_queries_filters_generic_arxiv_terms(self):
        queries = ["groups", "clinical decision support", "women", "brain tumor imaging"]
        selected = select_source_queries(queries, source="arxiv", max_queries=5)
        self.assertEqual(selected, ["clinical decision support", "brain tumor imaging"])


if __name__ == "__main__":
    unittest.main()
