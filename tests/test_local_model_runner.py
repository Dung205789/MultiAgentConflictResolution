import unittest
import json
import tempfile
from pathlib import Path

from app.main import collect_unique_extraction_texts
from src.local_models.runner import TransformerAgent


class TransformerExtractionParseTests(unittest.TestCase):
    def _make_agent(self) -> TransformerAgent:
        agent = TransformerAgent.__new__(TransformerAgent)
        agent.agent_id = "agent_test"
        agent.model_name = "test-model"
        agent.device = "cpu"
        agent.quantization_mode = "none"
        agent.strict_loading = False
        agent.generator = None
        agent.last_extraction_response = ""
        return agent

    def test_parse_single_json_object(self):
        agent = self._make_agent()
        response = """
        {
          "subject": "Thomas Kyd",
          "predicate": "birth_place",
          "object_val": "London",
          "confidence": 0.9
        }
        """
        items = agent._parse_extraction_response(response, "Thomas Kyd was born in the city of London")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["subject"], "Thomas Kyd")
        self.assertEqual(items[0]["predicate"], "birth_place")
        self.assertEqual(items[0]["object_val"], "London")

    def test_parse_line_fields(self):
        agent = self._make_agent()
        response = """
        subject: Thomas Kyd
        predicate: birth_place
        object_val: London
        """
        items = agent._parse_extraction_response(response, "Thomas Kyd was born in the city of London")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["predicate"], "birth_place")
        self.assertEqual(items[0]["object_val"], "London")

    def test_parse_openai_style_predicate_phrase_is_canonicalized(self):
        agent = self._make_agent()
        response = """
        {
          "items": [
            {
              "subject": "Thomas Kyd",
              "predicate": "was born in",
              "object_val": "the city of London",
              "confidence": 0.85
            }
          ]
        }
        """
        items = agent._parse_extraction_response(response, "Thomas Kyd was born in the city of London")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["predicate"], "birth_place")
        self.assertEqual(items[0]["object_val"], "London")


class TransformerExtractionCacheTests(unittest.TestCase):
    def _make_agent(self, cache_path: str) -> TransformerAgent:
        agent = TransformerAgent.__new__(TransformerAgent)
        agent.agent_id = "agent_cache"
        agent.model_name = "test-model"
        agent.device = "cpu"
        agent.quantization_mode = "none"
        agent.strict_loading = True
        agent.generator = object()
        agent.last_extraction_response = ""
        agent._init_extraction_cache(cache_path)
        return agent

    def test_persistent_cache_reuses_results_across_agent_instances(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = str(Path(tmpdir) / "extract_cache.jsonl")
            text = "Thomas Kyd was born in the city of London"
            response = json.dumps({
                "items": [
                    {
                        "subject": "Thomas Kyd",
                        "predicate": "birth_place",
                        "object_val": "London",
                        "confidence": 0.9,
                    }
                ]
            })

            agent1 = self._make_agent(cache_path)
            calls = {"count": 0}

            def generate_response_first(prompt: str, **kwargs) -> str:
                calls["count"] += 1
                return response

            agent1.generate_response = generate_response_first
            first = agent1.extract_memories(text)
            self.assertEqual(calls["count"], 1)
            self.assertEqual(agent1.get_extraction_cache_stats()["cache_writes"], 1)
            self.assertEqual(first[0]["predicate"], "birth_place")

            agent2 = self._make_agent(cache_path)

            def generate_response_second(prompt: str, **kwargs) -> str:
                raise AssertionError("cache hit should avoid a second model call")

            agent2.generate_response = generate_response_second
            second = agent2.extract_memories(text)
            self.assertEqual(second, first)
            self.assertEqual(agent2.get_extraction_cache_stats()["cache_hits"], 1)
            self.assertTrue(Path(cache_path).exists())

    def test_collect_unique_extraction_texts_dedupes_across_scenarios(self):
        scenarios = [
            {
                "scenario_id": "s0",
                "ordered_events": [
                    {"event_type": "write_proposal", "proposal": {"raw_text": "A"}},
                    {"event_type": "write_proposal", "proposal": {"raw_text": "B"}},
                ],
            },
            {
                "scenario_id": "s1",
                "ordered_events": [
                    {"event_type": "write_proposal", "proposal": {"raw_text": "A"}},
                    {"event_type": "write_proposal", "proposal": {"raw_text": "C"}},
                ],
            },
        ]
        self.assertEqual(collect_unique_extraction_texts(scenarios), ["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()
