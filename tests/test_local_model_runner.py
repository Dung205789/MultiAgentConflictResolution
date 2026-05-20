import unittest

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


if __name__ == "__main__":
    unittest.main()
