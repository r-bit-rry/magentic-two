import json
from typing import Literal, Union
import unittest

from pydantic import BaseModel

from structured_output import Ledger
from utils import extract_inline_json_schema
class TestExtractInlineJsonSchema(unittest.TestCase):
    def test_extract_inline_json_schema(self):
        prompt = """
Please output an answer in pure JSON format according to the following schema. The JSON object must be parsable as-is. DO NOT OUTPUT ANYTHING OTHER THAN JSON, AND DO NOT DEVIATE FROM THIS SCHEMA:

    {{
       "is_request_satisfied": {{
            "reason": string,
            "answer": boolean
        }},
        "is_in_loop": {{
            "reason": string,
            "answer": boolean
        }},
        "is_progress_being_made": {{
            "reason": string,
            "answer": boolean
        }},
        "next_speaker": {{
            "reason": string,
            "answer": string (select from: "{names}")
        }},
        "instruction_or_question": {{
            "reason": string,
            "answer": string
        }}
    }}
        """

        pydantic_schema = Ledger.model_json_schema()
        expected_schema = {'type': 'object', 'properties': {'is_request_satisfied': {'required': ['reason', 'answer'], 'reason': {'type': 'string'}, 'answer': {'type': 'boolean'}}, 'is_in_loop': {'required': ['reason', 'answer'], 'reason': {'type': 'string'}, 'answer': {'type': 'boolean'}}, 'is_progress_being_made': {'required': ['reason', 'answer'], 'reason': {'type': 'string'}, 'answer': {'type': 'boolean'}}, 'next_speaker': {'required': ['reason', 'answer'], 'reason': {'type': 'string'}, 'answer': {'type': 'string', 'enum': ['Alice', 'Bob', 'Charlie']}}, 'instruction_or_question': {'required': ['reason', 'answer'], 'reason': {'type': 'string'}, 'answer': {'type': 'string'}}}, 'required': ['is_request_satisfied', 'is_in_loop', 'is_progress_being_made', 'next_speaker', 'instruction_or_question']}
        actual_schema = extract_inline_json_schema(prompt.format(names="Alice, Bob, Charlie"))
        print("=== Expected Schema ===")
        print(expected_schema)
        print("=== Actual Schema ===")
        print(actual_schema)
        print("=== Pydantic Schema ===")
        print(pydantic_schema)
        # self.assertTrue(Ledger.model_validate_json(actual_schema))
        self.assertEqual(expected_schema, actual_schema)


if __name__ == "__main__":
    unittest.main()
