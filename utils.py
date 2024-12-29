import json
import re
from typing import Dict

def extract_inline_json_schema(prompt: str) -> Dict:
    """
    Extracts the inlined JSON object schema from the given prompt and converts it into a standard JSON schema.

    Args:
        prompt (str): The prompt string containing the embedded JSON schema.

    Returns:
        Dict: A dictionary representing the JSON schema.
    """

    # Updated regex to capture the JSON schema block
    match = re.search(r"(\{[\s\S]*\})", prompt, re.DOTALL)

    if not match:
        raise LookupError("No JSON schema found in the prompt.")

    json_schema_str = match.group(1)

    # Remove any newline characters and excess whitespace for consistent processing
    json_schema_str = re.sub(r'\n', ' ', json_schema_str)
    json_schema_str = re.sub(r'\s+', ' ', json_schema_str)

    # Handle enumerations first
    # Example: "answer": string (select from: "Alice, Bob, Charlie")
    enum_pattern = r'"(\w+)":[\w\d"\s]+?\(select from:\s*"?([^"]+)"?\)'
    json_schema_str = re.sub(
        enum_pattern,
        lambda m: f'"{m.group(1)}": "enum: {m.group(2)}"',
        json_schema_str
    )

    # Replace type placeholders with quoted strings
    # This avoids replacing types within enum annotations
    type_pattern = r'":\s*(string|boolean|integer|array|float|object)'
    json_schema_str = re.sub(
        type_pattern,
        r'": "\1"',
        json_schema_str
    )

    try:
        inline_schema = json.loads(json_schema_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON schema\n {prompt}\n: {e}")

    # Convert to standard JSON schema format
    standard_schema = {"type": "object", "properties": {}, "required": []}

    for key, value in inline_schema.items():
        standard_schema["properties"][key] = {"required": []}
        for sub_key, sub_value in value.items():
            # Handle enumerations
            if isinstance(sub_value, str) and sub_value.startswith("enum:"):
                enum_values = [item.strip().strip('"') for item in sub_value.replace("enum:", "").split(",")]
                standard_schema["properties"][key][sub_key] = {
                    "type": "string",
                    "enum": enum_values,
                }
            else:
                # Map type strings to JSON schema types
                type_mapping = {
                    "string": "string",
                    "boolean": "boolean",
                    "integer": "integer",
                    "array": "array",
                    "float": "float",
                    "object": "object",
                }
                prop_type = type_mapping.get(sub_value.lower(), "string")
                standard_schema["properties"][key][sub_key] = {"type": prop_type}
            standard_schema["properties"][key]["required"].append(sub_key)
        standard_schema["required"].append(key)

    return standard_schema
