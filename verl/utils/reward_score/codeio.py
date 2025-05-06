import json
import re
from typing import Any, Dict, Union, List

def extract_json_from_text(text: str) -> str:
    """
    Extract JSON content from text that might contain code blocks with or without markdown.
    
    Args:
        text: The input text which may contain code blocks or raw JSON
        
    Returns:
        Extracted JSON content without markdown formatting
    """
    text = str(text)
    
    # Try to extract from code blocks with language marker
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    code_blocks = re.findall(code_block_pattern, text)
    
    if code_blocks:
        # Use the first code block if found
        return code_blocks[0].strip()
    
    # If no code blocks found, assume the text itself contains JSON
    return text.strip()

def normalize_json_string(json_str: str) -> str:
    """
    Normalize JSON string to ensure it's a valid JSON object.
    
    Args:
        json_str: JSON string to normalize, could be partial (missing outer braces)
        
    Returns:
        Normalized JSON string
    """
    json_str = str(json_str).strip()
    
    # Check if the string starts with a quote (likely a key fragment)
    if json_str.startswith('"') and ':' in json_str and not json_str.startswith('{"'):
        # It's likely a fragment, add outer braces
        json_str = '{' + json_str + '}'
    
    # Handle strings that look like they should be objects but don't have braces
    if not (json_str.startswith('{') or json_str.startswith('[')) and ('"' in json_str and ':' in json_str):
        json_str = '{' + json_str + '}'
    
    # Fix stringified arrays - convert string "[97]" to actual array representation
    array_pattern = r'"(\[[\d, ]*\])"'
    json_str = re.sub(array_pattern, lambda m: m.group(1), json_str)
    
    return json_str

def parse_json_safely(json_str: Union[str, Any]) -> Any:
    """
    Parse JSON string into a Python object with error handling.
    
    Args:
        json_str: JSON string to parse
        
    Returns:
        Parsed JSON object or the original string if parsing fails
    """
    if not isinstance(json_str, str):
        return json_str  # Already parsed
        
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            # Replace single quotes with double quotes
            json_str = json_str.replace("'", '"')
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                # Try wrapping with braces if needed
                if not (json_str.startswith('{') or json_str.startswith('[')):
                    json_str = '{' + json_str + '}'
                return json.loads(json_str)
            except json.JSONDecodeError:
                # If all attempts fail, return string
                return json_str

def normalize_array_strings(obj: Any) -> Any:
    """
    Convert stringified arrays like "[1,2,3]" to actual arrays [1,2,3]
    
    Args:
        obj: Any Python object that might contain stringified arrays
        
    Returns:
        Object with normalized arrays
    """
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if isinstance(v, str) and v.startswith('[') and v.endswith(']'):
                try:
                    result[k] = json.loads(v)
                except json.JSONDecodeError:
                    result[k] = v
            else:
                result[k] = normalize_array_strings(v)
        return result
    elif isinstance(obj, list):
        return [normalize_array_strings(item) for item in obj]
    else:
        return obj

def normalize_json(json_input: Union[str, Any]) -> Any:
    """
    Convert the input to a standardized JSON object.
    
    Args:
        json_input: Either a string containing JSON or a Python object
        
    Returns:
        Normalized JSON as Python object
    """
    if isinstance(json_input, str):
        # Extract JSON from text (handles code blocks)
        extracted = extract_json_from_text(json_input)
        # Normalize the JSON string
        normalized = normalize_json_string(extracted)
        # Parse to Python object
        obj = parse_json_safely(normalized)
        # Handle stringified arrays in object
        return normalize_array_strings(obj)
    else:
        # If it's already a Python object, handle possible stringified arrays
        return normalize_array_strings(json_input)

def are_values_equivalent(val1: Any, val2: Any) -> bool:
    """
    Check if two values are equivalent, with special handling for various types.
    
    Args:
        val1: First value
        val2: Second value
        
    Returns:
        True if values are equivalent, False otherwise
    """
    # Handle None/null
    if val1 is None and val2 is None:
        return True
    
    # Numeric comparison (treat ints and floats as equivalent if same value)
    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
        return abs(val1 - val2) < 1e-10  # Allow small floating point differences
    
    # String comparison (case-insensitive for string values)
    if isinstance(val1, str) and isinstance(val2, str):
        # If they look like arrays, parse them
        if val1.startswith('[') and val1.endswith(']') and val2.startswith('[') and val2.endswith(']'):
            try:
                arr1 = json.loads(val1)
                arr2 = json.loads(val2)
                return compare_json_equivalence(arr1, arr2)
            except json.JSONDecodeError:
                pass
        return val1.lower() == val2.lower()
    
    # Regular equality for other types
    return val1 == val2

def compare_json_equivalence(obj1: Any, obj2: Any) -> bool:
    """
    Compare two JSON objects for semantic equivalence.
    
    Args:
        obj1: First JSON object
        obj2: Second JSON object
        
    Returns:
        True if objects are equivalent, False otherwise
    """
    # Normalize inputs if they're strings
    if isinstance(obj1, str):
        obj1 = normalize_json(obj1)
    if isinstance(obj2, str):
        obj2 = normalize_json(obj2)
    
    # List of possible wrapper keys
    wrapper_keys = ["input", "output"]
    
    # Check if obj1 has a wrapper
    obj1_wrapper = None
    inner_obj1 = None
    if isinstance(obj1, dict):
        for key in wrapper_keys:
            if key in obj1 and isinstance(obj1[key], dict):
                obj1_wrapper = key
                inner_obj1 = obj1[key]
                break
    
    # Check if obj2 has a wrapper
    obj2_wrapper = None
    inner_obj2 = None
    if isinstance(obj2, dict):
        for key in wrapper_keys:
            if key in obj2 and isinstance(obj2[key], dict):
                obj2_wrapper = key
                inner_obj2 = obj2[key]
                break
    
    # Handle wrapper patterns
    if inner_obj1 is not None:
        if inner_obj2 is not None:
            # Both have wrappers (could be different ones)
            return compare_json_equivalence(inner_obj1, inner_obj2)
        elif isinstance(obj2, dict):
            # Only obj1 has a wrapper, check if obj2 has same content
            if set(inner_obj1.keys()).issubset(set(obj2.keys())):
                return compare_json_equivalence(inner_obj1, obj2)
    
    # Check reverse case when only obj2 has a wrapper
    if inner_obj2 is not None and inner_obj1 is None:
        if isinstance(obj1, dict) and set(inner_obj2.keys()).issubset(set(obj1.keys())):
            return compare_json_equivalence(obj1, inner_obj2)
    
    # Simple equality for non-container types
    if not (isinstance(obj1, (dict, list)) and isinstance(obj2, (dict, list))):
        return are_values_equivalent(obj1, obj2)
    
    # Different types
    if type(obj1) != type(obj2):
        return False
    
    # List comparison
    if isinstance(obj1, list):
        if len(obj1) != len(obj2):
            return False
        
        # For short lists, try both ordered and unordered comparison
        if len(obj1) <= 5:
            # Try unordered first (for sets represented as lists)
            try:
                sorted1 = sorted(obj1, key=str)
                sorted2 = sorted(obj2, key=str)
                if all(compare_json_equivalence(a, b) for a, b in zip(sorted1, sorted2)):
                    return True
            except TypeError:
                # If sorting fails (e.g., mixed types), fall back to ordered comparison
                pass
        
        # Ordered comparison
        return all(compare_json_equivalence(a, b) for a, b in zip(obj1, obj2))
    
    # Dictionary comparison
    if isinstance(obj1, dict):
        if set(obj1.keys()) != set(obj2.keys()):
            return False
        
        return all(compare_json_equivalence(obj1[key], obj2[key]) for key in obj1.keys())
    
    # Default case (shouldn't reach here)
    return obj1 == obj2

def extract_answer_from_model_output(model_output: str) -> str:
    """
    Extract the answer from model output (fallback if prime_math.match_answer is unavailable).
    
    Args:
        model_output: The model's output text
        
    Returns:
        Extracted answer
    """
    # Extract JSON from text
    extracted = extract_json_from_text(model_output)
    return extracted

def compute_score(model_output: str, ground_truth: str) -> Dict[str, bool]:
    """
    Compute score based on equivalence of model output and ground truth.
    
    Args:
        model_output: The model's output
        ground_truth: The expected ground truth
        
    Returns:
        Dict with score results
    """
    model_output = str(model_output)
    ground_truth = str(ground_truth)
    

    response = extract_answer_from_model_output(model_output)
    
    # Compare for equivalence
    score = compare_json_equivalence(response, ground_truth)
    
    # print(f"=========RESPONSE=========")
    # print(f"{response}")
    # print(f"=========GROUND TRUTH=======")
    # print(f"{ground_truth}")
    # print(f"================================")
    # if score:
    #     print(f"✅ Correct")
    # else:
    #     print(f"❌ Incorrect")
    
    return {"score": score, "acc": score}


# Example usage:
if __name__ == "__main__":
    # Example 1
    model_out1 = '''```json
{"input": {"upper_limit": 2924}}
```'''
    ground_truth1 = '"input": {"upper_limit": 2719}'
    result1 = compute_score(model_out1, ground_truth1)
    print(f"Example 1 result: {result1}")
    
    # Example 2
    model_out2 = '''```json
{
  "preorder": "[97]",
  "inorder": "[97]"
}
```'''
    ground_truth2 = '"input": {"preorder": "[97]", "inorder": "[97]"}'
    result2 = compute_score(model_out2, ground_truth2)
    print(f"Example 2 result: {result2}")
    
    # Example 3 - testing "output" wrapper
    model_out3 = '''```json
{"output": {"result": 42}}
```'''
    ground_truth3 = '"output": {"result": 42}'
    result3 = compute_score(model_out3, ground_truth3)
    print(f"Example 3 result: {result3}")
    
    # Example 4 - testing when one uses "input" and one uses "output"
    model_out4 = '''```json
{"input": {"answer": true}}
```'''
    ground_truth4 = '"output": {"answer": true}'
    result4 = compute_score(model_out4, ground_truth4)
    print(f"Example 4 result: {result4}")