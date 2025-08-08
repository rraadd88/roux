import ast

def to_pms(input_string: str) -> dict:
    """
    Converts a multiline string of Python variable assignments into a dictionary.
    
    This version is strict:
    - Raises ValueError if a non-empty line does not contain an assignment operator '='.
    - Raises ValueError if the assigned value is not a valid Python literal
      (e.g., contains function calls, imports, or invalid syntax).
    - Handles spaces around assignments, comments, and empty lines.
    """
    result_dict = {}
    
    # Iterate through each line, keeping track of the original line and number for errors
    for i, original_line in enumerate(input_string.split('\n')):
        line_num = i + 1 # Human-readable line number (starts from 1)

        # 1. Remove comments: Find the first '#' and take the part before it.
        # This correctly handles lines with no '#' as well.
        processing_line = original_line
        if '#' in processing_line:
            processing_line = processing_line.split('#', 1)[0]
        
        # 2. Strip leading/trailing whitespace from the (now comment-free) line.
        # This handles indentation like '    last_var = "end"' correctly.
        processing_line = processing_line.strip()

        # 3. Skip purely empty lines (after stripping comments and whitespace).
        # These are not considered errors, just ignored.
        if not processing_line:
            continue

        # 4. Check for assignment operator '='
        if '=' not in processing_line:
            raise ValueError(
                f"Error on line {line_num}: No assignment operator '=' found. "
                f"Line content: '{original_line}'"
            )

        # 5. Split and evaluate the assignment
        try:
            var_name, value_str = processing_line.split('=', 1)
            var_name = var_name.strip()
            value_str = value_str.strip()

            # Safely evaluate the value string using ast.literal_eval
            evaluated_value = ast.literal_eval(value_str)
            result_dict[var_name] = evaluated_value
        except (ValueError, SyntaxError) as e:
            # Catch errors from ast.literal_eval (invalid literals or syntax)
            raise ValueError(
                f"Error on line {line_num}: Value for '{var_name}' is not a valid Python literal "
                f"or contains a syntax error. Line content: '{original_line}' -> {e}"
            ) from e # Chain the original exception for debugging
        except Exception as e:
            # Catch any other unexpected errors during the parsing of this line
            raise ValueError(
                f"Error on line {line_num}: Unexpected error during parsing. "
                f"Line content: '{original_line}' -> {e}"
            ) from e
            
    return result_dict