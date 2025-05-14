import re
import random
import ast
import operator


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # TWK NOTE: Not all solutions have this so we needlessly return None for a lot of cases
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # elif "<think>" in solution_str:
    #     solution_str = solution_str.split("<think>", 1)[1]
    # else:
    #     return None
    # TWK NOTE: Gets tripped up by string ending with a newline after the <answer> tag
    # solution_str = solution_str.split('\n')[-1] 
    if "<answer>" not in solution_str:
        # If there is no <answer> tag, return None
        return None

    # TWK NOTE: Jump straight to extracting the answer 
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        solution = matches[-1].group(1).strip()
        sol_parts = solution.split('=')
        if len(sol_parts) > 1:
            # If there are multiple parts, take the longest one
            #  this is in place to try and extract the equation which 
            #  will always be longer than the result
            sol_parts = sorted(sol_parts, key=len, reverse=True)
            # Remove any leading or trailing whitespace
            final_answer = sol_parts[0].strip()
        else:
            final_answer = solution
    else:
        final_answer = None
    return final_answer


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]
        
        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)
        
        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as e:
        return None


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    target = ground_truth['target']
    numbers = ground_truth['numbers']
    
    equation = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")
        print(f"Valid equation?: {validate_equation(equation, numbers)}")
        print(f"Evaluated equation: {evaluate_equation(equation)}")
        print(f"--------------------------------")

    if equation is None:
        if do_print:
            print(f"No equation found")
        reward = -1.0 # We want the full negative value here
        format_reward = 0.0
        correct = False
        result = None
    else:
        reward = 0.0
        format_reward = format_score
    
        # Validate equation uses correct numbers
        if not validate_equation(equation, numbers):
            if do_print:
                print(f"Invalid equation")
            correct = False
        
        else:
            correct = False
            try:
                result = evaluate_equation(equation)
                if result is None:
                    if do_print:
                        print(f"Could not evaluate equation")
                
                if abs(result - target) < 1e-5: # Account for floating point precision
                    if do_print:
                        print(f"Correct equation: {equation} = {result}")
                    reward = score
                    correct = True

                else:
                    if do_print:
                        print(f"Wrong result: equation = {result}, target={target}")
                    reward = -1.0  # TWK NOTE: A slight variation here, we want to penalize an incorrect answer ?

            except:
                if do_print:
                    print(f"Error evaluating function")
                    result = None
                    correct = False
                    reward = 0.0

    # TWK NOTE: another slight variation here... Returning a dict... Combining accuarcy and format rewards
    return_dict = {
        "score": reward + format_reward,
        "format_reward": format_reward,
        "acc": correct,
    }
    try:
        return_dict["pred"] = f"{equation} = {result}"
    except:
        return_dict["pred"] = "'result' or 'equation' is not available'"
    
    return return_dict