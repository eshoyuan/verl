import functools
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List

import requests
import tenacity
import json
# import multiprocessing as mp
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple
import ast
import re
import astor


class SafeLiteralValidator(ast.NodeVisitor):
    # Blacklisted nodes that should be rejected
    UNSAFE_NODES = {
        ast.Call,  # Function calls (e.g., eval(), exec(), sum([1, 2, 3]))
        ast.Attribute,  # Attribute access (e.g., obj.method, os.system)
        # ast.Name,  # Variable names (e.g., built-ins like open, eval)
        ast.BinOp,  # Binary operations (+, -, *, /, etc.)
        # ast.UnaryOp,  # Unary operations (-1, ~x, etc.)
        ast.Compare,  # Comparisons (==, !=, >, <, etc.)
        ast.Subscript,  # Indexing (e.g., lst[0], dict['key'])
        ast.Lambda,  # Lambda functions
        ast.IfExp,  # Ternary expressions (x if cond else y)
        ast.ListComp,  # List comprehensions
        ast.SetComp,  # Set comprehensions
        ast.DictComp,  # Dictionary comprehensions
        ast.GeneratorExp,  # Generator expressions
    }

    def visit(self, node):
        '''Reject blacklisted nodes and allow everything else.'''
        if isinstance(node, tuple(self.UNSAFE_NODES)):
            raise ValueError(f'Unsafe expression detected: {type(node).__name__}')
        self.generic_visit(node)  # Recursively check child nodes


def is_safe_literal(code) -> bool:
    '''Check if the given string represents a safe literal.'''
    try:
        tree = ast.parse(code, mode='eval')  # Parse as an expression
        SafeLiteralValidator().visit(
            tree.body
        )  # Validate only the body of the expression
        return True
    except Exception as e:
        return False


def test_is_safe_literal():
    # Example usage
    assert is_safe_literal('42')  # ✅ Allowed
    assert is_safe_literal('hello')  # ✅ Allowed
    assert is_safe_literal('[1, 2, 3]')  # ✅ Allowed
    assert is_safe_literal("{'key': 'value'}")  # ✅ Allowed
    assert not is_safe_literal(
        "[4, 5==4, {6: 'a'}]"
    )  # ❌ Not allowed (list with comparison)
    assert not is_safe_literal('2 == 1')  # ❌ Not allowed (comparison)
    assert not is_safe_literal('sum([1, 2, 3])')  # ❌ Not allowed (function call)
    assert not is_safe_literal('x + 1')  # ❌ Not allowed (binary operation)
    assert not is_safe_literal('lambda x: x+1')  # ❌ Not allowed (lambda)


def is_safe_args(code) -> bool:
    """Check if the given string represents a safe argument passing."""
    dummy_call = f"f( {code} )"
    try:
        node_expr = ast.parse(dummy_call, mode='eval')  # Parse as an expression
        node_call: ast.Call = node_expr.body
        for arg in node_call.args:
            SafeLiteralValidator().visit(arg)
        for kwarg in node_call.keywords:
            SafeLiteralValidator().visit(kwarg)
        return True
    except Exception as e:
        return False


def test_is_safe_args():
    # Example usage
    assert is_safe_args('42')  # ✅ Allowed
    assert is_safe_args('hello')  # ✅ Allowed
    assert is_safe_args("{'x': 'y',}, x=1, y=['a', 'b',]")  # ✅ Allowed
    assert not is_safe_args('x + y')  # ❌ Not allowed (binary operation)
    assert not is_safe_args('sum([1, 2, 3])')  # ❌ Not allowed (function call)
    assert not is_safe_args('lambda x: x+1')  # ❌ Not allowed (lambda)
    assert not is_safe_args('1 + 1, 2 + 2')  # ❌ Not allowed (binary operation)

import tiktoken
def extract_last_python_block(text: str) -> str:
    """
    Extracts the content of the last Python code block from the given text.
    Supports blocks marked with ```python or ``` alone.
    """
    # Regex to match ``` or ```python code fences
    pattern = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
    blocks = pattern.findall(text)
    if blocks:
        return blocks[-1].strip()
    return ""




@dataclass
class ExecutorResponse:
    exit_code: int
    stdout: str
    stderr: str
    stdout_match: bool

    def to_dict(self):
        return {
            'exit_code': self.exit_code,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'stdout_match': self.stdout_match,
        }


@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=64),
    retry=tenacity.retry_if_exception_type(requests.exceptions.RequestException),
)
def execute_python_code(
    code: str,
    stdin_str: str = '',
    stdout_expected: str = '',
    url: str = 'http://127.0.0.1:33000/execute_python',
    timeout: int = 5,
) -> ExecutorResponse:
    payload = {
        'code': code,
        'stdin_str': stdin_str,
        'stdout_expected': stdout_expected,
        'timeout': timeout,
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    assert response.status_code == 200, f'{response.status_code = }'
    rdict = response.json()
    resp = ExecutorResponse(
        exit_code=rdict['exit_code'],
        stdout=rdict['stdout'],
        stderr=rdict['stderr'],
        stdout_match=rdict['stdout_match'],
    )

    return resp


def execute_test_cases_mt(
    code: str,
    test_cases: List[List[str]],
    url: str = 'http://127.0.0.1:33000/execute_python',
    timeout: int = 5,
    num_workers: int = 0,
) -> List[ExecutorResponse]:
    exec_partial = functools.partial(
        execute_python_code,
        code=code,
        url=url,
        timeout=timeout,
    )
    if num_workers <= 0:
        num_workers = len(test_cases)
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [
            pool.submit(exec_partial, stdin_str=stdin, stdout_expected=stdout)
            for stdin, stdout in test_cases
        ]
    results = [future.result() for future in futures]

    return results

def score_format(answer) -> Tuple[float, str]:
    if answer is None:
        return 0.0, ''
    code = extract_last_python_block(
        answer
    )
    return 1.0, code

def score_accuracy(prediction: str, raw_sample: dict) -> float:
    if raw_sample['category'] == 'input_prediction':
        if not is_safe_args(prediction):
            return 0.0
    elif raw_sample['category'] == 'output_prediction':
        if not is_safe_literal(prediction):
            return 0.0
    assertion_complete = raw_sample['assertion'].replace(
        'YOUR_PREDICTION_HERE', prediction
    )
    # prevent reward hacking by code injection (not needed for now)
    program_verify = f'{raw_sample["program"]}\n\n{assertion_complete}'
    result = execute_python_code(code=program_verify, timeout=8)
    return 1.0 if result.exit_code == 0 else 0.0

def compute_score(solution_str, data_source, ground_truth, extra_info, method='strict', format_score=0.1, score=1.):
    do_print = random.randint(1, 50) == 1

    extracted_answers = solution_str
    if do_print:
        print(f"Extracted answers: {extracted_answers}")
    format_reward, extracted_code = score_format(extracted_answers)
    if do_print:
        print(f"--------------------------------")
        print(f"Format reward: {format_reward}")
        print(f"Extracted code: {extracted_code}")

    # ground_truth = json.loads(ground_truth)
    # decide if should use json.loads automatically
    ground_truth = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    accuracy_reward = score_accuracy(extracted_code, ground_truth)
    return 0.1 * format_reward + 0.9 * accuracy_reward
    