# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import Any, Optional, Tuple
from uuid import uuid4
import numpy as np
from asteval import Interpreter
import json
from verl.utils.reward_score import gsm8k

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
aeval = Interpreter()

def evaluate(expression: str) -> str:
    """
    Evaluate a Python expression using asteval, then convert any NumPy
    types into plain Python types so that json.dumps won’t fail.
    Returns a JSON string:
      - On success: {"result": <python_value>}
      - On failure: {"error": "Invalid expression: <message>"}
    """
    def _normalize(obj):
        # NumPy scalar (e.g. int64, float64) → Python scalar
        if isinstance(obj, np.generic):
            return obj.item()
        # NumPy array → nested Python lists
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Built-in containers: recurse
        if isinstance(obj, list):
            return [_normalize(i) for i in obj]
        if isinstance(obj, tuple):
            return tuple(_normalize(i) for i in obj)
        if isinstance(obj, dict):
            return { _normalize(k): _normalize(v) for k, v in obj.items() }
        # Everything else (int, float, str, etc.) is already JSONable
        return obj

    try:
        raw = aeval(expression)
        result = _normalize(raw)
        return json.dumps({"result": result}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Invalid expression: {e}"}, ensure_ascii=False)
    
class ExecTool(BaseTool):
    """A demo tool for calculating the reward of gsm8k.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "evaluate",
                "description": (
                    "Evaluate a Python expression (including arithmetic, list indexing, slicing, "
                    "and other pure expressions) and return its result."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["expression"],
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "A Python expression to evaluate, e.g. '2+3*4' or '[1,2,3][1:3]'"
                        }
                    }
                }
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        expression = parameters.get("expression", "")
        answer = evaluate(expression)
        print(f"Evaluating expression: {expression}")
        return answer, 0, {}


    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
