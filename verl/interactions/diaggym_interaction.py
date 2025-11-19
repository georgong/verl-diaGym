from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
import os
import json
from .base import BaseInteraction
from openai import OpenAI

SEP = "<SEP>"
stop_tokens = [SEP, "<endoftext>"]

API_KEY = os.getenv("OPENAI_API_KEY", "None")
API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:8079/v1")


class DiagGym:
    def __init__(self, model_name_or_path: str, api_key: str = API_KEY, api_base: str = API_BASE) -> None:
        self.model_name_or_path = model_name_or_path
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def simulate(self, context: str, past_events_list: list, exam_name: str) -> Optional[str]:
        """
        Generate exam results based on patient context and past events.
        """
        context = context.rstrip() + "\nThe following summarizes the results from the patient's medical examination:\n"
        
        if len(past_events_list) == 0:
            input_prompt = (
                context
                + "Exam name:\n" + exam_name
                + "\nExam results:\n"
            )
        else:
            past_events_str = [
                f"Exam name:\n{event_name}\nExam results:\n{resp}"
                for (event_name, resp) in past_events_list
            ]
            input_prompt = (
                context
                + SEP.join(past_events_str)
                + SEP
                + "Exam name:\n" + exam_name
                + "\nExam results:\n"
            )
        
        response = self.client.completions.create(
            model=self.model_name_or_path,
            prompt=input_prompt,
            max_tokens=8192,
            temperature=1.0,
            stop=stop_tokens
        )
        return response.choices[0].text.strip()


class DiagGymInteraction(BaseInteraction):
    """
    Stateful wrapper around DiagGym.

    Internal state layout:
        diag_dict[instance_id] = {
            "context": str,                  # patient-level context
            "past_events_list": List[Tuple[str, str]],  # (exam_name, exam_result)
            "target": Any,                   # ground-truth diagnosis or label
            "score": float,                  # accumulated reward (placeholder)
            "done": bool,                    # whether diagnosis has been completed
        }
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        # Initialize DiagGym once; shared across instances.
        model_name = self.config.get("name", "DiagGym")
        self.gym = DiagGym(model_name)
        # Per-instance diagnostic state.
        self.diag_dict: Dict[str, Dict[str, Any]] = {}

    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """
        Create a new interaction instance and store initial state.

        Expected kwargs:
            context: str
                Initial patient context.
            past_events_list: list[tuple[str, str]]
                List of (exam_name, exam_result) for previously completed exams.
            target: Any
                Ground-truth diagnosis / label for this case.
        """
        if instance_id is None:
            instance_id = str(uuid4())

        context: str = kwargs.get("context", "") or ""
        past_events_list: list = kwargs.get("past_events_list", [])
        if not isinstance(past_events_list, list):
            past_events_list = []
        target = kwargs.get("target", None)

        self.diag_dict[instance_id] = {
            "context": context,
            "past_events_list": list(past_events_list),
            "target": target,
            "score": 0.0,
            "done": False,
        }

        return instance_id

    # ---------- Abstract parsing / processing hooks ---------- #

    def _parse_messages(self, messages: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
        """
        Expected return:
            {
                "exam_name": Optional[str],
                "diagnosis": Optional[str],
            }
        """
        last = messages[-1]["content"]
        try:
            data = json.loads(last)
        except Exception:
            data = {}

        if data.get("type") == "request_exam":
            return {"exam_name": data.get("exam_name"), "diagnosis": None}
        elif data.get("type") == "make_diagnosis":
            return {"exam_name": None, "diagnosis": data.get("diagnosis")}
        else:
            return {"exam_name": None, "diagnosis": None}

    def _process_exam_result(self, exam_name: str, raw_result: str) -> Tuple[str, str]:
        """
        Post-process raw exam result text into a structured event.

        Typical return value:
            (exam_name, raw_result)

        Downstream code may override this to perform summarization,
        normalization, or conversion into a richer structure.
        """
        return exam_name, raw_result

    # ---------- Main interaction entrypoint ---------- #

    async def generate_response(
        self,
        instance_id: str,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> Tuple[bool, str, float, Dict[str, Any]]:
        """
        Run one interaction turn.

        Turn-level semantics:
            1. Messages are parsed via _parse_messages:
               - If "diagnosis" is present:
                    * Diagnosis is stored.
                    * Instance is marked as done.
                    * Sequence is terminated (should_terminate_sequence = True).
               - Else if "exam_name" is present:
                    * A new exam is simulated via DiagGym.
                    * Result is processed and appended to past_events_list.
                    * Sequence continues.
               - Else:
                    * No-op response is returned.

            2. Reward is not computed here. current_turn_score is kept as 0.0
               and is expected to be filled by an external reward function.
        """
        if instance_id not in self.diag_dict:
            raise KeyError(f"Instance id {instance_id} not found in diag_dict. start_interaction may be missing.")

        state = self.diag_dict[instance_id]
        context: str = state["context"]
        past_events_list: list = state["past_events_list"]

        parsed = self._parse_messages(messages)
        exam_name: Optional[str] = parsed.get("exam_name")
        diagnosis: Optional[str] = parsed.get("diagnosis")

        should_terminate_sequence: bool = False
        response_content: str
        current_turn_score: float = 0.0
        additional_data: Dict[str, Any] = {}

        # Case 2: final diagnosis has been produced.
        if diagnosis:
            state["final_diagnosis"] = diagnosis
            state["done"] = True

            response_content = diagnosis
            should_terminate_sequence = True

            additional_data = {
                "type": "diagnosis",
                "diagnosis": diagnosis,
                "target": state.get("target"),
            }
            return should_terminate_sequence, response_content, current_turn_score, additional_data

        # Case 1: a new exam should be performed.
        if exam_name:
            raw_result = self.gym.simulate(
                context=context,
                past_events_list=past_events_list,
                exam_name=exam_name,
            )

            if raw_result is None:
                response_content = "No exam result could be generated at this time."
                additional_data = {
                    "type": "exam",
                    "exam_name": exam_name,
                    "raw_result": None,
                }
                return should_terminate_sequence, response_content, current_turn_score, additional_data

            event = self._process_exam_result(exam_name, raw_result)
            past_events_list.append(event)
            state["past_events_list"] = past_events_list

            response_content = raw_result
            additional_data = {
                "type": "exam",
                "exam_name": exam_name,
                "raw_result": raw_result,
                "num_past_events": len(past_events_list),
            }
            return should_terminate_sequence, response_content, current_turn_score, additional_data

        # Fallback: no exam_name / diagnosis detected.
        response_content = "No valid exam request or diagnosis was detected in this turn."
        additional_data = {
            "type": "noop",
        }
        return should_terminate_sequence, response_content, current_turn_score, additional_data

    async def calculate_score(self, instance_id: str, **kwargs: Any) -> float:
        """
        Return the current score associated with the given instance.

        This method is intended to be overridden or extended with
        task-specific reward computation logic.
        """
        state = self.diag_dict.get(instance_id)
        if not state:
            return 0.0
        return float(state.get("score", 0.0))

    async def finalize_interaction(self, instance_id: str, **kwargs: Any) -> None:
        """
        Finalize an interaction after reward has been computed.

        Currently:
            - Logs a completion message.
            - Removes the instance state from diag_dict.
        """
        print(f"diagnose complete for {instance_id}")
        self.diag_dict.pop(instance_id, None)
