"""OpenRouter or TypeSafe makes choices; an optional small OpenAI-compatible model writes field values."""

import json
import math
import os
import time

import httpx

from .questions import CHOICE_JSON, CHOICE_REPAIR, NEXT_ACTION, TARGET, TEXT_VALUE

CLIENT = httpx.Client(http2=True, timeout=25)
PROVIDERS = ("openrouter", "typesafe")
SCHEMA_FALLBACKS = set()
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL_DEFAULT = "inception/mercury-2.5"
TEXT_MODEL_DEFAULT = "inception/mercury-2.5"


def provider(required=True):
    """An explicit JEVA_PROVIDER wins; otherwise whichever key is present decides."""
    name = os.environ.get("JEVA_PROVIDER", "").strip().lower()
    if name:
        if name not in PROVIDERS:
            raise ValueError(f"JEVA_PROVIDER must be one of: {', '.join(PROVIDERS)}")
        return name
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.environ.get("TYPESAFE_API_KEY"):
        return "typesafe"
    if required:
        raise ValueError("Set OPENROUTER_API_KEY (or TYPESAFE_API_KEY) before choosing.")
    return "unconfigured"


def decision_model():
    if provider() == "openrouter":
        return os.environ.get("OPENROUTER_MODEL", OPENROUTER_MODEL_DEFAULT)
    return os.environ.get("TYPESAFE_MODEL", "jev-latest")


def openrouter_headers():
    """Attribution headers OpenRouter uses for ranking; sent only when configured."""
    headers = {}
    if os.environ.get("OPENROUTER_SITE_URL"):
        headers["HTTP-Referer"] = os.environ["OPENROUTER_SITE_URL"]
    if os.environ.get("OPENROUTER_APP_NAME"):
        headers["X-Title"] = os.environ["OPENROUTER_APP_NAME"]
    return headers


def post_json(url, key, body):
    for attempt in range(3):
        try:
            response = CLIENT.post(
                url, json=body, headers={"Authorization": f"Bearer {key}", **openrouter_headers()}
            )
        except httpx.HTTPError:
            raise RuntimeError("Model connection failed; no action executed.") from None
        if response.status_code in {429, 529, 503} and attempt < 2:
            time.sleep(0.5 * 2**attempt)
            continue
        if response.is_error:
            raise RuntimeError(f"Model provider returned HTTP {response.status_code}; no action executed.")
        return response.json()
    raise RuntimeError("Model unavailable")


def json_object(content):
    """A small JSON object, tolerating a fenced block even though response_format is requested."""
    text = content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.startswith("json") else text
    return json.loads(text)


def choice_schema(questions):
    """Strict schema so the decoder, not the model, guarantees every offered key is present."""

    def choice(keys):
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["choice", "confidence", "probabilities"],
            "properties": {
                "choice": {"type": "string", "enum": list(keys)},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "probabilities": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(keys),
                    "properties": {key: {"type": "number", "minimum": 0, "maximum": 1} for key in keys},
                },
            },
        }

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["answers"],
        "properties": {
            "answers": {
                "type": "object",
                "additionalProperties": False,
                "required": list(questions),
                "properties": {name: choice(question["criteria"]) for name, question in questions.items()},
            }
        },
    }


def response_format_for(model, questions):
    """Structured outputs pin the key sets; models without them fall back to plain JSON mode."""
    if os.environ.get("OPENROUTER_SCHEMA", "1") == "0" or model in SCHEMA_FALLBACKS:
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {"name": "choice", "strict": True, "schema": choice_schema(questions)},
    }


def reasoning_settings():
    """Only sent when configured, so models without reasoning support are unaffected."""
    mode = os.environ.get("OPENROUTER_REASONING", "").strip().lower()
    if not mode:
        return {}
    if mode == "none":
        return {"reasoning": {"enabled": False}}
    return {"reasoning": {"effort": mode}}


def decision_request(body, check=None):
    """One choice request. Both providers answer the same questions as {"answers", "model", "usage"}.

    OpenRouter gets one bounded repair attempt when its answer does not parse or does not satisfy the
    caller's checks. The repaired answer faces exactly the same validation; nothing is normalized.
    """
    if provider() == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY is required for the openrouter provider.")
        base = os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL).rstrip("/")
        messages = [
            {"role": "system", "content": CHOICE_JSON},
            {"role": "user", "content": json.dumps({"state": body["state"], "questions": body["questions"]})},
        ]
        last = "no answer received"
        for attempt in range(2):
            payload = {
                "model": body["model"],
                "max_tokens": int(os.environ.get("OPENROUTER_MAX_TOKENS", "2048")),
                "response_format": response_format_for(body["model"], body["questions"]),
                **reasoning_settings(),
                "messages": messages,
            }
            try:
                result = post_json(base + "/chat/completions", key, payload)
            except RuntimeError as error:
                # A provider that cannot honour the schema answers 4xx; the same request still works as JSON mode.
                if payload["response_format"]["type"] != "json_schema" or "HTTP 4" not in str(error):
                    raise
                SCHEMA_FALLBACKS.add(body["model"])
                payload["response_format"] = {"type": "json_object"}
                result = post_json(base + "/chat/completions", key, payload)
            content = ""
            try:
                content = result["choices"][0]["message"]["content"]
                answers = json_object(content)["answers"]
                if not isinstance(answers, dict) or not answers:
                    raise ValueError("answers must be a non-empty object")
                if check:
                    check(answers)
                return {
                    "answers": answers,
                    "model": result.get("model") or body["model"],
                    "usage": result.get("usage", {}),
                }
            except (KeyError, IndexError, TypeError, ValueError) as error:
                last = error
                if attempt:
                    break
                messages = [
                    *messages,
                    {"role": "assistant", "content": content[:4000]},
                    {"role": "user", "content": CHOICE_REPAIR.format(reason=error)},
                ]
        raise ValueError(f"OpenRouter returned no valid choice JSON ({last}); no action executed.")
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise ValueError("TYPESAFE_API_KEY is required for the typesafe provider.")
    result = post_json("https://api.typesafe.ai/v1/systemone", key, body)
    return {"answers": result["answers"], "model": result["model"], "usage": result.get("usage", {})}


def validate_choice(answer, ids):
    try:
        probabilities = answer["probabilities"]
        numbers = [*probabilities.values(), answer["confidence"]]
        valid = (
            answer["choice"] in ids
            and set(probabilities) == set(ids)
            and all(type(n) in (int, float) and math.isfinite(n) and 0 <= n <= 1 for n in numbers)
            and abs(sum(probabilities.values()) - 1) < 0.02
            and probabilities[answer["choice"]] >= max(probabilities.values()) - 1e-6
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Invalid model response; no action executed.")
    return answer


def action_space(actions):
    """One index per observed element; each operation has its own valid target choices."""
    elements, indices, targets, controls = [], {}, {}, {}
    operations = {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT"}
    for action in actions:
        kind = action["kind"]
        if kind not in operations:
            controls[action["id"].upper()] = action
            continue
        node = action["node"]
        if node not in indices:
            index = str(len(elements) + 1)
            indices[node] = index
            element = {k: action[k] for k in ("role", "value", "checked", "selected", "expanded") if k in action}
            element.update(index=index, label=action["label"].split(" → ")[0], operations=[])
            if kind == "select":
                element["value"] = action.get("current_value", "")
                element["options"] = []
            elements.append(element)
        index = indices[node]
        operation = operations[kind]
        group = targets.setdefault(operation, {})
        element = elements[int(index) - 1]
        if operation not in element["operations"]:
            element["operations"].append(operation)
        target = index
        if kind == "select":
            target = f"{index}:{len(element['options']) + 1}"
            element["options"].append({"index": target, "label": action["label"], "value": action["value"]})
        group[target] = action
    return elements, targets, controls


def choose(state, goal, history):
    elements, targets, controls = action_space(state["actions"])
    labels = {
        "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
        "TYPE_TEXT": "Enter or replace text in an editable field. A small LLM will supply the value from the goal.",
        "SELECT": "Select an observed dropdown value.",
    }
    operations = {key: labels[key] for key in targets}
    operations.update({key: value["label"] for key, value in controls.items()})
    operations.update(DONE="Every requirement is visibly satisfied.", BLOCKED="No supported operation can progress.")
    questions = {
        "operation": {"type": "choice", "criteria": operations, "instructions": {"goal": goal, "rules": NEXT_ACTION}}
    }
    for operation, candidates in targets.items():
        questions[operation.lower() + "_target"] = {
            "type": "choice",
            "criteria": {
                index: {
                    "element": f"[{index}] {a['label']}",
                    "current_value": a.get("current_value", a.get("value", "")),
                    **{k: a[k] for k in ("role", "checked", "selected", "expanded") if k in a},
                }
                for index, a in candidates.items()
            },
            "instructions": {"goal": goal, "operation": operation, "rules": [NEXT_ACTION, TARGET]},
        }
    body = {
        "model": decision_model(),
        "state": {
            "page": {k: state[k] for k in ("url", "title", "text")},
            "elements": elements,
            "recent_actions": [
                {k: h.get(k) for k in ("action", "kind", "text", "page_changed")} for h in history[-10:]
            ],
        },
        "questions": questions,
    }
    started = time.perf_counter()

    def check_answers(answers):
        try:
            selected = validate_choice(answers.get("operation", {}), operations)["choice"]
            if selected in targets:
                validate_choice(answers.get(selected.lower() + "_target", {}), targets[selected])
        except ValueError:
            required = {"operation": sorted(operations)}
            required.update({op.lower() + "_target": sorted(keys) for op, keys in targets.items()})
            raise ValueError(f"the object must carry exactly these keys: {json.dumps(required)}") from None

    result = decision_request(body, check_answers)
    operation_answer = validate_choice(result["answers"].get("operation", {}), operations)
    operation = operation_answer["choice"]
    target = None
    target_answer = None
    probabilities = {}
    if operation in targets:
        # Unused target heads cannot cause an action. Validate the head selected by the operation.
        target_answer = validate_choice(result["answers"].get(operation.lower() + "_target", {}), targets[operation])
        target = target_answer["choice"]
        choice = targets[operation][target]["id"]
        probabilities = {a["id"]: target_answer["probabilities"][index] for index, a in targets[operation].items()}
    else:
        choice = controls[operation]["id"] if operation in controls else operation
        probabilities[choice] = operation_answer["probabilities"][operation]
    return {
        "choice": choice,
        "operation": operation,
        "target": target,
        "confidence": operation_answer["confidence"],
        "probabilities": probabilities,
        "operation_probabilities": operation_answer["probabilities"],
        "target_probabilities": target_answer["probabilities"] if target_answer else {},
        "target_confidence": target_answer["confidence"] if target_answer else None,
        "raw_answers": result["answers"],
        "model": result["model"],
        "usage": result.get("usage", {}),
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "request": body,
    }


def field_context(goal, action, page, history):
    return {
        "goal": goal,
        "field": {k: action.get(k) for k in ("label", "role", "value")},
        "page": {"title": page["title"], "text": page["text"][:6000]},
        "recent_actions": [{k: h.get(k) for k in ("action", "text")} for h in history[-6:]],
    }


def field_text(context):
    key = os.environ.get("TEXT_MODEL_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError(
            "TYPE_TEXT needs TEXT_MODEL_API_KEY or OPENROUTER_API_KEY; "
            "no text is hardcoded or guessed by the executor."
        )
    base = os.environ.get("TEXT_MODEL_BASE_URL", OPENROUTER_BASE_URL).rstrip("/")
    model = os.environ.get("TEXT_MODEL", TEXT_MODEL_DEFAULT)
    reasoning = {"thinking": {"type": "disabled"}} if "api.deepseek.com/" in base else {"reasoning": {"effort": "low"}}
    if os.environ.get("TEXT_MODEL_REASONING") == "none":
        reasoning = {"reasoning": {"enabled": False}}
    started = time.perf_counter()
    result = post_json(
        base + "/chat/completions",
        key,
        {
            "model": model,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
            **reasoning,
            "messages": [
                {"role": "system", "content": TEXT_VALUE},
                {
                    "role": "user",
                    "content": json.dumps(context),
                },
            ],
        },
    )
    try:
        output = json.loads(result["choices"][0]["message"]["content"])
        value = output["text"]
        if set(output) != {"text"} or not isinstance(value, str) or not value.strip() or len(value) > 2000:
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        raise ValueError("Text helper returned no valid field value; nothing typed.") from None
    return value, {
        "model": model,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "usage": result.get("usage", {}),
    }
