from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Union, Any
import os
import re
import json
import importlib
from glob import glob

from mcp_agent.agents.agent import Agent
from mcp_agent.agents.agent_spec import AgentSpec
from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.llm_selector import ModelSelector
from mcp_agent.workflows.router.router_llm import LLMRouter
from mcp_agent.workflows.parallel.parallel_llm import ParallelLLM
from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import (
    EvaluatorOptimizerLLM,
)
from mcp_agent.workflows.orchestrator.orchestrator import (
    Orchestrator,
    OrchestratorOverrides,
)
from mcp_agent.workflows.deep_orchestrator.config import DeepOrchestratorConfig
from mcp_agent.workflows.deep_orchestrator.orchestrator import DeepOrchestrator
from mcp_agent.workflows.swarm.swarm import Swarm, SwarmAgent
from mcp_agent.workflows.intent_classifier.intent_classifier_base import Intent
from mcp.types import ModelPreferences


def _agent_from_spec(spec: AgentSpec, context=None) -> Agent:
    # Import locally to avoid circular import at module import time
    from mcp_agent.agents.agent import Agent as _Agent

    return _Agent(
        name=spec.name,
        instruction=spec.instruction,
        server_names=spec.server_names or [],
        functions=spec.functions or [],
        connection_persistence=spec.connection_persistence,
        human_input_callback=spec.human_input_callback,
        context=context,
    )


def _parse_model_identifier(model_id: str) -> tuple[Optional[str], str]:
    """Parse a model identifier that may be prefixed with provider (e.g., 'openai:gpt-4o')."""
    if ":" in model_id:
        prov, name = model_id.split(":", 1)
        return (prov.strip().lower() or None, name.strip())
    return (None, model_id)


def _select_provider_and_model(
    *,
    provider: Optional[str],
    model_preferences: Optional[Union[str, ModelPreferences]],
    context=None,
) -> tuple[str, Optional[str]]:
    """Return (provider, model_name) using a string model id or ModelSelector.

    - If model_preferences is a str, treat it as model id; allow 'provider:model' pattern.
    - If it's a ModelPreferences, use ModelSelector.
    - Otherwise, return default provider and no model.
    """
    prov = (provider or "openai").lower()
    if isinstance(model_preferences, str):
        inferred_provider, model_name = _parse_model_identifier(model_preferences)
        return (inferred_provider or prov, model_name)
    if isinstance(model_preferences, ModelPreferences):
        selector = ModelSelector(context=context)
        info = selector.select_best_model(
            model_preferences=model_preferences, provider=prov
        )
        return (info.provider.lower(), info.name)
    return (prov, None)


def _get_provider_class(prov: str):
    p = prov.lower()
    if p == "openai":
        from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

        return OpenAIAugmentedLLM
    if p == "anthropic":
        from mcp_agent.workflows.llm.augmented_llm_anthropic import (
            AnthropicAugmentedLLM,
        )

        return AnthropicAugmentedLLM
    if p == "azure":
        from mcp_agent.workflows.llm.augmented_llm_azure import AzureAugmentedLLM

        return AzureAugmentedLLM
    if p == "google":
        from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM

        return GoogleAugmentedLLM
    if p == "bedrock":
        from mcp_agent.workflows.llm.augmented_llm_bedrock import BedrockAugmentedLLM

        return BedrockAugmentedLLM
    if p == "ollama":
        from mcp_agent.workflows.llm.augmented_llm_ollama import OllamaAugmentedLLM

        return OllamaAugmentedLLM
    raise ValueError(f"Unsupported provider: {prov}")


def _llm_factory(
    *,
    provider: Optional[str],
    model_preferences: Optional[Union[str, ModelPreferences]],
    context=None,
) -> Callable[[Agent], AugmentedLLM]:
    prov, model_name = _select_provider_and_model(
        provider=provider, model_preferences=model_preferences, context=context
    )
    provider_cls = _get_provider_class(prov)

    def _default_params() -> RequestParams | None:
        if model_name and isinstance(model_preferences, ModelPreferences):
            return RequestParams(model=model_name, modelPreferences=model_preferences)
        if model_name and isinstance(model_preferences, str):
            return RequestParams(model=model_name)
        if isinstance(model_preferences, ModelPreferences):
            return RequestParams(modelPreferences=model_preferences)
        return None

    return lambda agent: provider_cls(
        agent=agent, default_request_params=_default_params(), context=context
    )


def create_llm(
    agent_name: str,
    server_names: Optional[List[str]] = None,
    instruction: Optional[str] = None,
    provider: Optional[str] = "openai",
    model_preferences: Optional[Union[str, ModelPreferences]] = None,
    context=None,
) -> AugmentedLLM:
    agent = _agent_from_spec(
        AgentSpec(
            name=agent_name, instruction=instruction, server_names=server_names or []
        ),
        context=context,
    )
    factory = _llm_factory(
        provider=provider, model_preferences=model_preferences, context=context
    )
    return factory(agent=agent)


async def create_router_llm(
    *,
    server_names: Optional[List[str]] = None,
    agents: Optional[List[AgentSpec | AugmentedLLM]] = None,
    functions: Optional[List[Callable]] = None,
    routing_instruction: Optional[str] = None,
    provider: str = "openai",
    context=None,
    **kwargs,
) -> LLMRouter:
    normalized_agents: List[Agent] = []
    for a in agents or []:
        if isinstance(a, AgentSpec):
            normalized_agents.append(_agent_from_spec(a, context=context))
        else:
            normalized_agents.append(a.agent)
    if provider.lower() == "openai":
        from mcp_agent.workflows.router.router_llm_openai import OpenAILLMRouter

        return await OpenAILLMRouter.create(
            server_names=server_names,
            agents=normalized_agents,
            functions=functions,
            routing_instruction=routing_instruction,
            context=context,
            **kwargs,
        )
    elif provider.lower() == "anthropic":
        from mcp_agent.workflows.router.router_llm_anthropic import AnthropicLLMRouter

        return await AnthropicLLMRouter.create(
            server_names=server_names,
            agents=normalized_agents,
            functions=functions,
            routing_instruction=routing_instruction,
            context=context,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


async def create_embedding_router(
    *,
    provider: str = "openai",
    server_names: Optional[List[str]] = None,
    agents: Optional[List[AgentSpec | AugmentedLLM]] = None,
    functions: Optional[List[Callable]] = None,
    context=None,
):
    normalized_agents: List[Agent] = []
    for a in agents or []:
        if isinstance(a, AgentSpec):
            normalized_agents.append(_agent_from_spec(a, context=context))
        else:
            normalized_agents.append(a.agent)
    prov = provider.lower()
    if prov == "openai":
        from mcp_agent.workflows.router.router_embedding_openai import (
            OpenAIEmbeddingRouter,
        )

        return await OpenAIEmbeddingRouter.create(
            server_names=server_names,
            agents=normalized_agents,
            functions=functions,
            context=context,
        )
    if prov == "cohere":
        from mcp_agent.workflows.router.router_embedding_cohere import (
            CohereEmbeddingRouter,
        )

        return await CohereEmbeddingRouter.create(
            server_names=server_names,
            agents=normalized_agents,
            functions=functions,
            context=context,
        )
    raise ValueError(f"Unsupported provider: {provider}")


def create_orchestrator(
    *,
    available_agents: Sequence[AgentSpec | AugmentedLLM],
    plan_type: str = "full",
    provider: Optional[str] = "openai",
    model_preferences: Optional[ModelPreferences] = None,
    overrides: OrchestratorOverrides | None = None,
    name: Optional[str] = None,
    context=None,
    **kwargs,
) -> Orchestrator:
    factory = _llm_factory(
        provider=provider, model_preferences=model_preferences, context=context
    )
    normalized: List[Agent | AugmentedLLM] = []
    for item in available_agents:
        if isinstance(item, AgentSpec):
            normalized.append(_agent_from_spec(item, context=context))
        else:
            normalized.append(item)
    return Orchestrator(
        llm_factory=factory,
        name=name,
        plan_type=plan_type,  # type: ignore[arg-type]
        available_agents=normalized,
        context=context,
        overrides=overrides,
        **kwargs,
    )


def create_deep_orchestrator(
    *,
    available_agents: Sequence[AgentSpec | AugmentedLLM],
    config: Optional[DeepOrchestratorConfig] = None,
    provider: Optional[str] = "openai",
    model_preferences: Optional[ModelPreferences] = None,
    context=None,
    **kwargs,
) -> DeepOrchestrator:
    factory = _llm_factory(
        provider=provider, model_preferences=model_preferences, context=context
    )
    agents_mixed: List[Agent | AugmentedLLM] = []
    for item in available_agents:
        if isinstance(item, AgentSpec):
            agents_mixed.append(_agent_from_spec(item, context=context))
        else:
            agents_mixed.append(item)
    if config is None:
        config = DeepOrchestratorConfig()
    # Inject available agents
    config.available_agents = agents_mixed
    return DeepOrchestrator(
        llm_factory=factory,
        config=config,
        context=context,
        **kwargs,
    )


def create_parallel_llm(
    *,
    fan_in: Union[AugmentedLLM, Callable[[Any], Any], AgentSpec],
    fan_out: Optional[List[Union[AugmentedLLM, AgentSpec, Callable]]] = None,
    provider: Optional[str] = "openai",
    model_preferences: Optional[ModelPreferences] = None,
    context=None,
    **kwargs,
) -> ParallelLLM:
    factory = _llm_factory(
        provider=provider, model_preferences=model_preferences, context=context
    )
    fan_in_agent_or_llm: Union[AugmentedLLM, Agent, Callable[[Any], Any]]
    if isinstance(fan_in, AgentSpec):
        fan_in_agent_or_llm = _agent_from_spec(fan_in, context=context)
    else:
        fan_in_agent_or_llm = fan_in  # already AugmentedLLM or callable
    fan_out_agents: List[Agent | AugmentedLLM] = []
    fan_out_functions: List[Callable] = []
    for item in fan_out or []:
        if callable(item) and not isinstance(item, AugmentedLLM):
            fan_out_functions.append(item)  # function
        elif isinstance(item, AgentSpec):
            fan_out_agents.append(_agent_from_spec(item, context=context))
        else:
            fan_out_agents.append(item)  # AugmentedLLM

    return ParallelLLM(
        fan_in_agent=fan_in_agent_or_llm,  # type: ignore[arg-type]
        fan_out_agents=fan_out_agents or None,  # accepts Agents or LLMs
        fan_out_functions=fan_out_functions or None,
        llm_factory=factory,
        context=context,
        **kwargs,
    )


def create_evaluator_optimizer_llm(
    *,
    optimizer: Union[AugmentedLLM, AgentSpec],
    evaluator: Union[str, AugmentedLLM, AgentSpec],
    min_rating=None,
    max_refinements: int = 3,
    provider: Optional[str] = "openai",
    model_preferences: Optional[ModelPreferences] = None,
    context=None,
    **kwargs,
) -> EvaluatorOptimizerLLM:
    factory = _llm_factory(
        provider=provider, model_preferences=model_preferences, context=context
    )
    optimizer_obj: Union[AugmentedLLM, Agent]
    evaluator_obj: Union[str, AugmentedLLM, Agent]

    optimizer_obj = (
        _agent_from_spec(optimizer, context=context)
        if isinstance(optimizer, AgentSpec)
        else optimizer
    )
    if isinstance(evaluator, AgentSpec):
        evaluator_obj = _agent_from_spec(evaluator, context=context)
    else:
        evaluator_obj = evaluator

    return EvaluatorOptimizerLLM(
        optimizer=optimizer_obj,
        evaluator=evaluator_obj,
        min_rating=min_rating,
        max_refinements=max_refinements,
        llm_factory=factory,
        context=context,
        **kwargs,
    )


def create_swarm(
    *,
    name: str,
    instruction: Optional[Union[str, Callable[[dict], str]]] = None,
    server_names: Optional[List[str]] = None,
    functions: Optional[List[Callable]] = None,
    provider: str = "openai",
    context=None,
) -> Swarm:
    swarm_agent = SwarmAgent(
        name=name,
        instruction=instruction or "You are a helpful agent.",
        server_names=server_names or [],
        functions=functions or [],
        context=context,
    )
    if provider.lower() == "openai":
        from mcp_agent.workflows.swarm.swarm_openai import OpenAISwarm

        return OpenAISwarm(agent=swarm_agent)
    if provider.lower() == "anthropic":
        from mcp_agent.workflows.swarm.swarm_anthropic import AnthropicSwarm

        return AnthropicSwarm(agent=swarm_agent)
    raise ValueError(f"Unsupported provider: {provider}")


async def create_intent_classifier_llm(
    *,
    intents: List[Intent],
    provider: str = "openai",
    classification_instruction: Optional[str] = None,
    name: Optional[str] = None,
    context=None,
):
    prov = provider.lower()
    if prov == "openai":
        from mcp_agent.workflows.intent_classifier.intent_classifier_llm_openai import (
            OpenAILLMIntentClassifier,
        )

        llm_cls = _get_provider_class(prov)
        return await OpenAILLMIntentClassifier.create(
            llm=llm_cls(
                name=name, instruction=classification_instruction, context=context
            ),
            intents=intents,
            classification_instruction=classification_instruction,
            name=name,
            context=context,
        )
    if prov == "anthropic":
        from mcp_agent.workflows.intent_classifier.intent_classifier_llm_anthropic import (
            AnthropicLLMIntentClassifier,
        )

        llm_cls = _get_provider_class(prov)
        return await AnthropicLLMIntentClassifier.create(
            llm=llm_cls(
                name=name, instruction=classification_instruction, context=context
            ),
            intents=intents,
            classification_instruction=classification_instruction,
            name=name,
            context=context,
        )
    raise ValueError(f"Unsupported provider: {provider}")


# ------------------------- AgentSpec Loaders ----------------------------------


def _resolve_callable(ref: str) -> Callable:
    """Resolve a dotted reference 'package.module:attr' to a callable.
    Raises ValueError if not found or not callable.
    """
    if not isinstance(ref, str) or (":" not in ref and "." not in ref):
        raise ValueError(f"Invalid callable reference: {ref}")
    module_name, attr = ref.split(":", 1) if ":" in ref else ref.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    obj = getattr(mod, attr)
    if not callable(obj):
        raise ValueError(f"Referenced object is not callable: {ref}")
    return obj


def _normalize_agents_data(data: Any) -> list[dict]:
    """Normalize arbitrary parsed data into a list of agent dicts.

    Accepts:
      - {'agents': [...]} or {'agent': {...}} or a list of agents or a single agent dict
    """
    if data is None:
        return []
    if isinstance(data, dict):
        if "agents" in data and isinstance(data["agents"], list):
            return data["agents"]
        if "agent" in data and isinstance(data["agent"], dict):
            return [data["agent"]]
        # If the dict looks like a single agent (has a name), treat it as one
        if "name" in data:
            return [data]
        return []
    if isinstance(data, list):
        return data
    return []


def _agent_spec_from_dict(
    obj: dict, context=None, *, default_instruction: Optional[str] = None
) -> AgentSpec:
    name = obj.get("name")
    if not name:
        raise ValueError("AgentSpec requires a 'name'")
    instruction = obj.get("instruction")
    # If no explicit instruction, fall back to 'description' or provided default body text
    if not instruction:
        desc = obj.get("description")
        if default_instruction and desc:
            instruction = f"{desc}\n\n{default_instruction}".strip()
        else:
            instruction = default_instruction or desc
    server_names = obj.get("server_names") or obj.get("servers") or []
    connection_persistence = obj.get("connection_persistence", True)
    functions = obj.get("functions", [])
    # If no servers provided, consider 'tools' as a hint for server names
    if not server_names and "tools" in obj:
        tools_val = obj.get("tools")
        if isinstance(tools_val, str):
            server_names = [t.strip() for t in tools_val.split(",") if t.strip()]
        elif isinstance(tools_val, list):
            server_names = [str(t).strip() for t in tools_val if str(t).strip()]
    resolved_functions: list[Callable] = []
    for f in functions:
        if callable(f):
            resolved_functions.append(f)
        elif isinstance(f, str):
            resolved_functions.append(_resolve_callable(f))
        else:
            raise ValueError(f"Unsupported function entry: {f}")
    human_cb = obj.get("human_input_callback")
    if isinstance(human_cb, str):
        human_cb = _resolve_callable(human_cb)

    return AgentSpec(
        name=name,
        instruction=instruction,
        server_names=list(server_names),
        functions=resolved_functions,
        connection_persistence=connection_persistence,
        human_input_callback=human_cb,
    )


def _load_yaml(text: str) -> Any:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise ImportError("PyYAML is required to load YAML agent specs") from e
    return yaml.safe_load(text)


def _extract_front_matter_md(text: str) -> Optional[str]:
    """Extract YAML front-matter delimited by --- at the top of a Markdown file."""
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            return text[4:end]
    return None


def _extract_front_matter_and_body_md(text: str) -> tuple[Optional[str], str]:
    """Return (front_matter_yaml, body_text)."""
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            fm = text[4:end]
            body = text[end + len("\n---") :].lstrip("\n")
            return fm, body
    return None, text


def _extract_code_blocks_md(text: str) -> list[tuple[str, str]]:
    """Return list of (lang, code) for fenced code blocks."""
    pattern = re.compile(r"```(\w+)?\n([\s\S]*?)```", re.MULTILINE)
    return [(m.group(1) or "", m.group(2)) for m in pattern.finditer(text)]


def load_agent_specs_from_text(
    text: str, *, fmt: Optional[str] = None, context=None
) -> List[AgentSpec]:
    """Load AgentSpec list from text in yaml/json/md.

    - YAML: either a list or {'agents': [...]}
    - JSON: same as YAML
    - Markdown: supports YAML front-matter or fenced code blocks with yaml/json containing agents
    """
    specs: list[AgentSpec] = []
    fmt_lower = (fmt or "").lower()
    try_parsers = []
    if fmt_lower in ("yaml", "yml"):
        try_parsers = [lambda t: _load_yaml(t)]
    elif fmt_lower == "json":
        try_parsers = [lambda t: json.loads(t)]
    elif fmt_lower == "md":
        fm, body = _extract_front_matter_and_body_md(text)
        if fm is not None:
            try_parsers.append(lambda _t, fm=fm: ("__FM__", _load_yaml(fm), body))
        for lang, code in _extract_code_blocks_md(text):
            lang = (lang or "").lower()
            if lang in ("yaml", "yml"):
                try_parsers.append(
                    lambda _t, code=code: ("__YAML__", _load_yaml(code), "")
                )
            elif lang == "json":
                try_parsers.append(
                    lambda _t, code=code: ("__JSON__", json.loads(code), "")
                )
    else:
        # Try yaml then json by default
        try_parsers = [lambda t: _load_yaml(t), lambda t: json.loads(t)]

    for parser in try_parsers:
        try:
            data = parser(text)
        except Exception:
            continue
        body_text: Optional[str] = None
        if (
            isinstance(data, tuple)
            and len(data) == 3
            and isinstance(data[1], (dict, list))
        ):
            # Markdown parser variant returned (tag, parsed, body)
            _, parsed, body_text = data
            data = parsed

        agents_data = _normalize_agents_data(data)
        for obj in agents_data:
            try:
                specs.append(
                    _agent_spec_from_dict(
                        obj, context=context, default_instruction=body_text
                    )
                )
            except Exception:
                continue
        if specs:
            break
    return specs


def load_agent_specs_from_file(path: str, context=None) -> List[AgentSpec]:
    ext = os.path.splitext(path)[1].lower()
    fmt = None
    if ext in (".yaml", ".yml"):
        fmt = "yaml"
    elif ext == ".json":
        fmt = "json"
    elif ext in (".md", ".markdown"):
        fmt = "md"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return load_agent_specs_from_text(text, fmt=fmt, context=context)


def load_agent_specs_from_dir(
    path: str, pattern: str = "**/*.*", context=None
) -> List[AgentSpec]:
    """Load AgentSpec list by scanning a directory for yaml/json/md files."""
    results: List[AgentSpec] = []
    for fp in glob(os.path.join(path, pattern), recursive=True):
        if os.path.isdir(fp):
            continue
        ext = os.path.splitext(fp)[1].lower()
        if ext not in (".yaml", ".yml", ".json", ".md", ".markdown"):
            continue
        try:
            results.extend(load_agent_specs_from_file(fp, context=context))
        except Exception:
            continue
    return results


async def create_intent_classifier_embedding(
    *,
    intents: List[Intent],
    provider: str = "openai",
    context=None,
):
    if provider.lower() == "openai":
        from mcp_agent.workflows.intent_classifier.intent_classifier_embedding_openai import (
            OpenAIEmbeddingIntentClassifier,
        )

        return await OpenAIEmbeddingIntentClassifier.create(
            intents=intents, context=context
        )
    if provider.lower() == "cohere":
        from mcp_agent.workflows.intent_classifier.intent_classifier_embedding_cohere import (
            CohereEmbeddingIntentClassifier,
        )

        return await CohereEmbeddingIntentClassifier.create(
            intents=intents, context=context
        )
    raise ValueError(f"Unsupported provider: {provider}")
