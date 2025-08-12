# import os
# from textwrap import dedent

# import pytest

# from mcp_agent.workflows.factory import (
#     AgentSpec,
#     load_agent_specs_from_text,
#     load_agent_specs_from_file,
#     load_agent_specs_from_dir,
# )


# def sample_fn():
#     return "ok"


# def test_yaml_agents_list_parses_agentspecs(tmp_path):
#     yaml_text = dedent(
#         """
#         agents:
#           - name: finder
#             instruction: You can read files
#             server_names: [filesystem]
#           - name: fetcher
#             servers: [fetch]
#             instruction: You can fetch URLs
#         """
#     )
#     specs = load_agent_specs_from_text(yaml_text, fmt="yaml")
#     assert len(specs) == 2
#     assert isinstance(specs[0], AgentSpec)
#     assert specs[0].name == "finder"
#     assert specs[0].instruction == "You can read files"
#     assert specs[0].server_names == ["filesystem"]
#     assert specs[1].name == "fetcher"
#     assert specs[1].server_names == ["fetch"]


# def test_json_single_agent_object(tmp_path):
#     json_text = dedent(
#         """
#         {"agent": {"name": "coder", "instruction": "Modify code", "servers": ["filesystem"]}}
#         """
#     )
#     specs = load_agent_specs_from_text(json_text, fmt="json")
#     assert len(specs) == 1
#     spec = specs[0]
#     assert spec.name == "coder"
#     assert spec.instruction == "Modify code"
#     assert spec.server_names == ["filesystem"]


# def test_markdown_front_matter_and_body_merges_instruction():
#     md_text = dedent(
#         """
#         ---
#         name: code-reviewer
#         description: Expert code reviewer, use proactively
#         tools: filesystem, fetch
#         ---

#         You are a senior code reviewer ensuring high standards.

#         Provide feedback organized by priority.
#         """
#     )
#     specs = load_agent_specs_from_text(md_text, fmt="md")
#     assert len(specs) == 1
#     spec = specs[0]
#     assert spec.name == "code-reviewer"
#     # instruction should combine description + body when explicit instruction absent
#     assert "Expert code reviewer" in (spec.instruction or "")
#     assert "senior code reviewer" in (spec.instruction or "")
#     # tools map to server_names if servers/server_names absent
#     assert spec.server_names == ["filesystem", "fetch"]


# def test_markdown_code_blocks_yaml_and_json():
#     md_text = dedent(
#         """
#         Here are some agents:

#         ```yaml
#         agents:
#           - name: a
#             servers: [filesystem]
#         ```

#         And some JSON:

#         ```json
#         {"agent": {"name": "b", "servers": ["fetch"]}}
#         ```
#         """
#     )
#     specs = load_agent_specs_from_text(md_text, fmt="md")
#     # At least one should be parsed from either block
#     assert any(s.name == "a" for s in specs) or any(s.name == "b" for s in specs)


# def test_functions_resolution_with_dotted_ref(tmp_path, monkeypatch):
#     yaml_text = dedent(
#         f"""
#         agents:
#           - name: tools-agent
#             servers: [filesystem]
#             functions:
#               - "tests.workflows.test_agentspec_loader:sample_fn"
#         """
#     )
#     specs = load_agent_specs_from_text(yaml_text, fmt="yaml")
#     assert len(specs) == 1
#     spec = specs[0]
#     assert len(spec.functions) == 1
#     assert spec.functions[0]() == "ok"


# def test_load_agents_from_dir(tmp_path):
#     # create multiple files in a temp directory
#     (tmp_path / "agents.yaml").write_text(
#         dedent(
#             """
#             agents:
#               - name: one
#                 servers: [filesystem]
#               - name: two
#                 servers: [fetch]
#             """
#         ),
#         encoding="utf-8",
#     )
#     (tmp_path / "agent.json").write_text(
#         "{""agent"": {""name"": "json-agent", ""servers"": [""fetch""]}}",
#         encoding="utf-8",
#     )
#     specs = load_agent_specs_from_dir(str(tmp_path))
#     names = {s.name for s in specs}
#     assert {"one", "two", "json-agent"}.issubset(names)




