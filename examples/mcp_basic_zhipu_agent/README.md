# MCP Zhipu AI Agent Example - "Finder" Agent

This example demonstrates how to create and run a basic "Finder" agent using Zhipu AI's GLM-4 model with MCP. The agent can access both `fetch` and `filesystem` MCP servers, allowing it to retrieve information from URLs and the local file system.

## Prerequisites

- Valid Zhipu AI API key
- Python 3.10+ environment

## Setup

Before running the agent, ensure you have:

1. Register and obtain a Zhipu AI API key:
   - Visit [Zhipu AI website](https://open.bigmodel.cn/) to register an account
   - Create an API key in the console

2. Configure the API key:
   - Create a `mcp_agent.secrets.yaml` file and add your API key:
     ```yaml
     zhipu:
       api_key: "your-zhipu-api-key-here"
     ```

## Running the Example

Install dependencies and run the example:

```bash
# Install dependencies
pip install -e ..

# Run the example
python main.py
```

## Example Features

This example demonstrates:

1. Using Zhipu AI's GLM-4 model within the MCP architecture
2. Retrieving web content via the fetch server
3. Reading local files via the filesystem server
4. Multi-turn conversation support
5. Support for prompts and responses in both English and Chinese

## Supported Models

Zhipu AI provides various large language models, including:

- glm-4 - Zhipu base large model
- GLM-4-Plus - Enhanced large model
- GLM-4-Long - Large model with longer context support
- GLM-4-FlashX-250414 - High-performance Flash model
- GLM-4-Flash-250414 - Standard Flash model
- GLM-4-Air-250414 - Lightweight large model
- glm-4v - Zhipu vision large model
- glm-3-turbo - Zhipu basic conversation model

This example uses the `glm-4` model by default, but you can change to any supported model in the configuration. 