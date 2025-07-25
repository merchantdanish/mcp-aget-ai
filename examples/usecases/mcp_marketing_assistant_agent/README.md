# MCP Marketing Content Agent

This example demonstrates a content creation agent that adapts to your brand voice and generates content optimized for different platforms. The agent uses an evaluation-driven approach and persistent memory to ensure consistent, high-quality communication.

## How It Works

1. **PromptManager**: Manages external prompt templates with variable substitution for different content types
2. **DocumentProcessor**: Handles document processing via markitdown MCP server for content samples and references
3. **URLContentFetcher**: Fetches and processes content from URLs using fetch MCP server
4. **ContextAssembler**: Gathers and organizes comprehensive context from various sources
5. **MarketingContentAgent**: Main agent that coordinates the content creation pipeline with quality evaluation
6. **EvaluatorOptimizer**: Ensures content meets quality standards through iterative refinement

The architecture focuses on maintaining brand consistency while optimizing content for different platforms through an evaluation-first approach.

```plaintext
┌──────────────┐      ┌───────────────┐      ┌───────────────┐
│   Content    │─────▶│   Context     │─────▶│   Content     │◀─┐
│   Request    │      │   Assembler   │      │   Creator     │  │
└──────────────┘      └───────────────┘      └───────────────┘  │
       │                     ▲                       │          │
       │                     │                       ▼          │
       │              ┌──────┴──────┐        ┌───────────────┐  │      
       │              │  Document   │        │   Quality     ├──┘
       │              │  Processor  │        │   Evaluator   │
       │              └─────────────┘        └───────────────┘
       │                     ▲                
       │              ┌──────┴──────┐
       └─────────────▶│   URL       │
                      │   Fetcher   │
                      └─────────────┘
```

## `1` App set up

First, clone the repo and navigate to the marketing agent example:

```bash
git clone https://github.com/lastmile-ai/mcp-agent.git
cd mcp-agent/examples/usecases/mcp_marketing_assistent_agent
```

Install `uv` (if you don't have it):

```bash
pip install uv
```

Sync `mcp-agent` project dependencies:

```bash
uv sync
```

Install the required MCP servers:

```bash
npm install -g @modelcontextprotocol/server-memory
pip install markitdown-mcp
```

## `2` Set up configuration and secrets

Copy and configure your secrets:

```bash
cp mcp_agent.secrets.yaml.example mcp_agent.secrets.yaml
```

Add your OpenAI API key to `mcp_agent.secrets.yaml`:

```yaml
openai:
  api_key: "YOUR_OPENAI_API_KEY"
```

Configure your brand voice in `company_config.yaml`:

```yaml
company:
  name: "Your Company"
  industry: "Your Industry"
  target_audience:
    - "Primary Audience"
    - "Secondary Audience"

platforms:
  linkedin:
    max_word_count: 100
    tone: "Professional but conversational"
  twitter:
    max_word_count: 50
    tone: "Sharp and witty"
```

## `3` Add content samples

Create directories for your content:

```bash
mkdir -p content_samples posts company_docs
```

Add your existing content to train the agent:
- `content_samples/`: Add social media posts, blog content
- `company_docs/`: Add brand guidelines, company info
- `posts/`: Where generated content will be saved

## `4` Run locally

Generate a LinkedIn post:

```bash
uv run main.py "Write a linkedin post about our new feature"
```

Create a Twitter thread:

```bash
uv run main.py "Create a twitter thread about our latest release"
```

Generate an email announcement:

```bash
uv run main.py "Draft an email about our upcoming webinar"
```

The agent will:
1. Load your brand configuration
2. Process relevant content samples
3. Generate platform-optimized content
4. Evaluate against quality standards
5. Save approved content to the posts directory
