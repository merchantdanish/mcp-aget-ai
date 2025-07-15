# MCP Research & Analysis Agent - Real Estate Example

This example demonstrates a configurable research and analysis agent that can be adapted for any domain expertise. Simply replace the agent instructions to create specialized research workflows for finance, healthcare, legal, marketing, or any other field requiring data collection, quality verification, and report generation.

## How It Works

This is a universal research agent framework with pluggable expertise:

1. **Interactive Elicitation**: Asks domain-specific questions to understand user needs
2. **API Integration**: Connects to relevant data sources (RentSpider for real estate, but easily swappable)
3. **Quality Control**: Uses EvaluatorOptimizer to ensure research meets excellence standards
4. **Expert Analysis**: Provides domain-specific insights and recommendations
5. **Report Generation**: Creates comprehensive reports tailored to the field

**Adaptable to any domain**: Change the agent instructions and API integrations to create research agents for finance, healthcare, legal research, market analysis, academic research, or any other expertise area.

```plaintext
┌──────────────┐      ┌───────────────────┐      ┌───────────┐
│ Orchestrator │─────▶│ Research          │─────▶│ Research  │◀─────┐
│ Workflow     │      │ Quality Controller│      │ Agent     │      │
└──────────────┘      └───────────────────┘      └───────────┘      │
       │                        │                      │            │
       │                        ▼                      │            │
       │                 ┌──────────────┐              ▼            │
       │                 │ Domain API   │    ┌───────────────────┐  │
       │                 │ Integration  │    │ Research Quality  ├──┘ 
       │                 └──────────────┘    │ Evaluator         │   
       │                                     └───────────────────┘  
       │             ┌─────────────────┐        
       └────────────▶│ Expert          │
       │             │ Analysis Agent  │
       │             └─────────────────┘
       │             ┌─────────────────┐
       └────────────▶│ Report Writer   │
                     │ Agent           │
                     └─────────────────┘
```

## `1` App Setup

```bash
git clone https://github.com/lastmile-ai/mcp-agent.git
cd mcp-agent/examples/usecases/mcp_realtor_agent
uv init
uv sync
uv pip install -r requirements.txt
npm install -g g-search-mcp
```

## `2` Set up secrets and environment variables

```bash
cp mcp_agent.secrets.yaml.example mcp_agent.secrets.yaml
```

Add your API keys:

```yaml
openai:
  api_key: "YOUR_OPENAI_API_KEY"
```

Optional domain-specific API (RentSpider for real estate example):

```bash
export DOMAIN_API_KEY="your-domain-specific-api-key"
```

## `3` Run locally

```bash
uv run main.py "Austin, TX"
uv run main.py "Denver, CO"
uv run main.py "Miami, FL"
```

## Interactive Experience

The agent will ask domain-relevant questions like:
- **Real Estate**: Property types, budget range, investment goals
- **Finance**: Portfolio size, risk tolerance, investment timeline  
- **Healthcare**: Patient demographics, symptoms, treatment history
- **Legal**: Case type, jurisdiction, legal precedents needed

Reports are saved with expert analysis and actionable recommendations for your specific domain.