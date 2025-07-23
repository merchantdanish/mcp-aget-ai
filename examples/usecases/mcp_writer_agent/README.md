# Simple Content AI Agent

A streamlined content creation agent that learns your authentic writing style and helps create engaging content for any platform. This agent uses the evaluator-optimizer pattern with memory persistence to ensure content sounds naturally human while maintaining your unique voice across different platforms.

## Features

This simple agent provides:

1. **Voice Learning with Memory**: Automatically processes writing samples and stores voice patterns in memory server
2. **Smart Content Creation**: Uses evaluator-optimizer workflow to ensure authentic, high-quality output  
3. **PDF Processing**: Integrated MarkItDown support for extracting content from documents
4. **Platform Optimization**: Built-in guidance for Twitter, LinkedIn, Reddit, Medium, and Instagram
5. **Interactive Clarification**: Asks relevant questions only when needed to avoid assumptions
6. **Quality Assurance**: Voice guardian evaluates authenticity before accepting final content

The agent works with any content type and adapts to your natural writing style without complex configuration.

```plaintext
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│ Content Request ├─────▶│ Memory Retrieval ├─────▶│ Content Creator │
└─────────────────┘      └──────────────────┘      └─────────────────┘
         │                         │                        │
         │                         ▼                        ▼
         │                ┌─────────────────┐      ┌─────────────────┐
         │                │ Voice Samples   │      │ Evaluator       │◀───────┐
         │                │ (PDF + Text)    │      │ Optimizer       │        │
         │                └─────────────────┘      └─────────────────┘        │
         │                                                  │                 │
         │                                                  ▼                 │
         │                                         ┌─────────────────┐        │
         │                                         │ Quality Check   │        │
         │                                         │ (Voice Guard)   ├────────┘
         │                                         └─────────────────┘
         │                                                  │
         ▼                                                  ▼
┌─────────────────┐                                ┌─────────────────┐
│ Clarifying      │                                │ Final Content   │
│ Questions       │                                │ Output          │
└─────────────────┘                                └─────────────────┘
```

## Architecture

### How It Works

- **Memory-First**: Instantly stores and recalls your writing style from PDFs and text files—no database setup needed.
- **Evaluator-Optimizer Loop**: Generates content in your voice, checks for authenticity, and refines until it meets quality standards.
- **Smart Processing**: Seamlessly extracts content from documents, detects target platforms, and only asks clarifying questions when necessary.

## Setup

### Install Dependencies
```bash
pip install mcp-agent
npm install -g @modelcontextprotocol/server-memory
pip install markitdown-mcp
pip install 'markitdown[all]'
```

### Configure MCP Servers
Create `mcp_agent.config.yaml`:
```yaml
mcp:
  servers:
    memory:
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-memory"]
      description: "Knowledge graph memory system"
      
    markitdown:
      command: "markitdown-mcp"
      args: []
      description: "Document processing with MarkItDown"
```

### Add API Keys
Create `mcp_agent.secrets.yaml`:
```yaml
openai:
  api_key: "your-openai-api-key"
```

### Add Writing Samples
Create content samples directory and add your writing:
```bash
mkdir content_samples
# Add .md, .txt, or .pdf files with your writing samples
```

## Usage

### Basic Content Creation
```bash
python main.py "help me improve this draft for twitter"
```

### File Management
```bash
python main.py "show my files"
python main.py "list my content"
```

## How It Works

- On first run, the agent scans your `content_samples` folder, learns your writing style, and saves it to memory.
- For each request, it pulls your voice from memory, asks clarifying questions if needed, and creates content using an evaluator-optimizer loop.
- All content is checked for authenticity and clarity before saving to the `posts` folder.

## Platform Guidance

Built-in tips for each platform:

- **Twitter**: Short, engaging, thread-friendly
- **LinkedIn**: Professional, personal insights
- **Reddit**: Casual, story-driven, invites discussion
- **Medium**: Bold opinions, strong hooks, human tone
- **Instagram**: Visual, inspiring, authentic

## Content Samples

Place your existing writing in the content_samples directory:
```
content_samples/
├── my_writing.md
├── email_examples.txt
├── presentation.pdf
└── blog_posts.md
```

The agent will automatically process PDFs using MarkItDown and learn from all content types.

## Output

Generated content is saved in the posts directory:
```
posts/
├── linkedin_20250723_143022.md
├── twitter_20250723_143055.md
└── reddit_20250723_143128.md
```

The agent maintains simplicity while providing powerful content creation capabilities that sound naturally human and authentically yours.