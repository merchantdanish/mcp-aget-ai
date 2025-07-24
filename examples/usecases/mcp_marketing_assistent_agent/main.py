#!/usr/bin/env python3
"""
Marketing Content AI Agent Framework
====================================
Company-agnostic, sophisticated content creation using external prompts.
Supports any company via configuration and prompt customization.
"""

import asyncio
import sys
import json
import yaml
import re
from datetime import datetime
from pathlib import Path
import traceback

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import (
    EvaluatorOptimizerLLM,
    QualityRating,
)

# Configuration
COMPANY_CONFIG_FILE = "company_config.yaml"
PROMPTS_DIR = "prompts"
COMPANY_DOCS_DIR = "company_docs"
CONTENT_SAMPLES_DIR = "content_samples"
POSTS_DIR = "posts"

# Initialize app
app = MCPApp(name="marketing_content_agent")


class PromptManager:
    """Manages external prompt templates with variable substitution"""
    
    def __init__(self, prompts_dir: str = PROMPTS_DIR):
        self.prompts_dir = Path(prompts_dir)
        self.loaded_prompts = {}
    
    def load_prompt(self, prompt_name: str, **variables) -> str:
        """Load and format prompt template with variables"""
        if prompt_name not in self.loaded_prompts:
            prompt_file = self.prompts_dir / f"{prompt_name}.md"
            if not prompt_file.exists():
                raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
            
            self.loaded_prompts[prompt_name] = prompt_file.read_text(encoding='utf-8')
        
        # Format with variables
        template = self.loaded_prompts[prompt_name]
        try:
            return template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing variable {e} for prompt {prompt_name}")
    
    def get_available_prompts(self) -> list:
        """List available prompt templates"""
        return [f.stem for f in self.prompts_dir.glob("*.md")]


class DocumentProcessor:
    """Handles document processing via markitdown MCP server"""
    
    async def process_document(self, file_path: str) -> str:
        """Process any document type using markitdown"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() in ['.md', '.txt']:
            try:
                return file_path.read_text(encoding='utf-8')
            except Exception as e:
                return f"[Error reading {file_path.name}: {e}]"
        
        # Use markitdown for other file types
        return await self._process_with_markitdown(file_path)
    
    async def _process_with_markitdown(self, file_path: Path) -> str:
        """Process document using markitdown MCP server"""
        try:
            doc_agent = Agent(
                name="doc_processor",
                instruction="Convert documents to markdown",
                server_names=["markitdown"]
            )
            
            async with doc_agent:
                absolute_path = file_path.resolve()
                file_uri = f"file:///{absolute_path}"
                
                result = await doc_agent.call_tool("convert_to_markdown", {"uri": file_uri})
                
                if result and hasattr(result, 'content') and result.content:
                    content = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
                    if content.strip() and not content.startswith("Error"):
                        return content.strip()
                
                return f"[Could not process {file_path.name}]"
                
        except Exception as e:
            return f"[Markitdown processing failed for {file_path.name}: {e}]"


class URLContentFetcher:
    """Fetches content from URLs using fetch MCP server"""
    
    def __init__(self):
        self.fetch_agent = None
        self.available_tools = []
    
    async def initialize(self):
        """Initialize fetch agent"""
        try:
            self.fetch_agent = Agent(
                name="url_fetcher",
                instruction="Fetch web content",
                server_names=["fetch"]
            )
            await self.fetch_agent.__aenter__()
            self.available_tools = list(self.fetch_agent.get_available_tools().keys())
        except Exception as e:
            print(f"⚠️ Fetch server not available: {e}")
    
    async def fetch_urls_from_text(self, text: str) -> str:
        """Extract URLs from text and fetch their content"""
        urls = self._extract_urls(text)
        if not urls:
            return ""
        
        fetched_content = []
        for url in urls:
            content = await self._fetch_single_url(url)
            if content:
                # Limit content length
                max_chars = 3000
                truncated = content[:max_chars]
                if len(content) > max_chars:
                    truncated += f"\n\n[Content truncated - original was {len(content)} chars]"
                
                fetched_content.append(f"=== Content from {url} ===\n{truncated}")
        
        return "\n\n".join(fetched_content)
    
    def _extract_urls(self, text: str) -> list:
        """Extract URLs from text"""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        return re.findall(url_pattern, text)
    
    async def _fetch_single_url(self, url: str) -> str:
        """Fetch content from a single URL"""
        if not self.fetch_agent:
            return ""
        
        # Try common tool names
        tool_names = ["fetch", "get_url", "read_url", "fetch_url"] + self.available_tools
        
        for tool_name in tool_names:
            try:
                result = await self.fetch_agent.call_tool(tool_name, {"url": url})
                if result and hasattr(result, 'content') and result.content:
                    content = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
                    return content.strip()
            except Exception:
                continue
        
        return ""
    
    async def cleanup(self):
        """Cleanup fetch agent"""
        if self.fetch_agent:
            try:
                await self.fetch_agent.__aexit__(None, None, None)
            except Exception:
                pass


class ContextAssembler:
    """Assembles context from various sources"""
    
    def __init__(self, doc_processor: DocumentProcessor, url_fetcher: URLContentFetcher):
        self.doc_processor = doc_processor
        self.url_fetcher = url_fetcher
    
    async def gather_comprehensive_context(self, request: str, platform: str, company_config: dict) -> dict:
        """Gather all relevant context"""
        # Load company documentation
        company_context = await self._load_directory_content(COMPANY_DOCS_DIR)
        
        # Load content samples (platform-specific if available)
        content_samples = await self._load_content_samples(platform)
        
        # Fetch URL content if any URLs in request
        url_content = await self.url_fetcher.fetch_urls_from_text(request)
        
        # Organize context intelligently
        organized_context = await self._organize_context(
            company_context, content_samples, url_content, request, platform, company_config
        )
        
        return {
            'company_docs': company_context,
            'content_samples': content_samples,
            'url_content': url_content,
            'organized': organized_context
        }
    
    async def _load_directory_content(self, directory: str) -> str:
        """Load all files from a directory"""
        dir_path = Path(directory)
        if not dir_path.exists():
            return ""
        
        content_blocks = []
        for file_path in dir_path.iterdir():
            if file_path.is_file():
                processed_content = await self.doc_processor.process_document(file_path)
                if processed_content and not processed_content.startswith("[Error"):
                    content_blocks.append(f"=== {file_path.name} ===\n{processed_content}")
        
        return "\n\n".join(content_blocks)
    
    async def _load_content_samples(self, platform: str) -> str:
        """Load content samples, prioritizing platform-specific ones"""
        samples_dir = Path(CONTENT_SAMPLES_DIR)
        if not samples_dir.exists():
            return ""
        
        content_blocks = []
        
        # Look for platform-specific samples first
        platform_files = list(samples_dir.glob(f"*{platform}*"))
        general_files = [f for f in samples_dir.iterdir() if f.is_file() and f not in platform_files]
        
        # Process platform-specific files first
        for file_path in platform_files + general_files:
            processed_content = await self.doc_processor.process_document(file_path)
            if processed_content and not processed_content.startswith("[Error"):
                content_blocks.append(f"=== {file_path.name} ===\n{processed_content}")
        
        return "\n\n".join(content_blocks)
    
    async def _organize_context(self, company_docs: str, content_samples: str, url_content: str, 
                              request: str, platform: str, company_config: dict) -> str:
        """Organize context by relevance using LLM"""
        try:
            prompt_manager = PromptManager()
            
            context_vars = {
                'request': request,
                'platform': platform,
                'company_name': company_config['company']['name'],
                'company_docs_length': len(company_docs),
                'content_samples_length': len(content_samples),
                'url_content_length': len(url_content),
                'total_context': company_docs + content_samples + url_content
            }
            
            organizer_prompt = prompt_manager.load_prompt("context_organizer_prompt", **context_vars)
            
            organizer_llm = OpenAIAugmentedLLM()
            organized = await organizer_llm.generate_str(
                message=organizer_prompt,
                request_params=RequestParams(model="gpt-4o")
            )
            
            return organized
            
        except Exception:
            # Fallback: simple concatenation
            parts = []
            if content_samples:
                parts.append(f"CONTENT SAMPLES:\n{content_samples}")
            if company_docs:
                parts.append(f"COMPANY DOCS:\n{company_docs}")
            if url_content:
                parts.append(f"URL CONTENT:\n{url_content}")
            
            return "\n\n".join(parts)


class MarketingContentAgent:
    """Main marketing content agent - company agnostic"""
    
    def __init__(self, config_file: str = COMPANY_CONFIG_FILE):
        self.config = self._load_config(config_file)
        self.prompt_manager = PromptManager()
        self.doc_processor = DocumentProcessor()
        self.url_fetcher = URLContentFetcher()
        self.context_assembler = ContextAssembler(self.doc_processor, self.url_fetcher)
    
    def _load_config(self, config_file: str) -> dict:
        """Load company configuration"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"⚠️ {config_file} not found. Creating default config...")
            self._create_default_config(config_file)
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️ Error loading config: {e}")
            return self._get_minimal_config()
    
    def _create_default_config(self, config_file: str):
        """Create default configuration file"""
        default_config = {
            'company': {
                'name': 'Your Company',
                'industry': 'Technology',
                'target_audience': ['Professionals', 'Decision makers']
            },
            'brand': {
                'voice': {
                    'personality': 'Professional yet approachable',
                    'tone_keywords': ['clear', 'helpful', 'authentic'],
                    'avoid': ['buzzwords', 'jargon', 'overly promotional']
                },
                'messaging_pillars': [
                    'Quality solutions',
                    'Customer focused',
                    'Innovation driven'
                ]
            },
            'platforms': {
                'linkedin': {
                    'max_word_count': 150,
                    'tone': 'Professional',
                    'guidelines': 'Business-focused, value-driven content'
                },
                'twitter': {
                    'max_word_count': 50,
                    'tone': 'Casual and direct',
                    'guidelines': 'Short, engaging, conversational'
                }
            },
            'quality_standards': {
                'excellence_criteria': [
                    'Authentic voice',
                    'Specific details',
                    'Clear value proposition'
                ],
                'poor_criteria': [
                    'Generic language',
                    'Overly promotional',
                    'Vague statements'
                ],
                'banned_phrases': [
                    'game-changer',
                    'revolutionary',
                    'cutting-edge'
                ]
            },
            'prompt_variables': {
                'instructions': 'Create authentic, engaging content that reflects our brand voice.',
                'good_examples': [
                    'Clear, specific communication',
                    'Helpful, actionable insights'
                ],
                'bad_examples': [
                    'Vague promotional language',
                    'Generic industry buzzwords'
                ]
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    def _get_minimal_config(self) -> dict:
        """Get minimal config as fallback"""
        return {
            'company': {'name': 'Company', 'industry': 'Business', 'target_audience': ['Customers']},
            'brand': {'voice': {'personality': 'Professional'}, 'messaging_pillars': []},
            'platforms': {'linkedin': {'max_word_count': 150}},
            'quality_standards': {'excellence_criteria': [], 'poor_criteria': [], 'banned_phrases': []},
            'prompt_variables': {'instructions': 'Create good content.', 'good_examples': [], 'bad_examples': []}
        }
    
    async def initialize(self):
        """Initialize all components"""
        await self.url_fetcher.initialize()
        self._ensure_directories_exist()
        print(f"🚀 Marketing Content Agent ready for {self.config['company']['name']}")
    
    def _ensure_directories_exist(self):
        """Ensure required directories exist"""
        for directory in [COMPANY_DOCS_DIR, CONTENT_SAMPLES_DIR, POSTS_DIR, PROMPTS_DIR]:
            Path(directory).mkdir(exist_ok=True)
    
    async def create_content(self, request: str, platform: str = "linkedin") -> dict:
        """Main content creation pipeline"""
        try:
            print(f"📝 Creating {platform} content for: {request}")
            
            # Gather comprehensive context
            print("🧠 Gathering context...")
            context = await self.context_assembler.gather_comprehensive_context(request, platform, self.config)
            
            # Get clarification if needed
            clarification = await self._get_clarification(request, platform, context)
            
            # Build enhanced request
            enhanced_request = self._build_enhanced_request(request, clarification)
            
            # Create content with evaluator-optimizer
            print("🎨 Creating optimized content...")
            content = await self._create_optimized_content(enhanced_request, platform, context)
            
            # Save content
            output_path = await self._save_content(content, platform)
            
            return {
                'content': content,
                'output_path': output_path,
                'platform': platform,
                'context_used': len(context['organized']),
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Error creating content: {e}")
            return {
                'success': False,
                'error': str(e),
                'details': traceback.format_exc()
            }
    
    async def _get_clarification(self, request: str, platform: str, context: dict) -> dict:
        """Get clarification if needed using external prompt"""
        try:
            prompt_vars = self._build_prompt_variables(request, platform, context['organized'])
            
            analyzer_prompt = self.prompt_manager.load_prompt("request_analyzer_prompt", **prompt_vars)
            
            analyzer_llm = OpenAIAugmentedLLM()
            analysis = await analyzer_llm.generate_str(
                message=analyzer_prompt,
                request_params=RequestParams(model="gpt-4o")
            )
            
            # Try to parse JSON response
            try:
                analysis_data = json.loads(analysis.strip().replace('```json', '').replace('```', ''))
                if analysis_data.get("needs_clarification", False):
                    return self._interactive_clarification(analysis_data)
            except json.JSONDecodeError:
                pass
            
            return {"needs_clarification": False}
            
        except Exception:
            # Simple fallback clarification
            if len(request.split()) < 5:
                goal = input("What's the main goal? (awareness/engagement/conversion): ").strip()
                if goal:
                    return {"needs_clarification": True, "campaign_goal": goal}
            
            return {"needs_clarification": False}
    
    def _interactive_clarification(self, analysis_data: dict) -> dict:
        """Handle interactive clarification"""
        print(f"\n🎯 {analysis_data.get('reason', 'Need more context')}")
        
        clarification = {"needs_clarification": True}
        
        for q in analysis_data.get("questions", []):
            answer = input(f"{q['question']} ").strip()
            if answer:
                clarification[q["key"]] = answer
        
        return clarification
    
    def _build_enhanced_request(self, original_request: str, clarification: dict) -> str:
        """Build enhanced request with clarification"""
        if not clarification.get("needs_clarification"):
            return original_request
        
        enhanced_parts = [f"Original request: {original_request}"]
        
        for key, value in clarification.items():
            if key != "needs_clarification" and value:
                enhanced_parts.append(f"{key.replace('_', ' ').title()}: {value}")
        
        return " | ".join(enhanced_parts)
    
    async def _create_optimized_content(self, request: str, platform: str, context: dict) -> str:
        """Create content using evaluator-optimizer with external prompts"""
        async with app.run():
            # Build variables for prompts
            prompt_vars = self._build_prompt_variables(request, platform, context['organized'])
            
            # Load prompts
            optimizer_prompt = self.prompt_manager.load_prompt("content_creator_prompt", **prompt_vars)
            evaluator_prompt = self.prompt_manager.load_prompt("quality_evaluator_prompt", **prompt_vars)
            
            # Create agents
            optimizer = Agent(
                name="content_creator",
                instruction=optimizer_prompt,
                server_names=[]
            )
            
            evaluator = Agent(
                name="quality_evaluator",
                instruction=evaluator_prompt,
                server_names=[]
            )
            
            # Use evaluator-optimizer
            content_creator = EvaluatorOptimizerLLM(
                optimizer=optimizer,
                evaluator=evaluator,
                llm_factory=OpenAIAugmentedLLM,
                min_rating=QualityRating.EXCELLENT
            )
            
            result = await content_creator.generate_str(
                message=request,
                request_params=RequestParams(model="gpt-4o")
            )
            
            return result
    
    def _build_prompt_variables(self, request: str, platform: str, context: str) -> dict:
        """Build variables for prompt templates"""
        company_info = self.config['company']
        brand_info = self.config['brand']
        platform_info = self.config['platforms'].get(platform, {})
        quality_info = self.config['quality_standards']
        custom_vars = self.config.get('prompt_variables', {})
        
        return {
            # Company info
            'company_name': company_info['name'],
            'industry': company_info['industry'],
            'target_audience': ', '.join(company_info['target_audience']),
            
            # Brand info
            'brand_personality': brand_info['voice']['personality'],
            'tone_keywords': ', '.join(brand_info['voice'].get('tone_keywords', [])),
            'avoid_items': ', '.join(brand_info['voice'].get('avoid', [])),
            'messaging_pillars': ', '.join(brand_info['messaging_pillars']),
            
            # Platform info
            'platform': platform.upper(),
            'max_word_count': platform_info.get('max_word_count', 200),
            'platform_tone': platform_info.get('tone', 'Professional'),
            'platform_guidelines': platform_info.get('guidelines', ''),
            
            # Quality standards
            'banned_phrases': ', '.join([f'"{p}"' for p in quality_info['banned_phrases']]),
            'excellence_criteria': '\n'.join([f"- {c}" for c in quality_info['excellence_criteria']]),
            'poor_criteria': '\n'.join([f"- {c}" for c in quality_info['poor_criteria']]),
            
            # Content
            'context': context,
            'request': request,
            
            # Custom variables
            'instructions': custom_vars.get('instructions', 'Create great content'),
            'good_examples': '\n'.join([f"- {ex}" for ex in custom_vars.get('good_examples', [])]),
            'bad_examples': '\n'.join([f"- {ex}" for ex in custom_vars.get('bad_examples', [])]),
            
            # Additional useful variables
            'current_date': datetime.now().strftime('%Y-%m-%d'),
            'context_length': len(context)
        }
    
    async def _save_content(self, content: str, platform: str) -> str:
        """Save content with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{platform}_{timestamp}.md"
        output_path = Path(POSTS_DIR) / filename
        
        # Add metadata
        content_with_metadata = f"""---
platform: {platform}
company: {self.config['company']['name']}
created: {datetime.now().isoformat()}
---

{content}"""
        
        try:
            output_path.write_text(content_with_metadata, encoding='utf-8')
            return str(output_path)
        except Exception as e:
            print(f"⚠️ Could not save to {output_path}: {e}")
            return f"Error saving to {output_path}"
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.url_fetcher.cleanup()


def detect_platform(request: str) -> str:
    """Detect platform from request"""
    request_lower = request.lower()
    platforms = ["twitter", "linkedin", "instagram", "facebook", "reddit", "medium", "newsletter", "email"]
    
    for platform in platforms:
        if platform in request_lower:
            return platform
    
    return "linkedin"  # Default


async def main():
    """Main function"""
    print("🎯 Marketing Content AI Agent Framework")
    print("🏢 Company-agnostic content creation")
    print("📝 Uses external prompts and configuration")
    
    # Get input
    if len(sys.argv) > 1:
        request = " ".join(sys.argv[1:])
    else:
        print("\n💡 TIP: Include URLs in your request for auto-fetch!")
        print("   Example: 'write a linkedin post about https://example.com/article'")
        request = input("\nWhat marketing content would you like me to create? ").strip()
    
    if not request:
        print("❌ No input provided")
        return False
    
    # Detect platform
    platform = detect_platform(request)
    
    # Create agent and initialize
    agent = MarketingContentAgent()
    await agent.initialize()
    
    try:
        # Create content
        result = await agent.create_content(request, platform)
        
        if result['success']:
            print("\n✅ Content created successfully!")
            print(f"📁 Saved to: {result['output_path']}")
            print(f"📊 Context used: {result['context_used']} characters")
            print("\n📄 Content Preview:")
            print("-" * 50)
            preview = result['content'][:300] + "..." if len(result['content']) > 300 else result['content']
            print(preview)
            print("-" * 50)
        else:
            print(f"❌ Failed to create content: {result['error']}")
            return False
        
    finally:
        await agent.cleanup()
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)