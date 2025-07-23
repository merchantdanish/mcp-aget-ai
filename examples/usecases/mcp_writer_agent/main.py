#!/usr/bin/env python3
"""
Content AI Agent - General Purpose
=========================================
Works with any content. Uses memory server. Asks clarifying questions.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import (
    EvaluatorOptimizerLLM,
    QualityRating,
)

# Simple configuration
POSTS_DIR = Path("posts")
CONTENT_SAMPLES_DIR = Path("content_samples")

# Initialize app
app = MCPApp(name="simple_content_agent")

def get_platform_guidance(platform: str) -> str:
    """Simple platform-specific writing guidance."""
    guidance = {
        "twitter": "Concise, engaging, thread-friendly. Like texting a friend an insight.",
        "linkedin": "Professional but human. Startup founder sharing lessons learned.",
        "reddit": "Casual coffee conversation. Personal stories, discussion questions.",
        "medium": "Substack-style with strong opinions. Personal hook, bold but human.",
        "instagram": "Visual storytelling with personal touch. Inspiring but authentic.",
        "generic": "Clear, engaging, and authentic to your voice."
    }
    return guidance.get(platform, guidance["generic"])

def simple_folder_query() -> str:
    """Simple folder query."""
    CONTENT_SAMPLES_DIR.mkdir(exist_ok=True)
    POSTS_DIR.mkdir(exist_ok=True)
    
    content_files = list(CONTENT_SAMPLES_DIR.rglob("*"))
    content_files = [f for f in content_files if f.is_file()]
    posts_files = list(POSTS_DIR.glob("*.md"))
    
    response = f"📁 **Content Samples**: {len(content_files)} files\n"
    response += f"📝 **Generated Posts**: {len(posts_files)} files\n"
    response += f"🕒 Scanned at: {datetime.now().strftime('%H:%M:%S')}"
    
    return response

async def store_content_samples_in_memory(agent_app):
    """Store all content samples in memory server for future use."""
    try:
        memory_agent = Agent(
            name="memory_manager",
            instruction="Store and retrieve content samples",
            server_names=["memory"]
        )
        
        async with memory_agent:
            # Clear existing voice samples to ensure fresh storage
            try:
                # Try to delete existing voice samples
                await memory_agent.call_tool("delete_entities", {"entityNames": ["voice_samples"]})
                print("🗑️ Cleared existing voice samples")
            except Exception:
                pass  # No existing samples to delete

            # Read all content samples
            samples = []
            
            # Text files
            text_files = list(CONTENT_SAMPLES_DIR.glob("*.md")) + list(CONTENT_SAMPLES_DIR.glob("*.txt"))
            for file_path in text_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            samples.append({"file": file_path.name, "content": content, "type": "text"})
                except (IOError, UnicodeDecodeError) as e:
                    print(f"⚠️ Could not read {file_path.name}: {str(e)}")
                    continue
            
            # PDFs using MarkItDown if available
            pdf_files = list(CONTENT_SAMPLES_DIR.glob("*.pdf"))
            if pdf_files:
                try:
                    doc_processor = Agent(
                        name="doc_processor",
                        instruction="Convert PDFs to text",
                        server_names=["markitdown"]
                    )
                    
                    async with doc_processor:
                        for pdf_path in pdf_files:
                            try:
                                abs_path = pdf_path.resolve()
                                file_uri = f"file://{abs_path}"
                                result = await doc_processor.call_tool("convert_to_markdown", {"uri": file_uri})
                                
                                if result and hasattr(result, 'content') and result.content:
                                    content = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
                                    if content.strip():
                                        samples.append({"file": pdf_path.name, "content": content.strip(), "type": "pdf"})
                                        print(f"✅ Processed PDF: {pdf_path.name}")
                            except Exception as e:
                                print(f"⚠️ Could not process {pdf_path.name}: {str(e)}")
                                
                except Exception as e:
                    print(f"📄 Found {len(pdf_files)} PDFs but MarkItDown not available: {str(e)}")
            
            # Store samples using a simpler approach
            if samples:
                # Create one main entity with all content
                all_content = []
                for sample in samples:
                    content_block = f"=== {sample['file']} ({sample['type'].upper()}) ===\n{sample['content']}"
                    all_content.append(content_block)
                
                combined_content = "\n\n".join(all_content)
                
                # Store as single entity
                await memory_agent.call_tool("create_entities", {
                    "entities": [{
                        "name": "content_samples",
                        "entityType": "voice_and_content",
                        "observations": [combined_content[:8000]]  # Limit to avoid size issues
                    }]
                })
                
                print(f"💾 Stored {len(samples)} content samples in memory ({len(combined_content)} chars)")
                print(f"📋 Files stored: {[s['file'] for s in samples]}")
                return True
            else:
                print("ℹ️ No content samples found to store")
                return False
                
    except Exception as e:
        print(f"⚠️ Memory storage failed: {str(e)}")
        return False

async def get_content_context_from_memory(agent_app, request: str):
    """Get relevant content context from memory based on the request."""
    try:
        memory_agent = Agent(
            name="memory_manager", 
            instruction="Retrieve relevant content context",
            server_names=["memory"]
        )
        
        async with memory_agent:
            # Get the content samples directly
            result = await memory_agent.call_tool("open_nodes", {"names": ["content_samples"]})
            
            if result and hasattr(result, 'content') and result.content:
                content = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
                
                # Parse the JSON response to extract the actual content
                import json  
                try:    
                    data = json.loads(content)
                    if "entities" in data and len(data["entities"]) > 0:
                        entity = data["entities"][0]
                        if "observations" in entity and len(entity["observations"]) > 0:
                           return entity["observations"][0]
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    print(f"⚠️ Failed to parse memory response: {str(e)}")
                return content
                
    except Exception as e:
        print(f"⚠️ Memory retrieval failed: {str(e)}")
        
    return ""

async def ask_clarifying_questions(request: str, platform: str) -> dict:
    """Ask clarifying questions if the request is ambiguous."""
    
    # Simple heuristics for when to ask questions
    is_creation_request = any(word in request.lower() for word in ['write', 'create', 'draft', 'post about'])
    is_vague = len(request.split()) < 4
    
    if not is_creation_request and not is_vague:
        return {"needs_clarification": False}
    
    clarification = {"needs_clarification": True}
    
    # Ask about intent if unclear
    if is_vague:
        print(f"\n🤔 Your request is quite brief: '{request}'")
        intent = input("What would you like me to help with?\n1. Create new content\n2. Improve existing content\n3. Something else\nChoice (1-3): ").strip()
        
        if intent == "1":
            clarification["intent"] = "create"
        elif intent == "2": 
            clarification["intent"] = "improve"
            clarification["existing_content"] = input("What content should I improve? ")
        else:
            clarification["intent"] = "other"
            clarification["details"] = input("Please describe what you'd like: ")
    else:
        clarification["intent"] = "create"
    
    # Ask about topic if creating content
    if clarification.get("intent") == "create" and is_vague:
        clarification["topic"] = input("What topic should I write about? ")
    
    # Ask about style/tone
    style_input = input(f"Any specific style/tone for {platform}? (press enter to use your natural voice): ").strip()
    if style_input:
        clarification["style_notes"] = style_input
    
    return clarification

async def create_content(request: str, platform: str, context: str, clarification: dict = None) -> str:
    """Create content using the evaluator-optimizer pattern."""
    
    async with app.run() as agent_app:
        logger = agent_app.logger
        
        # Build the prompt based on clarification
        if clarification and clarification.get("intent") == "improve":
            base_request = f"Improve this content: {clarification.get('existing_content', request)}"
        elif clarification and clarification.get("topic"):
            base_request = f"Write a {platform} post about: {clarification['topic']}"
        else:
            base_request = request
            
        # Create optimizer instructions
        optimizer_instruction = f"""You create content in the author's authentic voice using provided context.

Platform: {platform} - {get_platform_guidance(platform)}

Guidelines for Content Creation:
- Emulate the author's unique writing style and voice as reflected in the provided context and samples.
- Prioritize authenticity—ensure the content feels genuinely human, not AI-generated or formulaic.
- Thoughtfully incorporate relevant details, anecdotes, or insights from the context to enrich the content.
- Strive for a tone that is both engaging and approachable, while maintaining a natural flow.
- Adapt the content to align with {platform} best practices, including formatting, length, and audience expectations.
- Avoid clichés, generic phrasing, or overused expressions; aim for originality and personality.
- Ensure clarity, coherence, and a logical structure throughout the piece.
- If appropriate, include a compelling hook or introduction to capture attention.
- Maintain consistency in style, tone, and perspective from start to finish.

{f"Author's context and voice samples: {context[:1000]}" if context else ""}
{f"Style notes: {clarification.get('style_notes')}" if clarification and clarification.get('style_notes') else ""}"""

        evaluator_instruction = """Rate content for authenticity and quality.

Evaluation Criteria:
- EXCELLENT: Feels genuinely human—natural, authentic, engaging, and exceptionally well-written. Captures the author's unique voice and style throughout.
- GOOD: Strong overall, with minor areas for improvement. Mostly authentic and in the author's voice, but could be slightly more engaging or polished.
- FAIR: Noticeable issues with voice, tone, or quality. Parts may sound generic, formulaic, or less like the author.
- POOR: Clearly AI-generated, robotic, or lacking authenticity. Major problems with voice, coherence, or engagement. Has em dashes. 

Your Task:
Carefully read the content and rate it using the scale above. Focus on whether the writing truly sounds like the author—does it feel personal, original, and human, or does it come across as artificial or generic? Provide a brief explanation for your rating, highlighting specific strengths or areas for improvement."""

        # Create agents
        optimizer = Agent(
            name="content_creator",
            instruction=optimizer_instruction,
            server_names=[]
        )
        
        evaluator = Agent(
            name="quality_checker",
            instruction=evaluator_instruction, 
            server_names=[]
        )
        
        # Use evaluator-optimizer
        content_creator = EvaluatorOptimizerLLM(
            optimizer=optimizer,
            evaluator=evaluator,
            llm_factory=OpenAIAugmentedLLM,
            min_rating=QualityRating.GOOD
        )
        
        logger.info(f"Creating {platform} content")
        result = await content_creator.generate_str(
            message=base_request,
            request_params=RequestParams(model="gpt-4o")
        )
        
        return result

def save_result(content: str, platform: str) -> str:
    """Save content to posts folder."""
    POSTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{platform}_{timestamp}.md"
    output_path = POSTS_DIR / filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return str(output_path)

async def main():
    """Main function."""
    print("🎯 Simple Content AI Agent")
    print("💾 Uses memory server for voice samples")

    # Create directories
    CONTENT_SAMPLES_DIR.mkdir(exist_ok=True)
    POSTS_DIR.mkdir(exist_ok=True)
    
    # Get user input
    if len(sys.argv) > 1:
        request = " ".join(sys.argv[1:])
    else:
        request = input("What would you like me to help with? ")
    
    if not request.strip():
        print("❌ No input provided")
        return False
    
    # Handle folder queries
    if any(query in request.lower() for query in ["show files", "list content", "what files"]):
        print("📁 Checking folder contents...")
        result = simple_folder_query()
        print(f"\n{result}")
        return True
    
    # Detect platform
    platform = "generic"
    for p in ["twitter", "linkedin", "reddit", "medium", "instagram"]:
        if p in request.lower():
            platform = p
            break
    
    print(f"📝 Platform: {platform}")
    print(f"💭 Request: {request}")
    
    
    try:
        async with app.run() as agent_app:
            # Store content samples in memory
            print("💾 Checking content samples...")
            await store_content_samples_in_memory(agent_app)
            
            # Get context from memory
            print("🧠 Retrieving context from memory...")
            context = await get_content_context_from_memory(agent_app, request)
            
            if context:
                print(f"✅ Found relevant context ({len(context)} chars)")
            else:
                print("ℹ️ No previous context found")
        
        # Ask clarifying questions if needed
        clarification = await ask_clarifying_questions(request, platform)
        
        if clarification.get("needs_clarification"):
            print("✅ Got clarification, creating content...")
        
        # Create content
        improved_content = await create_content(request, platform, context, clarification)
        
        # Save result
        output_path = save_result(improved_content, platform)
        
        
        print("\n✅ Content created!")
        print(f"📁 Saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🎯 Simple Content AI Agent - General Purpose!")
    print("💾 Memory server integration")
    
    success = asyncio.run(main())
    exit(0 if success else 1)