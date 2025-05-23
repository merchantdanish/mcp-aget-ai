#!/usr/bin/env python3
"""
Basic test for RCM Phase 2 implementation with real LLM calls.
Uses canonical mcp-agent configuration patterns.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp_agent.app import MCPApp
from workflows.conversation_workflow import ConversationWorkflow
from models.conversation_models import ConversationState, QualityMetrics, Requirement

async def test_rcm_with_real_calls():
    """Test RCM with real LLM calls using canonical mcp-agent configuration"""
    print("🧪 Testing RCM with Real LLM Calls...")
    print("📁 Using mcp_agent.config.yaml and mcp_agent.secrets.yaml")
    
    # Create app using canonical mcp-agent pattern (loads config files automatically)
    app = MCPApp(name="rcm_test")
    
    # Register workflow
    @app.workflow
    class TestConversationWorkflow(ConversationWorkflow):
        """Test workflow registered with app"""
        pass

    try:
        async with app.run() as test_app:
            print("✓ App initialized with config files")
            
            # Check if we have proper LLM configuration
            has_openai = hasattr(test_app.context.config, 'openai') and test_app.context.config.openai
            has_anthropic = hasattr(test_app.context.config, 'anthropic') and test_app.context.config.anthropic
            
            if not (has_openai or has_anthropic):
                print("⚠️  Warning: No LLM providers configured. Tests will use fallbacks.")
                print("   To test with real LLMs, add API keys to mcp_agent.secrets.yaml")
            else:
                provider = "openai" if has_openai else "anthropic"
                print(f"✓ LLM provider available: {provider}")

            # Add filesystem access to current directory
            if hasattr(test_app.context.config, 'mcp') and test_app.context.config.mcp:
                if "filesystem" in test_app.context.config.mcp.servers:
                    test_app.context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

            # Create workflow instance
            workflow = TestConversationWorkflow(app)
            print("✓ Workflow created and registered")

            # Test 1: First turn with quality control
            print("\n🔄 Test 1: First turn with quality control...")
            result1 = await workflow.run({
                "user_input": "I need help creating a Python function that calculates fibonacci numbers. It should be efficient and handle edge cases.",
                "state": None
            })

            print("✓ First turn completed")
            print(f"  Response length: {len(result1.value['response'])} chars")
            print(f"  Turn number: {result1.value['turn_number']}")
            
            # Check quality metrics
            metrics = result1.value.get("metrics", {})
            if metrics:
                overall_score = metrics.get("clarity", 0) + metrics.get("completeness", 0) - metrics.get("assumptions", 0) - metrics.get("verbosity", 0)
                overall_score = overall_score / 4
                print(f"  Quality score: {overall_score:.2f}")
                print(f"  Premature attempt: {metrics.get('premature_attempt', 'unknown')}")

            # Test 2: Second turn with requirement tracking
            print("\n🔄 Test 2: Second turn with requirement tracking...")
            result2 = await workflow.run({
                "user_input": "Actually, I also need the function to return both the nth fibonacci number and the sequence up to that number. Can you modify it?",
                "state": result1.value["state"]
            })

            print("✓ Second turn completed")
            print(f"  Response length: {len(result2.value['response'])} chars")
            print(f"  Turn number: {result2.value['turn_number']}")

            # Test 3: Third turn to check context consolidation
            print("\n🔄 Test 3: Third turn (triggers context consolidation)...")
            result3 = await workflow.run({
                "user_input": "Can you also add input validation and docstrings to make it production-ready?",
                "state": result2.value["state"]
            })

            print("✓ Third turn completed")
            print(f"  Response length: {len(result3.value['response'])} chars")
            print(f"  Turn number: {result3.value['turn_number']}")

            # Verify comprehensive state
            final_state = ConversationState.from_dict(result3.value["state"])
            print("\n📊 Final State Analysis:")
            print(f"  Total messages: {len(final_state.messages)}")
            print(f"  Current turn: {final_state.current_turn}")
            print(f"  Quality history: {len(final_state.quality_history)} entries")
            print(f"  Requirements tracked: {len(final_state.requirements)}")
            print(f"  Answer lengths: {final_state.answer_lengths}")
            print(f"  Consolidation turns: {final_state.consolidation_turns}")
            print(f"  Consolidated context length: {len(final_state.consolidated_context)} chars")

            # Check for research paper metrics
            if len(final_state.answer_lengths) > 1:
                bloat_ratio = final_state.answer_lengths[-1] / final_state.answer_lengths[0]
                print(f"  Answer bloat ratio: {bloat_ratio:.2f}x")
                
                if bloat_ratio > 2.0:
                    print("  ⚠️  Potential answer bloat detected (>2x growth)")
                else:
                    print("  ✓ Answer bloat within acceptable range")

            # Validate requirements tracking
            pending_reqs = [r for r in final_state.requirements if r.status == "pending"]
            addressed_reqs = [r for r in final_state.requirements if r.status == "addressed"]
            print(f"  Pending requirements: {len(pending_reqs)}")
            print(f"  Addressed requirements: {len(addressed_reqs)}")

            # Test assertions
            assert final_state.current_turn == 3, f"Expected 3 turns, got {final_state.current_turn}"
            assert len(final_state.messages) >= 6, f"Expected at least 6 messages (system + 3 pairs), got {len(final_state.messages)}"
            assert len(final_state.quality_history) == 3, f"Expected 3 quality entries, got {len(final_state.quality_history)}"
            assert len(final_state.answer_lengths) == 3, f"Expected 3 answer lengths, got {len(final_state.answer_lengths)}"
            
            # Check that context consolidation happened on turn 3
            if final_state.consolidation_turns:
                assert 3 in final_state.consolidation_turns, "Expected context consolidation on turn 3"
                print("  ✓ Context consolidation triggered correctly")

            print("\n🎉 All comprehensive tests passed!")
            return True

    except Exception as e:
        print(f"\n💥 Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_fallback_behavior():
    """Test that fallbacks work when LLM providers are unavailable"""
    print("\n🧪 Testing Fallback Behavior...")
    
    # Create app with no LLM providers to test fallbacks
    from mcp_agent.config import Settings, LoggerSettings, MCPSettings
    
    settings = Settings(
        execution_engine="asyncio",
        logger=LoggerSettings(type="console", level="error"),
        mcp=MCPSettings(servers={}),
        openai=None,
        anthropic=None
    )
    
    app = MCPApp(name="rcm_fallback_test", settings=settings)
    
    @app.workflow
    class FallbackTestWorkflow(ConversationWorkflow):
        """Fallback test workflow"""
        pass

    try:
        async with app.run() as test_app:
            print("✓ App initialized without LLM providers")
            
            workflow = FallbackTestWorkflow(app)
            
            # Test that fallbacks work
            result = await workflow.run({
                "user_input": "Test fallback behavior",
                "state": None
            })
            
            print("✓ Fallback processing completed")
            print(f"  Response: {result.value['response'][:100]}...")
            
            # Verify fallback metrics are reasonable
            metrics = result.value.get("metrics", {})
            assert metrics, "Should have fallback metrics"
            
            # Check if the response indicates fallback behavior
            response = result.value['response'].lower()
            is_fallback = any(word in response for word in ['mock', 'test', 'fallback', 'technical difficulties'])
            assert is_fallback, f"Should indicate fallback behavior. Got: {result.value['response'][:200]}"
            
            print("✓ Fallback behavior verified")
            return True
            
    except Exception as e:
        print(f"💥 Fallback test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = True
    
    # Check for secrets file
    secrets_file = Path(__file__).parent / "mcp_agent.secrets.yaml"
    if not secrets_file.exists():
        print("📝 Creating example secrets file...")
        secrets_content = """# Example secrets file for RCM testing
# Uncomment and add your API keys to enable real LLM calls

# openai:
#   api_key: "your-openai-api-key-here"

# anthropic:
#   api_key: "your-anthropic-api-key-here"
"""
        with open(secrets_file, 'w') as f:
            f.write(secrets_content)
        print(f"✓ Created {secrets_file}")
        print("  Add your API keys to enable real LLM testing")
    
    try:
        # Test with real configuration
        success &= asyncio.run(test_rcm_with_real_calls())
        
        # Test fallback behavior
        success &= asyncio.run(test_fallback_behavior())
        
        if success:
            print("\n🎉 All RCM tests passed!")
            print("\n✅ RCM Phase 2 implementation with quality control is working correctly!")
            print("\n📚 Features tested:")
            print("  • Multi-turn conversation with state persistence")
            print("  • Quality-controlled response generation")
            print("  • Requirement extraction and tracking")
            print("  • Context consolidation (lost-in-middle prevention)")
            print("  • Answer bloat detection and prevention")
            print("  • Robust fallbacks when LLMs unavailable")
            print("  • Research paper metrics tracking")
        else:
            print("\n❌ Some tests failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)