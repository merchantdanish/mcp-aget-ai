import argparse
import asyncio
import concurrent.futures
import logging
import traceback

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

logger = logging.getLogger(__name__)


app = MCPApp(name="script_generation_fewshot_eval")


async def run() -> str:
    async with app.run():
        agent = Agent(
            name="agent",
            instruction="You are a helpful assistant.",
            server_names=["word_count"],
        )

        async with agent:
            llm = await agent.attach_llm(OpenAIAugmentedLLM)
            result = await llm.generate_str(
                message="How many words are there in: 'The quick brown fox jumps over the lazy dog'"
            )
            return result


def generate_step():
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(run())
        return result
    except Exception as e:
        logger.exception("Error during script generation", exc_info=e)
        return ""
    finally:
        pending = asyncio.all_tasks(loop=loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()


def main(concurrency: int) -> list[str]:
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(generate_step): idx for idx in range(concurrency)}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                print(f"[Thread {idx}] Result: {result}")
                results.append(result)
            except Exception as e:
                print(f"Step {idx} generated an exception: {e}")
                traceback.print_exc()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--concurrency", type=int, default=2, help="Number of concurrent requests"
    )

    args = parser.parse_args()

    results = main(args.concurrency)

    print("\n\n")
    for idx, result in enumerate(results):
        print(f"Story for step {idx} is {len(result.split())} words long.")
