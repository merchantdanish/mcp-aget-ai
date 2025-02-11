import asyncio
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams


from prompts import SCHEDULING_AGENT_INSTRUCTIONS
from servers.gmail import GmailService

CRED_FILE_PATH = "/Users/jerronlim/Desktop/projects/mcp-agent-jerron/flagship/scheduling-assistant/creds/client_creds.json"
TOKEN_PATH = "/Users/jerronlim/Desktop/projects/mcp-agent-jerron/flagship/scheduling-assistant/creds/gmail_app_tokens.json"

app = MCPApp(name="mcp-scheduling-assistant")


def generate_agent_prompt(thread_messages: list):
    messages_str = str(thread_messages)
    return f"""
    Below is the full conversation thread of an email:
    {messages_str}
    
    Please perform the necessary actions based on your instructions.
    """


async def main():
    async with app.run() as mcp_agent_app:
        logger = mcp_agent_app.logger

        gmail_service = GmailService(CRED_FILE_PATH, TOKEN_PATH)

        scheduling_agent = Agent(
            name="scheduler",
            instruction=SCHEDULING_AGENT_INSTRUCTIONS,
            server_names=["calendar", "gmail"],
        )

        async with scheduling_agent:
            llm = await scheduling_agent.attach_llm(OpenAIAugmentedLLM)

            while True:
                logger.info("Checking for unread emails.")
                emails = await gmail_service.get_unread_emails()

                if isinstance(emails, list) and len(emails) > 0:
                    logger.info(f"{len(emails)} new emails found.")
                    seen_thread = set()
                    for email in emails:
                        if email["threadId"] in seen_thread:
                            # Only process the newest message of the thread if there are multiple unseen messages from same thread
                            continue
                        message = await gmail_service.read_email(email["id"])
                        prompt = generate_agent_prompt(message)
                        result = await llm.generate_str(
                            message=prompt,
                            request_params=RequestParams(use_history=False),
                        )
                        logger.info(result)
                        print(f"\n\nAssistant: {result}")
                        seen_thread.add(email["threadId"])

                await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
