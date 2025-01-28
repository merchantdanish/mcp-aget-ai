import asyncio
import streamlit as st
from mcp_agent.human_input.types import (
    HumanInputCallback,
    HumanInputResponse,
)


class StreamlitHumanInputHandler(HumanInputCallback):
    """
    Handles human input requests in a Streamlit application while managing state
    and async communication with the MCP agent.
    """

    def __init__(self):
        if "pending_requests" not in st.session_state:
            st.session_state.pending_requests = {}

    async def __call__(self, request):
        request_id = request.request_id

        form_key = f"input_form_{request_id}"

        # Add request to pending if not already there
        if request_id not in st.session_state.pending_requests:
            st.session_state.pending_requests[request_id] = {
                "request": request,
                "start_time": asyncio.get_event_loop().time(),
                "response": None,
                "completed": False,
            }
        else:
            # If request is already completed, return the response
            if st.session_state.pending_requests[request_id].get("completed"):
                response = st.session_state.pending_requests[request_id]["response"]
                return response

        # Check timeout
        if request.timeout_seconds:
            start_time = st.session_state.pending_requests[request_id]["start_time"]
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= request.timeout_seconds:
                del st.session_state.pending_requests[request_id]
                raise TimeoutError("Request timed out waiting for human input")

        # Display input form
        with st.form(key=form_key):
            if request.description:
                st.info(f"**[HUMAN INPUT NEEDED]** {request.description}")

            response_text = st.text_input(request.prompt, key=f"input_{request_id}")

            submitted = st.form_submit_button("Submit")

            if submitted and response_text.strip():
                # Create response and store it
                response = HumanInputResponse(
                    request_id=request_id, response=response_text.strip()
                )
                st.session_state.pending_requests[request_id] = {
                    "request": request,
                    "response": response,
                    "completed": True,
                }

                # Return the final response
                return response

        # Return None to indicate we're still waiting for input
        return None


streamlit_input_callback = StreamlitHumanInputHandler()
"""
Process the human input request via Streamlit UI.
Yields responses when they become available.
"""
