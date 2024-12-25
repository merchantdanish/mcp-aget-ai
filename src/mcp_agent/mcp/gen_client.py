# from contextlib import asynccontextmanager
# from typing import Dict, Any
# from mcp import ClientSession
# from ..context import get_current_context


# async def handle_sampling_request(request, responder):
#     ctx = get_current_context()
#     session = ctx.upstream_session
#     if session is None:
#         # TODO: saqadri - consider handling the sampling request here as a client
#         await responder.send_error(
#             code=-32603, message="No upstream client available for sampling requests."
#         )
#         return
#     params = request["params"]
#     try:
#         result = await session.create_message(**params)
#         await responder.send_result(result)
#     except Exception as e:
#         await responder.send_error(code=-32603, message=str(e))


# class MCPClient:
#     def __init__(self, session: ClientSession):
#         self.session = session

#     async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
#         return await self.session.call_tool(name=name, arguments=arguments)

#     async def get_prompt(
#         self, name: str, arguments: Dict[str, Any] = {}
#     ) -> Dict[str, Any]:
#         return await self.session.get_prompt(name=name, arguments=arguments)

#     async def list_resources(self) -> Dict[str, Any]:
#         return await self.session.list_resources()

#     async def read_resource(self, uri: str) -> Dict[str, Any]:
#         return await self.session.read_resource(uri=uri)


# async def connect_transport(cfg: Dict[str, Any]):
#     transport = cfg.get("transport", "stdio")
#     if transport == "stdio":
#         from mcp.client import stdio_client, StdioServerParameters

#         return stdio_client(
#             StdioServerParameters(command=cfg["command"], args=cfg["args"])
#         )


# @asynccontextmanager
# async def gen_client(server_name: str, registry: Dict[str, Any] = SERVER_REGISTRY):
#     cfg = registry[server_name]
#     auth = cfg.get("auth", {})
#     init_hook_name = cfg.get("init_hook", None)

#     transport_ctx = await connect_transport(cfg)
#     async with transport_ctx as (r, w):
#         async with ClientSession(r, w) as session:
#             await session.initialize()
#             session.set_request_handler(
#                 "sampling/createMessage", handle_sampling_request
#             )
#             if init_hook_name and init_hook_name in INIT_HOOKS:
#                 init_hook = INIT_HOOKS[init_hook_name]
#                 await init_hook(session, auth)
#             yield MCPClient(session)
