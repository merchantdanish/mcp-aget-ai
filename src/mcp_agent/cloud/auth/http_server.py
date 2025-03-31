"""HTTP Server for MCP Agent Cloud authentication.

This module provides a FastAPI HTTP server for authentication endpoints
in MCP Agent Cloud.
"""

import os
import json
import uuid
import time
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, Header, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
import jwt
import redis

# Import auth service
from mcp_agent.cloud.auth.auth_service import AuthService
from mcp_agent.cloud.auth.http_handlers import MCPCloudAuthHTTPHandlers

# Create FastAPI app
app = FastAPI(title="MCP Agent Cloud Auth Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis connection if available
redis_client = None
redis_url = os.environ.get("REDIS_URL")
if redis_url:
    try:
        redis_client = redis.from_url(redis_url)
    except Exception as e:
        print(f"Error connecting to Redis: {str(e)}")

# Initialize auth service
auth_service = AuthService(config={
    "providers": {
        "self-contained": {
            "secret_key": os.environ.get("AUTH_SECRET_KEY", "demo_secret_key"),
        }
    }
})

# Initialize HTTP handlers
base_url = os.environ.get("BASE_URL", "http://localhost:8000")
http_handlers = MCPCloudAuthHTTPHandlers(auth_service, base_url)

@app.get("/")
async def root():
    """Root endpoint for the auth service."""
    return {"message": "MCP Agent Cloud Auth Service"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.get("/.well-known/oauth-authorization-server")
async def well_known(request: Request):
    """OAuth authorization server metadata."""
    return await http_handlers.well_known_handler(to_mcp_request(request))

@app.get("/authorize")
async def authorize(request: Request):
    """OAuth authorization endpoint."""
    response = await http_handlers.authorize_handler(to_mcp_request(request))
    return from_mcp_response(response)

@app.post("/authorize")
async def authorize_post(request: Request):
    """OAuth authorization endpoint (POST)."""
    mcp_request = to_mcp_request(request)
    mcp_request["method"] = "POST"
    response = await http_handlers.authorize_handler(mcp_request)
    return from_mcp_response(response)

@app.post("/token")
async def token(request: Request):
    """OAuth token endpoint."""
    response = await http_handlers.token_handler(to_mcp_request(request))
    return from_mcp_response(response)

@app.post("/register")
async def register(request: Request):
    """OAuth client registration endpoint."""
    response = await http_handlers.register_handler(to_mcp_request(request))
    return from_mcp_response(response)

@app.get("/callback")
async def callback(request: Request):
    """OAuth callback endpoint."""
    response = await http_handlers.callback_handler(to_mcp_request(request))
    return from_mcp_response(response)

@app.get("/validate_token")
async def validate_token(request: Request, authorization: Optional[str] = Header(None)):
    """Validate a token."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    token = authorization.split(" ")[1]
    is_valid, user_data = await auth_service.validate_token(token)
    
    if not is_valid:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return {"valid": True, "user": user_data}

def to_mcp_request(request: Request) -> Dict[str, Any]:
    """Convert a FastAPI request to an MCP request format."""
    return {
        "path": request.url.path,
        "method": request.method,
        "query": str(request.url.query),
        "headers": dict(request.headers),
        "body": request.body,
    }

def from_mcp_response(mcp_response: Dict[str, Any]) -> Response:
    """Convert an MCP response to a FastAPI response."""
    if mcp_response.get("status") == 302:
        return RedirectResponse(
            url=mcp_response["headers"]["Location"],
            status_code=302
        )
    
    content_type = mcp_response.get("headers", {}).get("Content-Type", "application/json")
    
    if "text/html" in content_type:
        return HTMLResponse(
            content=mcp_response["body"],
            status_code=mcp_response.get("status", 200)
        )
    
    return JSONResponse(
        content=json.loads(mcp_response["body"]) if isinstance(mcp_response["body"], str) else mcp_response["body"],
        status_code=mcp_response.get("status", 200)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)