# Adding a Custom Base URL for Anthropic in mcp-agent

This document explains how to configure a custom base URL for the Anthropic API in the mcp-agent repository.

## Overview

The mcp-agent repository supports configuring a custom base URL for the Anthropic API. This feature is useful for:

- Using proxy servers
- Connecting to alternative endpoints
- Working with custom deployments
- Routing through API gateways

## Configuration

To configure a custom base URL for Anthropic, add the `base_url` parameter to your Anthropic settings in your configuration file:

```yaml
# mcp_agent.config.yaml
anthropic:
  api_key: "your-api-key"
  base_url: "https://your-custom-endpoint.com/v1"  # Optional
```

## Example Usage

### Using with a Proxy Server

If you're using a proxy server to route your Anthropic API requests, you can configure it like this:

```yaml
anthropic:
  api_key: "your-api-key"
  base_url: "https://your-proxy-server.com/anthropic/v1"
```

### Using with LiteLLM

If you're using LiteLLM as a proxy for multiple LLM providers:

```yaml
anthropic:
  api_key: "your-litellm-api-key"
  base_url: "https://your-litellm-server.com/v1"
```

## Implementation Details

The `base_url` parameter is optional. If not provided, the client will use the default Anthropic API endpoint.

The implementation ensures:
1. Backward compatibility with existing configurations
2. The client initialization only includes the base_url parameter if it's provided
3. Both synchronous and asynchronous clients are supported

## Troubleshooting

If you encounter issues with your custom base URL:

1. Verify that your base URL is correctly formatted and accessible
2. Ensure that your proxy or custom endpoint correctly forwards requests to the Anthropic API
3. Check that your API key has the necessary permissions for the endpoint you're using

## Limitations

- The custom base URL must be compatible with the Anthropic API structure
- Some proxy servers may require additional configuration beyond just changing the base URL
- Custom endpoints must support the same API version as expected by the mcp-agent code