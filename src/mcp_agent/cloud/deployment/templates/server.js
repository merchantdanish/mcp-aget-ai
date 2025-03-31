const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const { exec } = require('child_process');

const app = express();
const port = process.env.PORT || 8000;

// Get server type from environment variable
const serverType = process.env.SERVER_TYPE || 'generic';

// Middleware
app.use(cors());
app.use(bodyParser.json());

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok', server: serverType });
});

// MCP endpoint - simulated for development/demo purposes
app.post('/servers/:serverName/:endpoint', (req, res) => {
  const serverName = req.params.serverName;
  const endpoint = req.params.endpoint;
  console.log(`Request received for server ${serverName}, endpoint ${endpoint}:`, req.body);
  
  // Based on server type, return different responses
  switch (serverType) {
    case 'fetch':
      handleFetchRequest(req, res);
      break;
    case 'http':
      handleHttpRequest(req, res);
      break;
    default:
      res.json({
        error: {
          message: `Unknown server type: ${serverType}`
        }
      });
  }
});

// For fetch server: Simulated fetch response
function handleFetchRequest(req, res) {
  const { url } = req.body;
  
  if (!url) {
    return res.json({
      error: {
        message: 'URL is required'
      }
    });
  }
  
  // Simulate a fetch response
  setTimeout(() => {
    res.json({
      result: {
        content: `Simulated content from ${url}`,
        statusCode: 200,
        headers: {
          'content-type': 'text/html'
        }
      }
    });
  }, 500);
}

// For HTTP server: Simulated HTTP response
function handleHttpRequest(req, res) {
  const { method, url, headers, body } = req.body;
  
  if (!url) {
    return res.json({
      error: {
        message: 'URL is required'
      }
    });
  }
  
  // Simulate an HTTP response
  setTimeout(() => {
    res.json({
      result: {
        statusCode: 200,
        body: `Simulated response for ${method || 'GET'} ${url}`,
        headers: {
          'content-type': 'application/json'
        }
      }
    });
  }, 500);
}

// Start the server
app.listen(port, () => {
  console.log(`MCP ${serverType} server listening at http://localhost:${port}`);
});