# Scheduling Assistant

A scheduling assistant that helps you triage meetings, suggest alternative time slots based on your calendar, and create calendar events for you and your invitees. Simply cc the agent when setting up meetings involving multiple parties.

## Installation

Install dependencies from `requirements.txt`:

```bash
uv add -r requirements.txt
```

## Setup

Before you begin, create a new [Google Cloud project](https://console.cloud.google.com/projectcreate).

### Gmail API Setup

1. [Enable the Gmail API](https://console.cloud.google.com/workspace-api/products).
2. [Configure an OAuth consent screen](https://console.cloud.google.com/apis/credentials/consent):
   - Select **External** (note that the app won't be published).
   - Add your personal email address as a **Test user**.
3. Add the OAuth scope: `https://www.googleapis.com/auth/gmail/modify`.

### Google Calendar API Setup

1. [Enable the Google Calendar API](https://console.cloud.google.com/workspace-api/products).
2. Add the OAuth scopes:
   - `https://www.googleapis.com/auth/calendar.events`
   - `https://www.googleapis.com/auth/calendar`

### Credentials Setup

1. [Create an OAuth Client ID](https://console.cloud.google.com/apis/credentials/oauthclient) with the application type **Desktop App**.
2. Download the JSON file containing your client's OAuth keys.
3. Rename the key file and save it to a secure location on your machine. Note the file path.
   - This absolute path will be passed as the parameter `--creds-file-path` when starting the server.

Finally, update your `mcp_agent.config.yaml` file:

```yaml
mcp:
  servers:
    gmail:
      command: "/opt/homebrew/bin/uv"
      args:
        [
          "--directory",
          "<directory of gmail mcp server>",
          "run",
          "gmail.py",
          "--creds-file-path",
          "<credentials file path>",
          "--token-path",
          "<path for the soon-to-be-created gmail token file>",
        ]
    calendar:
      command: "/opt/homebrew/bin/uv"
      args:
        [
          "--directory",
          "<directory of google calendar mcp server>",
          "run",
          "google_calendar.py",
          "--creds-file-path",
          "<credentials file path>",
          "--token-path",
          "<path for the soon-to-be-created google calendar token file>",
        ]
```

## Usage

Start the assistant with:

```bash
uv run main.py
```

On the first startup, you will be prompted to authorize the Google Calendar and Gmail APIs. Log in using the **Google account dedicated to your assistant**.

Once authorized, the agent will continuously poll the Gmail inbox for unread messages and process them accordingly.