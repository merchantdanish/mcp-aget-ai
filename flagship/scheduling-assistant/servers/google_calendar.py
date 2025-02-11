#!/usr/bin/env python
import os
import argparse
import asyncio
import logging

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Calendar API scope
SCOPES = ["https://www.googleapis.com/auth/calendar"]


class CalendarService:
    def __init__(self, creds_file_path: str, token_path: str):
        logger.info(f"Initializing CalendarService with creds file: {creds_file_path}")
        self.creds_file_path = creds_file_path
        self.token_path = token_path
        self.scopes = SCOPES
        self.creds = self._get_token()
        logger.info("Token retrieved successfully")
        self.service = self._get_service()
        logger.info("Google Calendar service initialized")

    def _get_token(self) -> Credentials:
        token = None
        if os.path.exists(self.token_path):
            logger.info("Loading token from file")
            token = Credentials.from_authorized_user_file(self.token_path, self.scopes)
        if not token or not token.valid:
            if token and token.expired and token.refresh_token:
                logger.info("Refreshing token")
                token.refresh(Request())
            else:
                logger.info("Fetching new token")
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.creds_file_path, self.scopes
                )
                token = flow.run_local_server(port=0)
            with open(self.token_path, "w") as token_file:
                token_file.write(token.to_json())
                logger.info(f"Token saved to {self.token_path}")
        return token

    def _get_service(self):
        try:
            service = build("calendar", "v3", credentials=self.creds)
            return service
        except HttpError as error:
            logger.error(f"Error initializing Calendar service: {error}")
            raise ValueError(f"An error occurred: {error}")

    async def list_events(self) -> dict:
        try:
            events_result = await asyncio.to_thread(
                self.service.events().list, calendarId="primary"
            )
            events = await asyncio.to_thread(events_result.execute)
            return events
        except HttpError as error:
            logger.error(f"Error listing events: {error}")
            return {"error": str(error)}

    async def create_event(
        self,
        summary: str,
        start: str,
        end: str,
        description: str = "",
        attendees: str = None,
    ) -> dict:
        event_body = {
            "summary": summary,
            "description": description,
            "start": {"dateTime": start},
            "end": {"dateTime": end},
        }
        if attendees:
            # Expecting a comma-separated string of email addresses
            event_body["attendees"] = [
                {"email": email.strip()} for email in attendees.split(",")
            ]
        try:
            created_event = await asyncio.to_thread(
                self.service.events().insert, calendarId="primary", body=event_body
            )
            result = await asyncio.to_thread(created_event.execute)
            return result
        except HttpError as error:
            logger.error(f"Error creating event: {error}")
            return {"error": str(error)}

    async def update_event(
        self,
        event_id: str,
        summary: str = None,
        start: str = None,
        end: str = None,
        description: str = None,
        attendees: str = None,
    ) -> dict:
        update_data = {}
        if summary is not None:
            update_data["summary"] = summary
        if description is not None:
            update_data["description"] = description
        if start is not None:
            update_data.setdefault("start", {})["dateTime"] = start
        if end is not None:
            update_data.setdefault("end", {})["dateTime"] = end
        if attendees is not None:
            update_data["attendees"] = [
                {"email": email.strip()} for email in attendees.split(",")
            ]
        try:
            updated_event = await asyncio.to_thread(
                self.service.events().patch,
                calendarId="primary",
                eventId=event_id,
                body=update_data,
            )
            result = await asyncio.to_thread(updated_event.execute)
            return result
        except HttpError as error:
            logger.error(f"Error updating event: {error}")
            return {"error": str(error)}

    async def delete_event(self, event_id: str) -> dict:
        try:
            delete_request = await asyncio.to_thread(
                self.service.events().delete, calendarId="primary", eventId=event_id
            )
            await asyncio.to_thread(delete_request.execute)
            return {"status": "success", "message": f"Event {event_id} deleted."}
        except HttpError as error:
            logger.error(f"Error deleting event: {error}")
            return {"error": str(error)}


# Global variable to hold the CalendarService instance
calendar_service: CalendarService = None

# Initialize FastMCP server
mcp = FastMCP("google_calendar")


@mcp.tool()
async def list_events() -> str:
    """Lists all events from the primary calendar"""
    events = await calendar_service.list_events()
    return str(events)


@mcp.tool()
async def create_event(
    summary: str, start: str, end: str, description: str = "", attendees: str = None
) -> str:
    """
    Creates a calendar event.
    "start" and "end" should be in RFC3339 format (e.g., 2025-09-02T14:00:00).
    Optionally include attendees as a comma-separated list of email addresses.
    """
    result = await calendar_service.create_event(
        summary, start, end, description, attendees
    )
    return str(result)


@mcp.tool()
async def update_event(
    event_id: str,
    summary: str = None,
    start: str = None,
    end: str = None,
    description: str = None,
    attendees: str = None,
) -> str:
    """
    Updates an existing calendar event.
    Only provided parameters will be updated.
    Optionally update attendees by providing a comma-separated list of email addresses.
    """
    result = await calendar_service.update_event(
        event_id, summary, start, end, description, attendees
    )
    return str(result)


@mcp.tool()
async def delete_event(event_id: str) -> str:
    """
    Deletes a calendar event by event_id.
    """
    result = await calendar_service.delete_event(event_id)
    return str(result)


def main():
    parser = argparse.ArgumentParser(description="Google Calendar FastMCP Server")
    parser.add_argument(
        "--creds-file-path", required=True, help="Path to OAuth2 credentials file"
    )
    parser.add_argument("--token-path", required=True, help="Path to store token file")
    args = parser.parse_args()

    global calendar_service
    calendar_service = CalendarService(args.creds_file_path, args.token_path)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
