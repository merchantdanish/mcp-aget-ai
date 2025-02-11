#!/usr/bin/env python3
import argparse
import asyncio
import logging
import os
import webbrowser
from email.message import EmailMessage
from email.header import decode_header
from base64 import urlsafe_b64encode, urlsafe_b64decode
from email import message_from_bytes

from mcp.server.fastmcp import FastMCP
import mcp.types as types


from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("gmail")

EMAIL_ADMIN_PROMPTS = """You are an email administrator.
You can draft, edit, read, trash, open, and send emails.
You've been given access to a specific gmail account.
You have the following tools available:
- send-email (to send new emails or reply within the same thread)
- get-unread-emails
- read-email
- trash-email
- open-email
- mark-email-as-read
Never send an email draft or trash an email unless the user confirms first.
Always ask for approval if not already given.
"""


def decode_mime_header(header: str) -> str:
    """Helper function to decode encoded email headers"""
    decoded_parts = decode_header(header)
    decoded_string = ""
    for part, encoding in decoded_parts:
        if isinstance(part, bytes):
            decoded_string += part.decode(encoding or "utf-8")
        else:
            decoded_string += part
    return decoded_string


class GmailService:
    def __init__(
        self,
        creds_file_path: str,
        token_path: str,
        scopes: list[str] = ["https://www.googleapis.com/auth/gmail.modify"],
    ):
        logger.info(f"Initializing GmailService with creds file: {creds_file_path}")
        self.creds_file_path = creds_file_path
        self.token_path = token_path
        self.scopes = scopes
        self.token = self._get_token()
        logger.info("Token retrieved successfully")
        self.service = self._get_service()
        logger.info("Gmail service initialized")
        self.user_email = self._get_user_email()
        logger.info(f"User email retrieved: {self.user_email}")

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
            service = build("gmail", "v1", credentials=self.token)
            return service
        except HttpError as error:
            logger.error(f"An error occurred building Gmail service: {error}")
            raise ValueError(f"An error occurred: {error}")

    def _get_user_email(self) -> str:
        profile = self.service.users().getProfile(userId="me").execute()
        return profile.get("emailAddress", "")

    async def send_email(
        self,
        recipient_id: str,
        subject: str,
        message: str,
        thread_id: str = None,
        cc: str = None,
    ) -> dict:
        """Creates and sends an email message. If thread_id is provided, replies within the same thread."""
        try:
            message_obj = EmailMessage()
            message_obj.set_content(message)
            message_obj["To"] = recipient_id
            message_obj["From"] = self.user_email
            message_obj["Subject"] = subject
            if cc:
                message_obj["Cc"] = cc
            if thread_id:
                message_obj["In-Reply-To"] = thread_id
                message_obj["References"] = thread_id

            encoded_message = urlsafe_b64encode(message_obj.as_bytes()).decode()
            create_message = {"raw": encoded_message}
            if thread_id:
                create_message["threadId"] = thread_id

            send_message = await asyncio.to_thread(
                self.service.users()
                .messages()
                .send(userId="me", body=create_message)
                .execute
            )
            logger.info(f"Message sent: {send_message['id']}")
            return {"status": "success", "message_id": send_message["id"]}
        except HttpError as error:
            return {"status": "error", "error_message": str(error)}

    async def open_email(self, email_id: str) -> str:
        try:
            url = f"https://mail.google.com/#all/{email_id}"
            webbrowser.open(url, new=0, autoraise=True)
            return "Email opened in browser successfully."
        except HttpError as error:
            return f"An HttpError occurred: {str(error)}"

    async def get_unread_emails(self) -> list | str:
        try:
            user_id = "me"
            query = "in:inbox is:unread category:primary"
            response = (
                self.service.users().messages().list(userId=user_id, q=query).execute()
            )
            messages = response.get("messages", [])
            while "nextPageToken" in response:
                page_token = response["nextPageToken"]
                response = (
                    self.service.users()
                    .messages()
                    .list(userId=user_id, q=query, pageToken=page_token)
                    .execute()
                )
                messages.extend(response.get("messages", []))
            return messages
        except HttpError as error:
            return f"An HttpError occurred: {str(error)}"

    async def _parse_email_content(self, raw_data: str) -> dict:
        """Helper function to parse email content from raw data."""
        decoded_data = urlsafe_b64decode(raw_data)
        mime_message = message_from_bytes(decoded_data)

        # Extract plain text content
        body = None
        if mime_message.is_multipart():
            for part in mime_message.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode()
                    break
        else:
            body = mime_message.get_payload(decode=True).decode()

        return {
            "content": body,
            "subject": decode_header(mime_message.get("subject", ""))[0][0],
            "from": mime_message.get("from", ""),
            "to": mime_message.get("to", ""),
            "date": mime_message.get("date", ""),
        }

    async def read_email(self, email_id: str) -> dict | str:
        """Fetches a single email by ID and marks it as read."""
        try:
            msg = (
                self.service.users()
                .messages()
                .get(userId="me", id=email_id, format="raw")
                .execute()
            )
            email_metadata = await self._parse_email_content(msg["raw"])
            await self.mark_email_as_read(email_id)
            logger.info(f"Email read: {email_id}")
            return email_metadata
        except HttpError as error:
            return f"An HttpError occurred: {str(error)}"

    async def trash_email(self, email_id: str) -> str:
        try:
            self.service.users().messages().trash(userId="me", id=email_id).execute()
            logger.info(f"Email moved to trash: {email_id}")
            return "Email moved to trash successfully."
        except HttpError as error:
            return f"An HttpError occurred: {str(error)}"

    async def mark_email_as_read(self, email_id: str) -> str:
        try:
            self.service.users().messages().modify(
                userId="me", id=email_id, body={"removeLabelIds": ["UNREAD"]}
            ).execute()
            logger.info(f"Email marked as read: {email_id}")
            return "Email marked as read."
        except HttpError as error:
            return f"An HttpError occurred: {str(error)}"


# Global GmailService instance
gmail_service = None

# FastMCP prompts


@mcp.prompt()
def manage_emails():
    """
    Generate a prompt to behave like an email administator.
    """
    return [
        types.PromptMessage(
            role="user",
            content=types.TextContent(
                type="text",
                text=EMAIL_ADMIN_PROMPTS,
            ),
        )
    ]


@mcp.prompt()
def draft_email(content: str, recipient: str, recipient_email: str):
    """
    Generates a prompt to draft an email based on provided parameters.

    Parameters:
        content (str): The topic or content around which the email should be drafted.
        recipient (str): The name or identifier of the email recipient.
        recipient_email (str): The email address of the recipient.
    """
    return [
        types.PromptMessage(
            role="user",
            content=types.TextContent(
                type="text",
                text=f"""Please draft an email about {content} for {recipient} ({recipient_email}).
                Include a subject line starting with 'Subject:' on the first line.
                Do not send the email yet, just draft it and ask the user for their thoughts.""",
            ),
        )
    ]


@mcp.prompt()
def edit_draft(changes: str, current_draft: str):
    """
    Generates a prompt to modify an existing email draft.

    Parameters:
        changes (str): The adjustments or changes requested for the email draft.
        current_draft (str): The current email draft to be revised.
    """
    return [
        types.PromptMessage(
            role="user",
            content=types.TextContent(
                type="text",
                text=f"""Please revise the current email draft:
                {current_draft}
                
                Requested changes:
                {changes}
                
                Please provide the updated draft.""",
            ),
        )
    ]


# FastMCP tools


@mcp.tool()
async def send_email(
    recipient_id: str, subject: str, message: str, thread_id: str = None, cc: str = None
):
    """
    Sends an email message using Gmail API.

    Parameters:
        recipient_id (str): The email address of the recipient.
        subject (str): The subject line of the email. If the first line of the message starts with "Subject:", it overwrites this value.
        message (str): The body of the email message.
        thread_id (str, optional): The ID of the email thread to reply to. Default is None.
        cc (str, optional): A comma-separated list of email addresses to be carbon copied on the email. Default is None.

    Returns:
        str: Confirmation message with the sent email's ID, or an error message if sending failed.
    """
    email_lines = message.split("\n")
    if email_lines and email_lines[0].startswith("Subject:"):
        subject = email_lines[0][8:].strip()
        message = "\n".join(email_lines[1:]).strip()
    send_response = await gmail_service.send_email(
        recipient_id, subject, message, thread_id, cc
    )
    if send_response.get("status") == "success":
        return f"Email sent successfully. Message ID: {send_response.get('message_id')}"
    else:
        return f"Failed to send email: {send_response.get('error_message')}"


# @mcp.tool()
# async def get_unread_emails():
#     """
#     Retrieves unread emails from the Gmail inbox.

#     Returns:
#         str: A string representation of the list of unread emails, or an error message.
#     """
#     unread_emails = await gmail_service.get_unread_emails()
#     return str(unread_emails)


@mcp.tool()
async def read_email(email_id: str):
    """
    Retrieves an email's content and metadata using its ID and marks it as read.

    Parameters:
        email_id (str): The unique identifier of the email to be read.

    Returns:
        str: A string representation of the email's content and metadata, or an error message.
    """
    email_data = await gmail_service.read_email(email_id)
    return str(email_data)


@mcp.tool()
async def trash_email(email_id: str):
    """
    Moves the specified email to the trash folder.

    Parameters:
        email_id (str): The unique identifier of the email to be trashed.

    Returns:
        str: Status message indicating success or error details.
    """
    result = await gmail_service.trash_email(email_id)
    return result


@mcp.tool()
async def mark_email_as_read(email_id: str):
    """
    Marks the provided email as read in the Gmail inbox.

    Parameters:
        email_id (str): The unique identifier of the email to be marked as read.

    Returns:
        str: Status message indicating the result of the operation.
    """
    result = await gmail_service.mark_email_as_read(email_id)
    return result


@mcp.tool()
async def open_email(email_id: str):
    """
    Opens the specified email in the default web browser.

    Parameters:
        email_id (str): The unique identifier of the email to open.

    Returns:
        str: Confirmation message if successful, or an error description.
    """
    result = await gmail_service.open_email(email_id)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gmail API FastMCP Server")
    parser.add_argument(
        "--creds-file-path", required=True, help="OAuth 2.0 credentials file path"
    )
    parser.add_argument(
        "--token-path",
        required=True,
        help="File location to store and retrieve access and refresh tokens",
    )
    args = parser.parse_args()

    # Initialize global GmailService instance
    gmail_service = GmailService(args.creds_file_path, args.token_path)

    # Run FastMCP server using stdio transport
    mcp.run(transport="stdio")
