"""Service credential manager for MCP Agent Cloud.

This module provides functionality for securely storing and retrieving
credentials for third-party services that MCPApps need to access.
"""

import os
import json
import uuid
import base64
import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime, timezone
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

class CredentialStore:
    """Base class for credential storage."""
    
    async def put(self, credential_id: str, credential_data: Dict[str, Any]) -> None:
        """Store credential data.
        
        Args:
            credential_id: Unique ID for the credential
            credential_data: Credential data to store
        """
        raise NotImplementedError("Subclasses must implement put()")
    
    async def get(self, credential_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve credential data.
        
        Args:
            credential_id: Unique ID for the credential
            
        Returns:
            Credential data if found, None otherwise
        """
        raise NotImplementedError("Subclasses must implement get()")
    
    async def delete(self, credential_id: str) -> bool:
        """Delete credential data.
        
        Args:
            credential_id: Unique ID for the credential
            
        Returns:
            True if deleted, False otherwise
        """
        raise NotImplementedError("Subclasses must implement delete()")
    
    async def list(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List credential metadata.
        
        Args:
            agent_id: Optional agent ID to filter by
            
        Returns:
            List of credential metadata
        """
        raise NotImplementedError("Subclasses must implement list()")

class FileCredentialStore(CredentialStore):
    """File-based credential storage.
    
    This implementation stores credentials as encrypted files on the local filesystem.
    It is intended for development and testing purposes only.
    In production, a more secure storage mechanism like a managed key vault would be used.
    """
    
    def __init__(self, store_dir: Optional[Path] = None):
        """Initialize the file credential store.
        
        Args:
            store_dir: Directory to store credentials in
        """
        self.store_dir = store_dir or Path(os.path.expanduser("~/.mcp-agent-cloud/credentials"))
        os.makedirs(self.store_dir, exist_ok=True)
        
        # Set secure permissions on the credential directory
        if os.name != 'nt':  # Not Windows
            os.chmod(self.store_dir, 0o700)  # Owner only permissions
    
    async def put(self, credential_id: str, credential_data: Dict[str, Any]) -> None:
        """Store credential data in a file.
        
        Args:
            credential_id: Unique ID for the credential
            credential_data: Credential data to store
        """
        file_path = self.store_dir / f"{credential_id}.json"
        
        try:
            with open(file_path, "w") as f:
                json.dump(credential_data, f, indent=2)
            
            # Set secure permissions on the credential file
            if os.name != 'nt':  # Not Windows
                os.chmod(file_path, 0o600)  # Owner only permissions
        except IOError as e:
            logger.error(f"Error saving credential {credential_id}: {str(e)}")
            raise
    
    async def get(self, credential_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve credential data from a file.
        
        Args:
            credential_id: Unique ID for the credential
            
        Returns:
            Credential data if found, None otherwise
        """
        file_path = self.store_dir / f"{credential_id}.json"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading credential {credential_id}: {str(e)}")
            return None
    
    async def delete(self, credential_id: str) -> bool:
        """Delete credential data file.
        
        Args:
            credential_id: Unique ID for the credential
            
        Returns:
            True if deleted, False otherwise
        """
        file_path = self.store_dir / f"{credential_id}.json"
        
        if not file_path.exists():
            return False
        
        try:
            os.remove(file_path)
            return True
        except IOError as e:
            logger.error(f"Error deleting credential {credential_id}: {str(e)}")
            return False
    
    async def list(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List credential metadata from files.
        
        Args:
            agent_id: Optional agent ID to filter by
            
        Returns:
            List of credential metadata
        """
        result = []
        
        for file_path in self.store_dir.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    cred_data = json.load(f)
                
                # Skip if not matching agent_id
                if agent_id and cred_data.get("agent_id") != agent_id:
                    continue
                
                # Remove sensitive data
                if "encrypted_data" in cred_data:
                    del cred_data["encrypted_data"]
                
                result.append(cred_data)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading credential {file_path.stem}: {str(e)}")
        
        return result

class ServiceCredentialManager:
    """Manages credentials for third-party services that MCPApps need to access.
    
    This manager provides secure storage and retrieval of credentials for external services
    like Gmail, GitHub, etc. that agents need to interact with.
    """
    
    def __init__(self, 
                 credential_store: Optional[CredentialStore] = None,
                 encryption_key: Optional[bytes] = None):
        """Initialize the service credential manager.
        
        Args:
            credential_store: Store for credentials
            encryption_key: Key for encrypting credentials
        """
        self.credential_store = credential_store or FileCredentialStore()
        
        # Generate or use encryption key
        if encryption_key:
            self.encryption_key = encryption_key
        else:
            # In production, this would be a securely stored master key
            # For development, we'll derive a key from a hardcoded password
            password = b"mcp-agent-cloud-dev-key"
            salt = b"mcp-agent-cloud-salt"
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            self.encryption_key = base64.urlsafe_b64encode(kdf.derive(password))
        
        # Initialize Fernet for encryption
        self.fernet = Fernet(self.encryption_key)
    
    def _encrypt_credentials(self, credentials: Dict[str, Any]) -> bytes:
        """Encrypt credentials data.
        
        Args:
            credentials: Credentials to encrypt
            
        Returns:
            Encrypted credentials
        """
        data = json.dumps(credentials).encode()
        return self.fernet.encrypt(data)
    
    def _decrypt_credentials(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt credentials data.
        
        Args:
            encrypted_data: Encrypted credentials
            
        Returns:
            Decrypted credentials
        """
        data = self.fernet.decrypt(encrypted_data)
        return json.loads(data.decode())
    
    async def store_credential(self, 
                             agent_id: str, 
                             service_name: str, 
                             credentials: Dict[str, Any]) -> str:
        """Store service credentials securely in the credential vault.
        
        Args:
            agent_id: ID of the agent the credentials belong to
            service_name: Name of the service (e.g., "gmail", "github")
            credentials: Credentials data
            
        Returns:
            Credential ID
        """
        # Generate a unique credential ID
        credential_id = f"cred-{uuid.uuid4().hex[:8]}"
        
        # Encrypt credentials
        encrypted_creds = self._encrypt_credentials(credentials)
        
        # Store in credential store
        await self.credential_store.put(
            credential_id,
            {
                "id": credential_id,
                "agent_id": agent_id,
                "service_name": service_name,
                "encrypted_data": encrypted_creds,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        )
        
        return credential_id
    
    async def get_credential(self, credential_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve service credentials from the vault.
        
        Args:
            credential_id: Credential ID
            
        Returns:
            Decrypted credentials if found, None otherwise
        """
        # Get encrypted credentials
        cred_data = await self.credential_store.get(credential_id)
        if not cred_data or "encrypted_data" not in cred_data:
            return None
        
        # Decrypt credentials
        try:
            decrypted_creds = self._decrypt_credentials(cred_data["encrypted_data"])
            return decrypted_creds
        except Exception as e:
            logger.error(f"Error decrypting credential {credential_id}: {str(e)}")
            return None
    
    async def delete_credential(self, credential_id: str) -> bool:
        """Delete service credentials from the vault.
        
        Args:
            credential_id: Credential ID
            
        Returns:
            True if deleted, False otherwise
        """
        return await self.credential_store.delete(credential_id)
    
    async def list_credentials(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List service credentials metadata.
        
        Args:
            agent_id: Optional agent ID to filter by
            
        Returns:
            List of credential metadata
        """
        return await self.credential_store.list(agent_id)
    
    async def rotate_credential(self, 
                              credential_id: str, 
                              new_credentials: Dict[str, Any]) -> bool:
        """Rotate service credentials.
        
        Args:
            credential_id: Credential ID
            new_credentials: New credentials data
            
        Returns:
            True if rotated, False otherwise
        """
        # Get existing credential data
        cred_data = await self.credential_store.get(credential_id)
        if not cred_data:
            return False
        
        # Update with new encrypted credentials
        cred_data["encrypted_data"] = self._encrypt_credentials(new_credentials)
        cred_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        # Store updated credentials
        await self.credential_store.put(credential_id, cred_data)
        
        return True