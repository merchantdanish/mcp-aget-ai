"""Application registry for MCP Agent Cloud.

This module provides a registry for tracking MCPApps deployed to the cloud.
It supports in-memory and persistent storage options.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class AppRegistry:
    """Registry for tracking MCPApps."""
    
    def __init__(self, registry_file: str = None, redis_client = None):
        """Initialize the app registry.
        
        Args:
            registry_file: Path to the registry file for persistent storage
            redis_client: Redis client for distributed storage
        """
        self.registry_file = registry_file or os.path.expanduser("~/.mcp-agent-cloud/apps.json")
        self.redis_client = redis_client
        self.apps = {}
        
        # Ensure registry directory exists
        if registry_file:
            os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
        
        # Load existing registry
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load the registry from storage."""
        # If redis client is available, use it
        if self.redis_client:
            try:
                # Get all apps from Redis
                for key in self.redis_client.keys("app:*"):
                    app_data = json.loads(self.redis_client.get(key))
                    self.apps[app_data["id"]] = app_data
            except Exception as e:
                logger.error(f"Error loading registry from Redis: {str(e)}")
                # Fall back to file-based registry
                self._load_file_registry()
        else:
            # Use file-based registry
            self._load_file_registry()
    
    def _load_file_registry(self) -> None:
        """Load the registry from a file."""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r") as f:
                    self.apps = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error decoding registry file: {self.registry_file}")
                self.apps = {}
            except IOError as e:
                logger.error(f"Error reading registry file: {str(e)}")
                self.apps = {}
    
    def _save_registry(self) -> None:
        """Save the registry to storage."""
        # If redis client is available, use it
        if self.redis_client:
            try:
                # Save apps to Redis
                for app_id, app_data in self.apps.items():
                    key = f"app:{app_id}"
                    self.redis_client.set(key, json.dumps(app_data))
            except Exception as e:
                logger.error(f"Error saving registry to Redis: {str(e)}")
                # Fall back to file-based registry
                self._save_file_registry()
        else:
            # Use file-based registry
            self._save_file_registry()
    
    def _save_file_registry(self) -> None:
        """Save the registry to a file."""
        try:
            with open(self.registry_file, "w") as f:
                json.dump(self.apps, f, indent=2)
        except IOError as e:
            logger.error(f"Error writing registry file: {str(e)}")
    
    def register_app(self, app: Dict[str, Any]) -> None:
        """Register a new app.
        
        Args:
            app: App data to register
        """
        app_id = app.get("id")
        if not app_id:
            raise ValueError("App ID is required")
            
        self.apps[app_id] = app
        self._save_registry()
        logger.info(f"Registered app: {app_id}")
    
    def update_app(self, app_id: str, app: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an existing app.
        
        Args:
            app_id: ID of the app to update
            app: Updated app data
            
        Returns:
            Updated app data or None if app not found
        """
        if app_id not in self.apps:
            return None
            
        self.apps[app_id].update(app)
        self._save_registry()
        logger.info(f"Updated app: {app_id}")
        return self.apps[app_id]
    
    def get_app(self, app_id: str) -> Optional[Dict[str, Any]]:
        """Get an app by ID.
        
        Args:
            app_id: ID of the app to get
            
        Returns:
            App data or None if app not found
        """
        return self.apps.get(app_id)
    
    def get_app_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get an app by name.
        
        Args:
            name: Name of the app to get
            
        Returns:
            App data or None if app not found
        """
        for app in self.apps.values():
            if app.get("name") == name:
                return app
        return None
    
    def delete_app(self, app_id: str) -> bool:
        """Delete an app.
        
        Args:
            app_id: ID of the app to delete
            
        Returns:
            True if app was deleted, False otherwise
        """
        if app_id not in self.apps:
            return False
            
        del self.apps[app_id]
        self._save_registry()
        logger.info(f"Deleted app: {app_id}")
        return True
    
    def list_apps(self) -> List[Dict[str, Any]]:
        """List all registered apps.
        
        Returns:
            List of app data
        """
        return list(self.apps.values())