"""
Global user context management for Bitomia.
This module provides a singleton class to store and manage user information globally.
"""

from typing import Optional
from threading import Lock


class UserContext:
    """
    Singleton class to store and manage current user information globally.
    Thread-safe implementation to handle concurrent requests.
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._user_id: Optional[str] = None
        self._username: Optional[str] = None
        self._home_dir: Optional[str] = None
        self._email: Optional[str] = None
        self._data_lock = Lock()
    
    def update_from_request(self, user_state) -> None:
        """
        Update user context from request.state.user object.
        
        Args:
            user_state: The user object from request.state.user
        """
        if user_state is None:
            return
        
        with self._data_lock:
            # Update user information from the user state object
            self._user_id = getattr(user_state, 'id', None) or getattr(user_state, 'user_id', None)
            self._username = getattr(user_state, 'username', None) or getattr(user_state, 'name', None)
            self._home_dir = getattr(user_state, 'homeDir', None) or getattr(user_state, 'home_dir', None)
            self._email = getattr(user_state, 'email', None)
    
    @property
    def user_id(self) -> Optional[str]:
        """Get current user ID."""
        with self._data_lock:
            return self._user_id
    
    @property
    def username(self) -> Optional[str]:
        """Get current username."""
        with self._data_lock:
            return self._username
    
    @property
    def home_dir(self) -> Optional[str]:
        """Get current user's home directory."""
        with self._data_lock:
            return self._home_dir
    
    @property
    def email(self) -> Optional[str]:
        """Get current user's email."""
        with self._data_lock:
            return self._email
    
    def clear(self) -> None:
        """Clear all user information."""
        with self._data_lock:
            self._user_id = None
            self._username = None
            self._home_dir = None
            self._email = None
    
    def to_dict(self) -> dict:
        """
        Export user context as a dictionary.
        
        Returns:
            Dictionary containing user information
        """
        with self._data_lock:
            return {
                'user_id': self._user_id,
                'username': self._username,
                'home_dir': self._home_dir,
                'email': self._email
            }
    
    def __repr__(self) -> str:
        return f"UserContext(user_id={self.user_id}, username={self.username}, home_dir={self.home_dir})"


# Global instance
user_context = UserContext()
