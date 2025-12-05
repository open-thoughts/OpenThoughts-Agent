#!/usr/bin/env python3
"""
Unified Configuration for OT-Agents

This module provides Supabase client configuration and connection management
that combines both the original Supabase configuration and terminal-bench configuration.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from supabase import create_client, Client
from supabase import ClientOptions

# Load environment variables
load_dotenv()


class SupabaseConfig:
    """Configuration class for Supabase connection."""
    
    @staticmethod
    def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
        """Get setting from environment variables."""
        env_val = os.environ.get(key.upper())
        if env_val is not None:
            return env_val
        return default

    @property
    def supabase_url(self) -> str:
        """Supabase project URL."""
        return self.get_setting("SUPABASE_URL", "")

    @property
    def supabase_anon_key(self) -> str:
        """Supabase anonymous/public key."""
        return self.get_setting("SUPABASE_ANON_KEY", "")

    @property
    def supabase_service_role_key(self) -> str:
        """Supabase service role key for admin operations."""
        return self.get_setting("SUPABASE_SERVICE_ROLE_KEY", "")

    @property
    def is_configured(self) -> bool:
        """Check if Supabase is properly configured."""
        return bool(self.supabase_url and self.supabase_anon_key)

    @property
    def has_admin_access(self) -> bool:
        """Check if admin access is available."""
        return bool(self.supabase_service_role_key)


# Create config instance
supabase_config = SupabaseConfig()


def create_supabase_client(use_admin: bool = False) -> Client:
    """
    Create and configure Supabase client.
    
    Args:
        use_admin: If True, use service role key for admin operations
        
    Returns:
        Configured Supabase client
        
    Raises:
        ValueError: If Supabase is not properly configured
    """
    if not supabase_config.is_configured:
        raise ValueError(
            "Supabase not configured. Please set SUPABASE_URL and SUPABASE_ANON_KEY "
            "in your environment variables."
        )
    
    # Choose the appropriate key
    if use_admin and supabase_config.has_admin_access:
        key = supabase_config.supabase_service_role_key
        # print("🔑 Using service role key for admin operations")
    else:
        key = supabase_config.supabase_anon_key
        if use_admin:
            print("⚠️  Admin access requested but no service role key found")
            print("   Some operations may fail due to RLS policies")
    
    # Create client (v2 API doesn't use ClientOptions the same way)
    return create_client(supabase_config.supabase_url, key)


def get_default_client() -> Client:
    """Get default Supabase client for regular operations."""
    return create_supabase_client(use_admin=False)


def get_admin_client() -> Client:
    """Get admin Supabase client for operations that bypass RLS."""
    return create_supabase_client(use_admin=True)


def test_connection() -> bool:
    """Test the connection to Supabase."""
    try:
        client = get_default_client()
        # Try a simple query to test connection
        response = client.table('datasets').select('id').limit(1).execute()
        print("✅ Connected to Supabase successfully!")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


if __name__ == "__main__":
    # Test configuration
    print("🔧 Supabase Configuration Test")
    print("=" * 40)
    print(f"URL configured: {'✅' if supabase_config.supabase_url else '❌'}")
    print(f"Anon key configured: {'✅' if supabase_config.supabase_anon_key else '❌'}")
    print(f"Service role key configured: {'✅' if supabase_config.supabase_service_role_key else '❌'}")
    print(f"Admin access available: {'✅' if supabase_config.has_admin_access else '❌'}")
    
    if supabase_config.is_configured:
        print("\n🧪 Testing connection...")
        test_connection()
    else:
        print("\n❌ Please configure your environment variables:")
        print("   SUPABASE_URL=your_project_url")
        print("   SUPABASE_ANON_KEY=your_anon_key")
        print("   SUPABASE_SERVICE_ROLE_KEY=your_service_role_key (optional)")