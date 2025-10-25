"""Native accelerator facades.

This package hosts optional accelerator adapters used by the backend. Each
adapter must expose a class with an `is_available()` probe and idempotent
application methods. Missing accelerators are handled gracefully by callers.
"""

