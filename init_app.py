"""
Application Initialization and Agent Registration.

This module initializes all components of the Agentic AI system:
- Registers tools
- Registers agents
- Configures logging
"""

import logging
import sys


def configure_console_encoding() -> None:
    """Make Windows consoles tolerant of emoji and Turkish text output."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def setup_logging(level: int = logging.INFO) -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # Reduce noise from external libraries
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def register_tools() -> None:
    """Register all tools in the global registry."""
    from tools.base import register_all_tools
    from config import STEAM_API_KEY
    register_all_tools(steam_api_key=STEAM_API_KEY)


def register_agents() -> None:
    """Register all agents in the global registry."""
    from agents.base import AgentRegistry
    from agents.profile_agent import ProfileAgent
    from agents.retrieval_agent import RetrievalAgent
    from agents.ranker_agent import RankerAgent
    from agents.matchmaking_agent import MatchmakingAgent
    
    registry = AgentRegistry()
    
    # Register each agent
    registry.register(ProfileAgent())
    registry.register(RetrievalAgent())
    registry.register(RankerAgent())
    registry.register(MatchmakingAgent())
    
    logging.info(f"Registered {len(registry.list_agents())} agents")


def initialize_app() -> None:
    """Initialize the entire Agentic AI system."""
    configure_console_encoding()

    print("\n" + "=" * 70)
    print("🚀 INITIALIZING AGENTIC AI SYSTEM")
    print("=" * 70)
    
    # Setup logging
    setup_logging()
    
    # Register tools
    print("📦 Registering tools...")
    register_tools()
    
    # Register agents
    print("🤖 Registering agents...")
    register_agents()
    
    print("✅ Agentic AI System initialized successfully!")
    print("=" * 70 + "\n")


# Initialize on import
_initialized = False

def ensure_initialized():
    """Ensure the system is initialized (call once)."""
    global _initialized
    if not _initialized:
        initialize_app()
        _initialized = True
