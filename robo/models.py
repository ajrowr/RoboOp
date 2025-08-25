
from types import SimpleNamespace

MODELS = SimpleNamespace(
    # Claude 4 Models (Latest Generation)
    CLAUDE_4_1_OPUS = 'claude-opus-4-1-20250805',
    CLAUDE_4_OPUS = "claude-opus-4-20250514",
    CLAUDE_4_SONNET = "claude-sonnet-4-20250514",
    
    # Claude 3.7 Models
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet-20250219",
    CLAUDE_3_7_SONNET_LATEST = "claude-3-7-sonnet-latest",
    
    # Claude 3.5 Models
    CLAUDE_3_5_SONNET_V2 = "claude-3-5-sonnet-20241022",
    CLAUDE_3_5_SONNET_V1 = "claude-3-5-sonnet-20240620",
    CLAUDE_3_5_SONNET_LATEST = "claude-3-5-sonnet-latest",
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-20241022",
    CLAUDE_3_5_HAIKU_LATEST = "claude-3-5-haiku-latest",
    
    # Claude 3 Models (Legacy)
    CLAUDE_3_OPUS = "claude-3-opus-20240229",
    CLAUDE_3_OPUS_LATEST = "claude-3-opus-latest",
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229",
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307",
    
    # Convenience aliases for latest models
    LATEST_OPUS = "claude-opus-4-20250514",
    LATEST_SONNET = "claude-sonnet-4-20250514",
    LATEST_HAIKU = "claude-3-5-haiku-latest",
    
    # Model families for easy access
    CLAUDE_4 = SimpleNamespace(
        OPUS = "claude-opus-4-20250514",
        SONNET = "claude-sonnet-4-20250514",
    ),
    
    CLAUDE_3_7 = SimpleNamespace(
        SONNET = "claude-3-7-sonnet-20250219",
        SONNET_LATEST = "claude-3-7-sonnet-latest",
    ),
    
    CLAUDE_3_5 = SimpleNamespace(
        SONNET_V2 = "claude-3-5-sonnet-20241022",
        SONNET_V1 = "claude-3-5-sonnet-20240620", 
        SONNET_LATEST = "claude-3-5-sonnet-latest",
        HAIKU = "claude-3-5-haiku-20241022",
        HAIKU_LATEST = "claude-3-5-haiku-latest",
    ),
    
    CLAUDE_3 = SimpleNamespace(
        OPUS = "claude-3-opus-20240229",
        OPUS_LATEST = "claude-3-opus-latest",
        SONNET = "claude-3-sonnet-20240229",
        HAIKU = "claude-3-haiku-20240307",
    )
)

# Usage examples:
# MODELS.CLAUDE_4_OPUS
# MODELS.LATEST_SONNET
# MODELS.CLAUDE_3_5.SONNET_V2
# MODELS.CLAUDE_4.OPUS

__all__ = ['MODELS']
