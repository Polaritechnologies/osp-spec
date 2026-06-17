"""
OSP Frame Constants
Open Smart Protocol v0.2

All values are derived from the OSP v0.2 specification.
"""

# ─── Protocol version ────────────────────────────────────────────────────────

OSP_VERSION: int = 0x02

# ─── Header layout ───────────────────────────────────────────────────────────

HEADER_SIZE: int = 20          # bytes, fixed
HEADER_STRUCT_FMT: str = "!BBHIIIHH"   # big-endian, packed, no padding
                                        # B  version       (1)
                                        # B  flags         (1)
                                        # H  content_type  (2)
                                        # I  publisher_id  (4)
                                        # I  sequence      (4)
                                        # I  timestamp     (4)
                                        # H  payload_len   (2)
                                        # H  checksum      (2)
                                        #                 ──
                                        #                 20 bytes

EXTENDED_HEADER_EXPIRY_SIZE: int = 8   # additional bytes when EXPIRES flag set
                                        # 8-byte Unix timestamp (uint64, big-endian)

MAX_PAYLOAD_TIER0: int = 160    # bytes — Tier 0 hard limit per spec
MAX_PAYLOAD_TIER1: int = 2048   # bytes — Tier 1 limit (2 KB)
MAX_PAYLOAD_TIER2: int = 65536  # bytes — Tier 2 limit (64 KB)
# Tier 3: no size constraint

# ─── Flags byte (bit positions) ──────────────────────────────────────────────

class Flags:
    """
    Bit masks for the flags byte (offset 1 in the OSP frame header).

    Bits 5-7 are reserved and must be zero in v0.2.
    """
    ACK_REQUIRED: int = 0b00000001   # bit 0 — gateway must track ACK
    EXPIRES:      int = 0b00000010   # bit 1 — 8-byte expiry follows fixed header
    DELTA:        int = 0b00000100   # bit 2 — payload is a delta update
    ENCRYPTED:    int = 0b00001000   # bit 3 — body is end-to-end encrypted
    PRIORITY:     int = 0b00010000   # bit 4 — deliver before queued non-priority
    RESERVED_MASK: int = 0b11100000  # bits 5-7 — must be zero

# ─── Content type registry (offset 2-3, uint16) ──────────────────────────────

class ContentType:
    """
    Registered OSP content types per spec Section 5.3.
    """
    TEXT_PLAIN       = 0x0001   # UTF-8 plain text — Tier 0 default
    STRUCTURED_JSON  = 0x0002   # JSON, OSP payload schema
    STRUCTURED_MSGPACK = 0x0003 # MessagePack, OSP payload schema — Tier 1-2 preferred
    GRAPH_JSON       = 0x0004   # Full entity relationship graph, JSON — Tier 3
    DELTA_MSGPACK    = 0x0005   # Delta update, MessagePack — requires prior state
    LLM_CONTEXT      = 0x0010   # Structured context fragment for LLM injection — Tier 3
    CUSTOM           = 0x00FF   # Publisher declares schema out of band

    # Human-readable labels for logging
    _LABELS: dict = {
        0x0001: "osp/text-plain",
        0x0002: "osp/structured-json",
        0x0003: "osp/structured-msgpack",
        0x0004: "osp/graph-json",
        0x0005: "osp/delta-msgpack",
        0x0010: "osp/llm-context",
        0x00FF: "osp/custom",
    }

    @classmethod
    def label(cls, code: int) -> str:
        return cls._LABELS.get(code, f"osp/unknown-{code:#06x}")

# ─── Capability tiers ─────────────────────────────────────────────────────────

class Tier:
    """
    Renderer capability tiers per spec Section 4.
    """
    TIER_0 = 0   # Minimal — SMS/cellular, display only, no ACK, max 160 bytes
    TIER_1 = 1   # Low-power — cellular data, simple ACK, e-ink, max 2 KB
    TIER_2 = 2   # Standard — HTTP/HTTPS, full ACK+retry, mobile/desktop, max 64 KB
    TIER_3 = 3   # Rich — HTTP/HTTPS or MQTT, streaming delta, LLM/dashboard/analytics

    MAX_PAYLOAD: dict = {
        0: MAX_PAYLOAD_TIER0,
        1: MAX_PAYLOAD_TIER1,
        2: MAX_PAYLOAD_TIER2,
        3: None,   # no constraint
    }

# ─── Delivery methods ─────────────────────────────────────────────────────────

class DeliveryMethod:
    WEBHOOK = "webhook"
    SMS     = "sms"
    MQTT    = "mqtt"
    PUSH    = "push"
    EMAIL   = "email"

# ─── Standard event types (Appendix A) ───────────────────────────────────────

class EventType:
    STORY_CREATED         = "story.created"
    STORY_UPDATED         = "story.updated"
    STORY_STATE_CHANGED   = "story.state_changed"
    ENTITY_SPIKE          = "entity.spike"
    ENTITY_SENTIMENT_SHIFT = "entity.sentiment_shift"
    TREND_DETECTED        = "trend.detected"
    RELATIONSHIP_NEW      = "relationship.new"
    GRAPH_UPDATED         = "graph.updated"
    CONTEXT_REFRESH       = "context.refresh"
    ALERT_CUSTOM          = "alert.custom"

    # Custom namespace prefix — publishers may extend
    CUSTOM_PREFIX = "custom."

    _STANDARD = {
        "story.created", "story.updated", "story.state_changed",
        "entity.spike", "entity.sentiment_shift", "trend.detected",
        "relationship.new", "graph.updated", "context.refresh",
        "alert.custom",
    }

    @classmethod
    def is_standard(cls, event_type: str) -> bool:
        return event_type in cls._STANDARD

    @classmethod
    def is_valid(cls, event_type: str) -> bool:
        return cls.is_standard(event_type) or event_type.startswith(cls.CUSTOM_PREFIX)

# ─── Narrative lifecycle states ───────────────────────────────────────────────

class NarrativeState:
    EMERGING   = "EMERGING"
    DEVELOPING = "DEVELOPING"
    PEAK       = "PEAK"
    DECLINING  = "DECLINING"
    CONCLUDED  = "CONCLUDED"
    CONTESTED  = "CONTESTED"

    _ALL = {EMERGING, DEVELOPING, PEAK, DECLINING, CONCLUDED, CONTESTED}

    @classmethod
    def is_valid(cls, state: str) -> bool:
        return state in cls._ALL
