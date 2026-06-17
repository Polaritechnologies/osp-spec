"""
osp.frame — OSP v0.2 Frame Layer

Public surface:

    from osp.frame import (
        OSPFrame,
        FrameError, VersionError, ChecksumError, TruncatedError,
        ReservedBitsError, PayloadTooLargeError,
        Flags, ContentType, Tier, DeliveryMethod, EventType, NarrativeState,
        OSP_VERSION, HEADER_SIZE,
    )
"""

from .constants import (
    OSP_VERSION,
    HEADER_SIZE,
    HEADER_STRUCT_FMT,
    EXTENDED_HEADER_EXPIRY_SIZE,
    MAX_PAYLOAD_TIER0,
    MAX_PAYLOAD_TIER1,
    MAX_PAYLOAD_TIER2,
    Flags,
    ContentType,
    Tier,
    DeliveryMethod,
    EventType,
    NarrativeState,
)

from .frame import (
    OSPFrame,
    FrameError,
    VersionError,
    ChecksumError,
    TruncatedError,
    ReservedBitsError,
    PayloadTooLargeError,
    _crc16_ccitt,   # exposed for testing; not part of stable API
)

__all__ = [
    "OSPFrame",
    "FrameError",
    "VersionError",
    "ChecksumError",
    "TruncatedError",
    "ReservedBitsError",
    "PayloadTooLargeError",
    "OSP_VERSION",
    "HEADER_SIZE",
    "HEADER_STRUCT_FMT",
    "EXTENDED_HEADER_EXPIRY_SIZE",
    "MAX_PAYLOAD_TIER0",
    "MAX_PAYLOAD_TIER1",
    "MAX_PAYLOAD_TIER2",
    "Flags",
    "ContentType",
    "Tier",
    "DeliveryMethod",
    "EventType",
    "NarrativeState",
    "_crc16_ccitt",
]
