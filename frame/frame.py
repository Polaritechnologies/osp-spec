"""
OSP Frame Encoder / Decoder
Open Smart Protocol v0.2 — Section 5

Handles the 20-byte fixed frame header and the 8-byte extended header
that follows when the EXPIRES flag is set. Everything above the payload
body lives here; payload schema is handled separately in osp.payload.

Usage
-----
Encoding:
    frame = OSPFrame.build(
        content_type=ContentType.LLM_CONTEXT,
        publisher_id=1,
        sequence=42,
        payload=b'...',
        flags=Flags.ACK_REQUIRED | Flags.PRIORITY,
        expires_at=int(time.time()) + 3600,
    )
    wire_bytes = frame.encode()

Decoding:
    frame, consumed = OSPFrame.decode(wire_bytes)
    # consumed = total bytes read (header + optional extended + payload body)

Errors:
    FrameError         — base class for all frame errors
    VersionError       — version byte does not match OSP_VERSION
    ChecksumError      — CRC-16/CCITT mismatch
    TruncatedError     — buffer shorter than declared payload
    ReservedBitsError  — bits 5-7 of flags byte are non-zero
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from typing import Optional

from .constants import (
    OSP_VERSION,
    HEADER_SIZE,
    HEADER_STRUCT_FMT,
    EXTENDED_HEADER_EXPIRY_SIZE,
    Flags,
    ContentType,
)


# ─── Exceptions ───────────────────────────────────────────────────────────────

class FrameError(Exception):
    """Base class for all OSP frame errors."""

class VersionError(FrameError):
    """Version byte does not match the expected OSP protocol version."""

class ChecksumError(FrameError):
    """CRC-16/CCITT checksum in the frame header does not match computed value."""

class TruncatedError(FrameError):
    """Buffer is shorter than the declared payload_len."""

class ReservedBitsError(FrameError):
    """Bits 5-7 of the flags byte are non-zero (reserved, must be zero in v0.2)."""

class PayloadTooLargeError(FrameError):
    """Payload exceeds the maximum size for the declared tier."""


# ─── CRC-16/CCITT ─────────────────────────────────────────────────────────────

def _crc16_ccitt(data: bytes) -> int:
    """
    Compute CRC-16/CCITT (poly 0x1021, init 0xFFFF) over *data*.

    Per spec Section 5.1: covers header bytes 0-17 (the 18 bytes before the
    2-byte checksum field). This is a transmission error detection mechanism,
    not a cryptographic guarantee.

    Known test vector: crc16_ccitt(b'123456789') == 0x29B1
    """
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc


# ─── OSPFrame dataclass ───────────────────────────────────────────────────────

@dataclass
class OSPFrame:
    """
    Represents a fully-parsed OSP frame: fixed header + optional extended
    header + payload body.

    Attributes
    ----------
    version : int
        Protocol version byte. Always OSP_VERSION (0x02) for frames produced
        by this implementation.
    flags : int
        Flags byte. Combine Flags.* constants with bitwise OR.
    content_type : int
        Payload content type. One of ContentType.* constants.
    publisher_id : int
        Registered publisher identifier (uint32).
    sequence : int
        Monotonic sequence number per publisher (uint32). Callers are
        responsible for incrementing this.
    timestamp : int
        Unix timestamp in seconds, UTC (uint32). Defaults to time.time()
        at construction if not supplied.
    payload_len : int
        Length of payload body in bytes. Derived from len(payload) on
        encode; parsed from wire on decode.
    checksum : int
        CRC-16/CCITT of header bytes 0-17. Computed on encode; verified
        on decode.
    payload : bytes
        The raw payload body bytes.
    expires_at : Optional[int]
        Unix timestamp (uint64) after which this payload must not be
        delivered. Present only when Flags.EXPIRES is set. Stored
        separately from the fixed header; occupies the extended header
        bytes immediately following the fixed header.
    """

    version:      int
    flags:        int
    content_type: int
    publisher_id: int
    sequence:     int
    timestamp:    int
    payload_len:  int
    checksum:     int
    payload:      bytes
    expires_at:   Optional[int] = field(default=None)

    # ── Construction helpers ──────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        *,
        content_type: int,
        publisher_id: int,
        sequence: int,
        payload: bytes,
        flags: int = 0,
        expires_at: Optional[int] = None,
        timestamp: Optional[int] = None,
    ) -> "OSPFrame":
        """
        Construct an OSPFrame ready to encode.

        Sets version to OSP_VERSION. Computes payload_len from len(payload).
        Checksum is computed during encode(), not here — it is set to 0
        in the dataclass until encode() is called.

        If expires_at is provided, automatically sets the EXPIRES flag bit.
        Callers may set additional flags via the flags parameter.
        """
        if expires_at is not None:
            flags |= Flags.EXPIRES

        ts = timestamp if timestamp is not None else int(time.time())

        return cls(
            version=OSP_VERSION,
            flags=flags,
            content_type=content_type,
            publisher_id=publisher_id,
            sequence=sequence,
            timestamp=ts,
            payload_len=len(payload),
            checksum=0,       # computed in encode()
            payload=payload,
            expires_at=expires_at,
        )

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(self) -> bytes:
        """
        Serialize the frame to wire bytes.

        Layout:
            [20-byte fixed header][optional 8-byte expiry][payload body]

        The checksum field (bytes 18-19) is computed over bytes 0-17 of
        the fixed header with the checksum field set to zero during
        computation, then written in.

        Returns the full wire representation as bytes.
        """
        if self.flags & Flags.RESERVED_MASK:
            raise ReservedBitsError(
                f"Flags byte {self.flags:#010b} has reserved bits set "
                f"(bits 5-7 must be zero in OSP v0.2)"
            )

        # Pack the fixed header with checksum = 0 for CRC computation
        header_no_crc = struct.pack(
            HEADER_STRUCT_FMT,
            self.version,
            self.flags,
            self.content_type,
            self.publisher_id,
            self.sequence,
            self.timestamp,
            self.payload_len,
            0,   # checksum placeholder
        )

        # CRC covers bytes 0-17 (the 18 bytes before the 2-byte checksum)
        crc = _crc16_ccitt(header_no_crc[:18])

        # Repack with the computed checksum
        header = struct.pack(
            HEADER_STRUCT_FMT,
            self.version,
            self.flags,
            self.content_type,
            self.publisher_id,
            self.sequence,
            self.timestamp,
            self.payload_len,
            crc,
        )

        # Store the computed checksum on self for inspection
        self.checksum = crc

        # Extended header: 8-byte expiry timestamp (uint64, big-endian)
        extended = b""
        if self.flags & Flags.EXPIRES:
            if self.expires_at is None:
                raise FrameError("EXPIRES flag is set but expires_at is None")
            extended = struct.pack("!Q", self.expires_at)

        return header + extended + self.payload

    # ── Decoding ──────────────────────────────────────────────────────────────

    @classmethod
    def decode(cls, data: bytes, *, verify_checksum: bool = True) -> tuple["OSPFrame", int]:
        """
        Parse an OSP frame from a bytes buffer.

        Parameters
        ----------
        data : bytes
            Raw wire bytes. May contain more than one frame; only the
            first frame is parsed. Use the returned consumed count to
            advance the buffer.
        verify_checksum : bool
            If True (default), raise ChecksumError on CRC mismatch.
            Set False only for testing / fuzzing.

        Returns
        -------
        (frame, consumed) where consumed is the total number of bytes
        consumed from data: HEADER_SIZE + optional 8-byte expiry + payload_len.

        Raises
        ------
        VersionError       — version != OSP_VERSION
        ChecksumError      — CRC mismatch (when verify_checksum=True)
        TruncatedError     — buffer too short
        ReservedBitsError  — bits 5-7 of flags are non-zero
        """
        if len(data) < HEADER_SIZE:
            raise TruncatedError(
                f"Buffer has {len(data)} bytes; minimum frame size is {HEADER_SIZE}"
            )

        (
            version,
            flags,
            content_type,
            publisher_id,
            sequence,
            timestamp,
            payload_len,
            checksum,
        ) = struct.unpack_from(HEADER_STRUCT_FMT, data, 0)

        # Version check
        if version != OSP_VERSION:
            raise VersionError(
                f"Unsupported OSP version {version:#04x}; "
                f"expected {OSP_VERSION:#04x}"
            )

        # Reserved bits check
        if flags & Flags.RESERVED_MASK:
            raise ReservedBitsError(
                f"Flags byte {flags:#010b} has reserved bits set "
                f"(bits 5-7 must be zero in OSP v0.2)"
            )

        # Checksum verification: CRC over bytes 0-17 with checksum field zeroed
        if verify_checksum:
            header_for_crc = bytearray(data[:18]) + b"\x00\x00"
            expected_crc = _crc16_ccitt(bytes(header_for_crc[:18]))
            if checksum != expected_crc:
                raise ChecksumError(
                    f"CRC mismatch: header has {checksum:#06x}, "
                    f"computed {expected_crc:#06x}"
                )

        # Advance cursor past fixed header
        cursor = HEADER_SIZE

        # Extended header: 8-byte expiry when EXPIRES flag is set
        expires_at: Optional[int] = None
        if flags & Flags.EXPIRES:
            if len(data) < cursor + EXTENDED_HEADER_EXPIRY_SIZE:
                raise TruncatedError(
                    f"EXPIRES flag set but buffer too short for extended header "
                    f"(need {cursor + EXTENDED_HEADER_EXPIRY_SIZE} bytes, "
                    f"have {len(data)})"
                )
            (expires_at,) = struct.unpack_from("!Q", data, cursor)
            cursor += EXTENDED_HEADER_EXPIRY_SIZE

        # Payload body
        if len(data) < cursor + payload_len:
            raise TruncatedError(
                f"Declared payload_len={payload_len} but only "
                f"{len(data) - cursor} bytes remain in buffer"
            )

        payload = bytes(data[cursor : cursor + payload_len])
        cursor += payload_len

        frame = cls(
            version=version,
            flags=flags,
            content_type=content_type,
            publisher_id=publisher_id,
            sequence=sequence,
            timestamp=timestamp,
            payload_len=payload_len,
            checksum=checksum,
            payload=payload,
            expires_at=expires_at,
        )

        return frame, cursor

    # ── Flag inspection helpers ───────────────────────────────────────────────

    @property
    def ack_required(self) -> bool:
        return bool(self.flags & Flags.ACK_REQUIRED)

    @property
    def is_delta(self) -> bool:
        return bool(self.flags & Flags.DELTA)

    @property
    def is_encrypted(self) -> bool:
        return bool(self.flags & Flags.ENCRYPTED)

    @property
    def is_priority(self) -> bool:
        return bool(self.flags & Flags.PRIORITY)

    @property
    def has_expiry(self) -> bool:
        return bool(self.flags & Flags.EXPIRES)

    def is_expired(self, now: Optional[int] = None) -> bool:
        """
        Return True if this payload has passed its expiry timestamp.

        Per spec Section 7.3: a renderer that receives a payload after its
        expiry timestamp should discard it. This helper can be called at
        the gateway (before delivery) or at the renderer (after receipt).
        """
        if not self.has_expiry or self.expires_at is None:
            return False
        t = now if now is not None else int(time.time())
        return t >= self.expires_at

    # ── Content type label ────────────────────────────────────────────────────

    @property
    def content_type_label(self) -> str:
        return ContentType.label(self.content_type)

    # ── Debug repr ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        flags_set = []
        if self.ack_required:  flags_set.append("ACK_REQUIRED")
        if self.has_expiry:    flags_set.append("EXPIRES")
        if self.is_delta:      flags_set.append("DELTA")
        if self.is_encrypted:  flags_set.append("ENCRYPTED")
        if self.is_priority:   flags_set.append("PRIORITY")
        flags_str = "|".join(flags_set) if flags_set else "none"

        return (
            f"OSPFrame("
            f"v={self.version:#04x}, "
            f"flags=[{flags_str}], "
            f"content_type={self.content_type_label}, "
            f"publisher_id={self.publisher_id}, "
            f"seq={self.sequence}, "
            f"ts={self.timestamp}, "
            f"payload_len={self.payload_len}, "
            f"crc={self.checksum:#06x}"
            + (f", expires_at={self.expires_at}" if self.has_expiry else "")
            + ")"
        )
