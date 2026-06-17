"""
Tests for osp.frame

Covers:
    - CRC-16/CCITT known test vector
    - Encode / decode round-trip (minimal frame)
    - Every individual flag bit: ACK_REQUIRED, EXPIRES, DELTA, ENCRYPTED, PRIORITY
    - Combined flags
    - EXPIRES extended header encode/decode
    - Expiry detection (is_expired)
    - Reserved bits rejection on both encode and decode
    - Version mismatch rejection on decode
    - Checksum mismatch rejection on decode
    - Truncated buffer errors (header, extended header, payload)
    - All ContentType codes round-trip
    - Multi-frame buffer (decode only reads first frame, returns correct consumed)
    - build() helper: auto-sets EXPIRES flag, auto-fills timestamp
    - Empty payload
    - Maximum uint32 publisher_id and sequence values
    - verify_checksum=False bypass
    - Property helpers: ack_required, is_delta, is_encrypted, is_priority, has_expiry, is_expired
"""

import sys
import os
import time
import struct

# Allow running from repo root without install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from frame import (
    OSPFrame,
    FrameError,
    VersionError,
    ChecksumError,
    TruncatedError,
    ReservedBitsError,
    Flags,
    ContentType,
    OSP_VERSION,
    HEADER_SIZE,
    _crc16_ccitt,
)


# ─── CRC ──────────────────────────────────────────────────────────────────────

class TestCRC:
    def test_known_vector(self):
        """CRC-16/CCITT of b'123456789' must be 0x29B1 per standard."""
        assert _crc16_ccitt(b"123456789") == 0x29B1

    def test_empty(self):
        """CRC of empty bytes is the init value 0xFFFF."""
        assert _crc16_ccitt(b"") == 0xFFFF

    def test_single_zero_byte(self):
        result = _crc16_ccitt(b"\x00")
        assert isinstance(result, int)
        assert 0 <= result <= 0xFFFF

    def test_deterministic(self):
        data = b"polari-osp-frame-test"
        assert _crc16_ccitt(data) == _crc16_ccitt(data)

    def test_different_data_different_crc(self):
        assert _crc16_ccitt(b"abc") != _crc16_ccitt(b"abd")


# ─── Round-trip helpers ───────────────────────────────────────────────────────

def make_frame(
    content_type=ContentType.STRUCTURED_JSON,
    publisher_id=1,
    sequence=1,
    payload=b'{"test": true}',
    flags=0,
    expires_at=None,
    timestamp=1750000000,
) -> OSPFrame:
    return OSPFrame.build(
        content_type=content_type,
        publisher_id=publisher_id,
        sequence=sequence,
        payload=payload,
        flags=flags,
        expires_at=expires_at,
        timestamp=timestamp,
    )


def roundtrip(frame: OSPFrame) -> OSPFrame:
    wire = frame.encode()
    decoded, consumed = OSPFrame.decode(wire)
    assert consumed == len(wire), f"consumed {consumed} != wire len {len(wire)}"
    return decoded


# ─── Basic encode / decode ────────────────────────────────────────────────────

class TestEncodeDecodeBasic:
    def test_header_size(self):
        f = make_frame()
        wire = f.encode()
        assert len(wire) == HEADER_SIZE + len(b'{"test": true}')

    def test_version_preserved(self):
        assert roundtrip(make_frame()).version == OSP_VERSION

    def test_flags_preserved(self):
        assert roundtrip(make_frame(flags=0)).flags == 0

    def test_content_type_preserved(self):
        d = roundtrip(make_frame(content_type=ContentType.LLM_CONTEXT))
        assert d.content_type == ContentType.LLM_CONTEXT

    def test_publisher_id_preserved(self):
        d = roundtrip(make_frame(publisher_id=999))
        assert d.publisher_id == 999

    def test_sequence_preserved(self):
        d = roundtrip(make_frame(sequence=65535))
        assert d.sequence == 65535

    def test_timestamp_preserved(self):
        d = roundtrip(make_frame(timestamp=1750000000))
        assert d.timestamp == 1750000000

    def test_payload_preserved(self):
        payload = b'{"headline": "test story", "confidence": 0.92}'
        d = roundtrip(make_frame(payload=payload))
        assert d.payload == payload

    def test_payload_len_matches(self):
        payload = b"hello world"
        d = roundtrip(make_frame(payload=payload))
        assert d.payload_len == len(payload)

    def test_checksum_is_computed(self):
        f = make_frame()
        f.encode()
        assert f.checksum != 0   # not the placeholder zero

    def test_empty_payload(self):
        d = roundtrip(make_frame(payload=b""))
        assert d.payload == b""
        assert d.payload_len == 0

    def test_binary_payload(self):
        payload = bytes(range(256))
        d = roundtrip(make_frame(payload=payload))
        assert d.payload == payload

    def test_max_uint32_publisher_id(self):
        d = roundtrip(make_frame(publisher_id=0xFFFFFFFF))
        assert d.publisher_id == 0xFFFFFFFF

    def test_max_uint32_sequence(self):
        d = roundtrip(make_frame(sequence=0xFFFFFFFF))
        assert d.sequence == 0xFFFFFFFF


# ─── Flag bits ────────────────────────────────────────────────────────────────

class TestFlags:
    def test_no_flags_default(self):
        f = make_frame(flags=0)
        d = roundtrip(f)
        assert not d.ack_required
        assert not d.is_delta
        assert not d.is_encrypted
        assert not d.is_priority
        assert not d.has_expiry

    def test_ack_required_flag(self):
        d = roundtrip(make_frame(flags=Flags.ACK_REQUIRED))
        assert d.ack_required
        assert not d.is_delta

    def test_delta_flag(self):
        d = roundtrip(make_frame(flags=Flags.DELTA))
        assert d.is_delta
        assert not d.ack_required

    def test_encrypted_flag(self):
        d = roundtrip(make_frame(flags=Flags.ENCRYPTED))
        assert d.is_encrypted

    def test_priority_flag(self):
        d = roundtrip(make_frame(flags=Flags.PRIORITY))
        assert d.is_priority

    def test_combined_flags(self):
        combined = Flags.ACK_REQUIRED | Flags.PRIORITY | Flags.DELTA
        d = roundtrip(make_frame(flags=combined))
        assert d.ack_required
        assert d.is_priority
        assert d.is_delta
        assert not d.is_encrypted
        assert not d.has_expiry

    def test_all_non_reserved_flags_combined(self):
        all_flags = (
            Flags.ACK_REQUIRED | Flags.EXPIRES | Flags.DELTA |
            Flags.ENCRYPTED | Flags.PRIORITY
        )
        future_ts = int(time.time()) + 3600
        f = OSPFrame.build(
            content_type=ContentType.STRUCTURED_JSON,
            publisher_id=1,
            sequence=1,
            payload=b"test",
            flags=all_flags,
            expires_at=future_ts,
            timestamp=1750000000,
        )
        d = roundtrip(f)
        assert d.ack_required
        assert d.has_expiry
        assert d.is_delta
        assert d.is_encrypted
        assert d.is_priority


# ─── EXPIRES extended header ──────────────────────────────────────────────────

class TestExpiresFlag:
    def _future(self) -> int:
        return int(time.time()) + 3600

    def test_expires_auto_sets_flag(self):
        f = make_frame(expires_at=self._future())
        assert f.flags & Flags.EXPIRES

    def test_expires_at_preserved(self):
        ts = self._future()
        d = roundtrip(make_frame(expires_at=ts))
        assert d.expires_at == ts

    def test_expires_wire_length(self):
        ts = self._future()
        f = make_frame(payload=b"body", expires_at=ts)
        wire = f.encode()
        # HEADER_SIZE + 8-byte extended + payload
        assert len(wire) == HEADER_SIZE + 8 + 4

    def test_expires_uint64_max(self):
        """expires_at is stored as uint64 — verify no truncation."""
        max_u64 = (2 ** 64) - 1
        d = roundtrip(make_frame(expires_at=max_u64))
        assert d.expires_at == max_u64

    def test_no_expires_at_without_flag(self):
        d = roundtrip(make_frame(flags=0))
        assert d.expires_at is None
        assert not d.has_expiry

    def test_expires_flag_raises_if_expires_at_none(self):
        """EXPIRES flag set manually without expires_at should raise on encode."""
        f = make_frame(flags=Flags.EXPIRES)
        f.expires_at = None   # force mismatch
        with pytest.raises(FrameError):
            f.encode()


# ─── is_expired helper ────────────────────────────────────────────────────────

class TestIsExpired:
    def test_not_expired_when_no_expiry(self):
        f = make_frame()
        assert not f.is_expired()

    def test_not_expired_when_in_future(self):
        f = make_frame(expires_at=int(time.time()) + 3600)
        assert not f.is_expired()

    def test_expired_when_in_past(self):
        f = make_frame(expires_at=int(time.time()) - 1)
        assert f.is_expired()

    def test_expired_at_exact_now(self):
        now = int(time.time())
        f = make_frame(expires_at=now)
        assert f.is_expired(now=now)   # >= boundary is expired

    def test_custom_now(self):
        f = make_frame(expires_at=1000)
        assert f.is_expired(now=1001)
        assert not f.is_expired(now=999)


# ─── Error cases ──────────────────────────────────────────────────────────────

class TestErrors:
    def test_version_mismatch_raises(self):
        f = make_frame()
        wire = bytearray(f.encode())
        wire[0] = 0x01   # wrong version
        # Checksum will also be wrong; version check fires first
        with pytest.raises(VersionError):
            OSPFrame.decode(bytes(wire), verify_checksum=False)

    def test_checksum_mismatch_raises(self):
        f = make_frame()
        wire = bytearray(f.encode())
        wire[18] ^= 0xFF   # corrupt checksum byte
        with pytest.raises(ChecksumError):
            OSPFrame.decode(bytes(wire))

    def test_checksum_bypass(self):
        f = make_frame()
        wire = bytearray(f.encode())
        wire[18] ^= 0xFF   # corrupt checksum byte
        # Should not raise when verify_checksum=False
        d, _ = OSPFrame.decode(bytes(wire), verify_checksum=False)
        assert d.publisher_id == f.publisher_id

    def test_reserved_bits_encode_raises(self):
        f = make_frame(flags=0b00100000)   # bit 5 set
        with pytest.raises(ReservedBitsError):
            f.encode()

    def test_reserved_bits_decode_raises(self):
        f = make_frame()
        wire = bytearray(f.encode())
        wire[1] |= 0b00100000   # set bit 5
        # Recompute checksum so version/checksum don't fire first
        crc = _crc16_ccitt(bytes(wire[:18]))
        wire[18] = (crc >> 8) & 0xFF
        wire[19] = crc & 0xFF
        with pytest.raises(ReservedBitsError):
            OSPFrame.decode(bytes(wire))

    def test_truncated_buffer_header(self):
        with pytest.raises(TruncatedError):
            OSPFrame.decode(b"\x02" * 10)   # less than HEADER_SIZE bytes

    def test_truncated_buffer_payload(self):
        f = make_frame(payload=b"long payload here")
        wire = f.encode()
        # Chop off the last few payload bytes
        with pytest.raises(TruncatedError):
            OSPFrame.decode(wire[:-5])

    def test_truncated_extended_header(self):
        ts = int(time.time()) + 3600
        f = make_frame(payload=b"x", expires_at=ts)
        wire = f.encode()
        # Chop inside the extended header (after HEADER_SIZE, before +8)
        truncated = wire[:HEADER_SIZE + 4]
        with pytest.raises(TruncatedError):
            OSPFrame.decode(truncated)

    def test_empty_buffer(self):
        with pytest.raises(TruncatedError):
            OSPFrame.decode(b"")


# ─── Content types ────────────────────────────────────────────────────────────

class TestContentTypes:
    @pytest.mark.parametrize("ct", [
        ContentType.TEXT_PLAIN,
        ContentType.STRUCTURED_JSON,
        ContentType.STRUCTURED_MSGPACK,
        ContentType.GRAPH_JSON,
        ContentType.DELTA_MSGPACK,
        ContentType.LLM_CONTEXT,
        ContentType.CUSTOM,
    ])
    def test_content_type_roundtrip(self, ct):
        d = roundtrip(make_frame(content_type=ct))
        assert d.content_type == ct

    def test_content_type_label_known(self):
        assert ContentType.label(ContentType.LLM_CONTEXT) == "osp/llm-context"
        assert ContentType.label(ContentType.STRUCTURED_JSON) == "osp/structured-json"

    def test_content_type_label_unknown(self):
        label = ContentType.label(0xDEAD)
        assert "unknown" in label
        assert "dead" in label.lower()


# ─── Multi-frame buffer ───────────────────────────────────────────────────────

class TestMultiFrameBuffer:
    def test_consumed_stops_at_first_frame(self):
        f1 = make_frame(payload=b"frame one", sequence=1)
        f2 = make_frame(payload=b"frame two", sequence=2)
        buf = f1.encode() + f2.encode()
        d1, consumed = OSPFrame.decode(buf)
        assert consumed == len(f1.encode())
        assert d1.payload == b"frame one"

    def test_second_frame_decodable_from_remainder(self):
        f1 = make_frame(payload=b"aaa", sequence=1)
        f2 = make_frame(payload=b"bbb", sequence=2)
        buf = f1.encode() + f2.encode()
        _, c1 = OSPFrame.decode(buf)
        d2, _ = OSPFrame.decode(buf[c1:])
        assert d2.payload == b"bbb"


# ─── build() helper ───────────────────────────────────────────────────────────

class TestBuildHelper:
    def test_auto_timestamp(self):
        before = int(time.time())
        f = OSPFrame.build(
            content_type=ContentType.STRUCTURED_JSON,
            publisher_id=1,
            sequence=1,
            payload=b"ts test",
        )
        after = int(time.time())
        assert before <= f.timestamp <= after

    def test_explicit_timestamp(self):
        f = OSPFrame.build(
            content_type=ContentType.STRUCTURED_JSON,
            publisher_id=1,
            sequence=1,
            payload=b"ts test",
            timestamp=1700000000,
        )
        assert f.timestamp == 1700000000

    def test_initial_checksum_is_zero(self):
        f = OSPFrame.build(
            content_type=ContentType.STRUCTURED_JSON,
            publisher_id=1,
            sequence=1,
            payload=b"x",
        )
        assert f.checksum == 0   # not computed until encode()

    def test_checksum_set_after_encode(self):
        f = OSPFrame.build(
            content_type=ContentType.STRUCTURED_JSON,
            publisher_id=1,
            sequence=1,
            payload=b"x",
        )
        f.encode()
        assert f.checksum != 0

    def test_version_always_osp_version(self):
        f = OSPFrame.build(
            content_type=ContentType.STRUCTURED_JSON,
            publisher_id=1,
            sequence=1,
            payload=b"x",
        )
        assert f.version == OSP_VERSION


# ─── repr ─────────────────────────────────────────────────────────────────────

class TestRepr:
    def test_repr_contains_content_type_label(self):
        f = make_frame(content_type=ContentType.LLM_CONTEXT)
        assert "osp/llm-context" in repr(f)

    def test_repr_contains_flags(self):
        f = make_frame(flags=Flags.ACK_REQUIRED | Flags.PRIORITY)
        r = repr(f)
        assert "ACK_REQUIRED" in r
        assert "PRIORITY" in r

    def test_repr_no_flags_shows_none(self):
        f = make_frame(flags=0)
        assert "none" in repr(f)

    def test_repr_contains_expires_at_when_set(self):
        ts = int(time.time()) + 3600
        f = make_frame(expires_at=ts)
        assert str(ts) in repr(f)
