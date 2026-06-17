**Version:** 0.2  
**License:** CC BY 4.0  
**Publisher:** [Polari Technologies LLC](https://polaritechnologies.com)  
**Spec site:** [opensmartprotocol.org](https://opensmartprotocol.org)

---

OSP is a transport-agnostic protocol for delivering structured intelligence payloads from publishers to renderers. It defines a compact binary frame header, a four-tier payload schema, and a gateway delivery model with acknowledgment and retry semantics.

## Repository contents

- `frame/` — 20-byte binary frame specification (CRC-16/CCITT, flags, content type registry)
- `payload/` — Four-tier Pydantic v2 payload schemas and tier adaptation logic
- `tests/` — 166 tests covering frame encode/decode and payload validation

## Tier model

| Tier | Profile | Max payload | Target device |
|------|---------|-------------|---------------|
| 0 | Plain text, headline only | 512 B | SMS, e-ink, pager |
| 1 | Structured, top entities | 2 KB | Low-bandwidth cellular |
| 2 | Full payload, sentiment, narrative state | 20 KB | Mobile, web |
| 3 | Graph, relationships, LLM context | Unlimited | Desktop, server |

## Frame header (20 bytes)
Offset  Len  Field

0       1    version        Protocol version. Current: 0x02

1       1    flags          Control bits (ACK_REQUIRED, EXPIRES, DELTA, ENCRYPTED, PRIORITY)

2       2    content_type   Payload encoding (see ContentType registry)

4       4    publisher_id   Registered publisher identifier

8       4    sequence       Monotonic sequence number per publisher

12      4    timestamp      Unix timestamp (seconds, UTC)

16      2    payload_len    Payload body length in bytes

18      2    checksum       CRC-16/CCITT of header bytes 0-17
Total header: 20 bytes

## Quick start

```python
from frame.frame import OSPFrame
from frame.constants import ContentType, Flags

payload = b'{"headline": "Example intelligence payload"}'

frame = OSPFrame.build(
    content_type=ContentType.STRUCTURED_JSON,
    publisher_id=1,
    sequence=1,
    payload=payload,
    flags=Flags.ACK_REQUIRED,
)

wire_bytes = frame.encode()
decoded, consumed = OSPFrame.decode(wire_bytes)
```

## Running tests

```bash
pip install pydantic
pytest tests/ -v
```

## Specification

Full protocol specification, whitepaper, and gateway reference architecture at [opensmartprotocol.org](https://opensmartprotocol.org).

## License

[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

You are free to implement, extend, and build on this specification. Attribution required.
