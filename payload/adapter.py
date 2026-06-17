"""
OSP Tier Adaptation
Open Smart Protocol v0.2 — Section 11

The gateway is responsible for adapting payloads to the declared tier of
each renderer. Adaptation is lossy by design: a Tier 0 payload is not a
full-fidelity representation of the intelligence.

Adaptation rules per spec Table (Section 11):

    Field                   Tier 0          Tier 1-2-3
    ─────────────────────── ─────────────── ──────────────────────────
    headline                Truncated 160   Full, up to 512 chars
    entities                First name only Full list with sentiment
    confidence              Omitted         Included
    narrative_state         Omitted         Included (Tier 2+)
    relationships           Omitted         Full graph (Tier 3 only)
    counter_narrative       Omitted         Full (Tier 2+ when CONTESTED)
    provenance              Omitted         Full (Tier 3 / LLM context)
    key_facts               Omitted         Full (LLM context type only)

Tier downgrade is always permitted; tier upgrade requires capability
declaration (enforced by renderer registration, not here).

Usage
-----
    from osp.payload.adapter import OSPAdapter
    from osp.frame import ContentType, Tier

    raw_dict = payload.model_dump(exclude_none=True)
    adapted  = OSPAdapter.adapt(raw_dict, renderer_tier=Tier.TIER_1)
    wire_json = json.dumps(adapted).encode()
"""

from __future__ import annotations

import json
from typing import Any

from frame.constants import Tier, ContentType


class OSPAdapter:
    """
    Adapts a fully-populated OSP payload dict to the profile appropriate
    for a given renderer tier.

    Input is always a dict produced by model.model_dump(exclude_none=True).
    Output is a dict safe to serialize and deliver to the renderer.
    """

    @classmethod
    def adapt(
        cls,
        payload: dict[str, Any],
        renderer_tier: int,
        content_type: int = ContentType.STRUCTURED_JSON,
    ) -> dict[str, Any]:
        """
        Return a new dict adapted to renderer_tier.

        Parameters
        ----------
        payload : dict
            Full payload dict from OSPCorePayload (or subclass) .model_dump().
        renderer_tier : int
            One of Tier.TIER_0 through Tier.TIER_3.
        content_type : int
            The content type being delivered. Affects which fields are
            included (e.g. key_facts only for LLM_CONTEXT).

        Returns
        -------
        dict — adapted payload ready for JSON serialization and delivery.
        """
        if renderer_tier == Tier.TIER_0:
            return cls._adapt_tier0(payload)
        elif renderer_tier == Tier.TIER_1:
            return cls._adapt_tier1(payload)
        elif renderer_tier == Tier.TIER_2:
            return cls._adapt_tier2(payload, content_type)
        else:  # TIER_3
            return cls._adapt_tier3(payload, content_type)

    @classmethod
    def adapt_to_bytes(
        cls,
        payload: dict[str, Any],
        renderer_tier: int,
        content_type: int = ContentType.STRUCTURED_JSON,
    ) -> bytes:
        """
        Adapt and serialize to UTF-8 JSON bytes.

        For Tier 0 the payload is plain text (just the truncated headline),
        not JSON — returns bytes of the headline string directly.
        """
        if renderer_tier == Tier.TIER_0:
            adapted = cls._adapt_tier0(payload)
            # Tier 0 wire format is plain text: just the headline
            return adapted["headline"].encode("utf-8")

        adapted = cls.adapt(payload, renderer_tier, content_type)
        return json.dumps(adapted, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    # ── Tier 0: ultra-compact, plain text, max 160 bytes ──────────────────────

    @classmethod
    def _adapt_tier0(cls, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Tier 0 profile per spec:
        - headline truncated to 160 chars
        - entities: first entity name only (omit all other entity fields)
        - confidence: omitted
        - narrative_state: omitted
        - relationships: omitted
        - counter_narrative: omitted
        - provenance: omitted
        - key_facts: omitted
        """
        headline = payload.get("headline", "")[:160]

        entities = payload.get("entities", [])
        first_entity_name = entities[0]["name"] if entities else None

        adapted: dict[str, Any] = {
            "event_type":   payload["event_type"],
            "publisher_id": payload["publisher_id"],
            "emitted_at":   payload["emitted_at"],
            "headline":     headline,
        }
        if first_entity_name:
            adapted["entity"] = first_entity_name   # single name string, not list

        return adapted

    # ── Tier 1: compact, structured text, up to 5 entities, max 2 KB ─────────

    @classmethod
    def _adapt_tier1(cls, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Tier 1 profile per spec:
        - headline: full (up to 512 chars, spec says full for Tier 1-2-3)
        - entities: full list with sentiment, capped at 5
        - confidence: included
        - narrative_state: omitted (Tier 2+ per spec table)
        - relationships: omitted
        - counter_narrative: omitted
        - provenance: omitted
        - key_facts: omitted
        """
        adapted: dict[str, Any] = {
            "event_type":   payload["event_type"],
            "publisher_id": payload["publisher_id"],
            "emitted_at":   payload["emitted_at"],
            "headline":     payload.get("headline", ""),
            "confidence":   payload["confidence"],
        }

        entities = payload.get("entities", [])[:5]
        if entities:
            adapted["entities"] = [
                {"id": e["id"], "name": e["name"], "type": e["type"], "sentiment": e["sentiment"]}
                for e in entities
            ]

        return adapted

    # ── Tier 2: full payload, entity graph, sentiment, narrative state ────────

    @classmethod
    def _adapt_tier2(
        cls, payload: dict[str, Any], content_type: int
    ) -> dict[str, Any]:
        """
        Tier 2 profile per spec:
        - All core + extended fields included
        - narrative_state: included
        - counter_narrative: included when CONTESTED
        - relationships: omitted (Tier 3 only)
        - provenance: omitted
        - key_facts: omitted
        """
        # Start with all core + extended fields, drop Tier 3-only fields
        tier3_only = {"relationships", "narrative_threads", "provenance", "key_facts",
                      "grounding_statement", "valid_until"}

        adapted = {k: v for k, v in payload.items() if k not in tier3_only}
        return adapted

    # ── Tier 3: extended payload, full graph, relationship network ────────────

    @classmethod
    def _adapt_tier3(
        cls, payload: dict[str, Any], content_type: int
    ) -> dict[str, Any]:
        """
        Tier 3 profile per spec:
        - All fields included
        - For non-LLM_CONTEXT content types: omit provenance, key_facts,
          grounding_statement, valid_until (those are LLM context only)
        """
        if content_type == ContentType.LLM_CONTEXT:
            # Full payload including LLM context fields
            return dict(payload)

        # graph-json and other Tier 3 types: drop LLM-context-only fields
        llm_only = {"grounding_statement", "key_facts", "open_questions",
                    "provenance", "valid_until"}
        return {k: v for k, v in payload.items() if k not in llm_only}

    # ── Payload size check ────────────────────────────────────────────────────

    @classmethod
    def check_size(
        cls,
        adapted_bytes: bytes,
        renderer_tier: int,
    ) -> tuple[bool, int, int | None]:
        """
        Check whether adapted bytes fit within the tier's size limit.

        Returns (fits, actual_size, limit) where limit is None for Tier 3
        (no constraint).
        """
        from frame.constants import Tier as T
        limit = T.MAX_PAYLOAD.get(renderer_tier)
        size = len(adapted_bytes)
        fits = (limit is None) or (size <= limit)
        return fits, size, limit
