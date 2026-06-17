"""
Tests for osp.payload — schemas and tier adaptation

Covers:
    OSPEntity
        - valid types accepted
        - invalid type rejected
        - sentiment clamping enforced
    OSPSentimentDistribution
        - valid distribution accepted
        - sum != 1.0 rejected
        - tolerance boundary (±0.01)
    OSPRelationship
        - valid relationship accepted
        - invalid type rejected
        - source == target rejected
    OSPNarrativeThread
        - valid thread accepted
        - velocity out of range rejected
    OSPCounterNarrative
        - valid counter-narrative accepted
    OSPProvenance
        - valid provenance accepted
    OSPCorePayload
        - valid core payload accepted
        - invalid event_type rejected
        - custom.* event type accepted
        - confidence out of range rejected
        - extra fields rejected
    OSPExtendedPayload
        - all extended fields optional
        - invalid narrative_state rejected
        - valid narrative states accepted
        - ISO country code validation
        - inheritance: all core fields present
    OSPGraphPayload
        - relationships and threads accepted
        - counter_narrative present when contested (soft check)
        - CONTESTED without counter_narrative sets flag but doesn't raise
    OSPLLMContextPayload
        - all LLM fields required
        - valid_until must be after emitted_at
        - open_questions present
    OSPAdapter
        - Tier 0: headline truncated to 160, confidence omitted, first entity only
        - Tier 1: full entities up to 5, confidence included, narrative_state omitted
        - Tier 2: narrative_state included, relationships omitted
        - Tier 3 graph-json: relationships included, LLM fields omitted
        - Tier 3 llm-context: all fields included
        - adapt_to_bytes Tier 0 returns plain text
        - adapt_to_bytes Tier 2+ returns JSON bytes
        - check_size: within limit, over limit, Tier 3 no limit
        - downgrade: Tier 3 payload adapted to Tier 1 produces Tier 1 profile
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pytest
from pydantic import ValidationError

from payload.schemas import (
    OSPEntity,
    OSPSentimentDistribution,
    OSPRelationship,
    OSPNarrativeThread,
    OSPCounterNarrative,
    OSPProvenance,
    OSPCorePayload,
    OSPExtendedPayload,
    OSPGraphPayload,
    OSPLLMContextPayload,
)
from payload.adapter import OSPAdapter
from frame.constants import Tier, ContentType


# ─── Fixtures ─────────────────────────────────────────────────────────────────

NOW = 1750000000
FUTURE = NOW + 3600


def entity(
    id="ent_001", name="Polari Technologies", type="ORG", sentiment=0.5
) -> dict:
    return {"id": id, "name": name, "type": type, "sentiment": sentiment}


def core_payload(**overrides) -> dict:
    base = {
        "event_type":   "story.created",
        "publisher_id": 1,
        "emitted_at":   NOW,
        "headline":     "Test headline",
        "confidence":   0.9,
        "entities":     [entity()],
    }
    base.update(overrides)
    return base


def extended_payload(**overrides) -> dict:
    base = core_payload()
    base.update({
        "narrative_state":        "DEVELOPING",
        "sentiment_distribution": {"positive": 0.5, "neutral": 0.3, "negative": 0.2},
        "source_count":           5,
        "article_count":          12,
        "geographic_focus":       ["US", "GB"],
        "topics":                 ["technology", "regulation"],
        "cluster_id":             "clus_abc123",
    })
    base.update(overrides)
    return base


def graph_payload(**overrides) -> dict:
    base = extended_payload()
    base.update({
        "relationships": [{
            "source_entity_id": "ent_001",
            "target_entity_id": "ent_002",
            "type":             "COLLABORATIVE",
            "strength":         0.8,
            "context":          ["policy"],
        }],
        "narrative_threads": [{
            "id":         "thread_001",
            "label":      "Regulatory response",
            "entity_ids": ["ent_001"],
            "velocity":   0.6,
        }],
    })
    base.update(overrides)
    return base


def llm_payload(**overrides) -> dict:
    base = graph_payload()
    base.update({
        "grounding_statement": "Polari Technologies announced a new product in June 2026.",
        "key_facts":           ["Polari raised Series A in 2026."],
        "open_questions":      ["Valuation not confirmed."],
        "provenance": [{
            "source":            "TechCrunch",
            "url":               "https://techcrunch.com/example",
            "published_at":      NOW - 300,
            "reliability_score": 0.85,
        }],
        "valid_until": FUTURE,
    })
    base.update(overrides)
    return base


# ─── OSPEntity ────────────────────────────────────────────────────────────────

class TestOSPEntity:
    def test_valid_org(self):
        e = OSPEntity(**entity())
        assert e.name == "Polari Technologies"
        assert e.type == "ORG"

    @pytest.mark.parametrize("t", ["PERSON", "ORG", "GPE", "LOC", "EVENT", "CONCEPT"])
    def test_all_valid_types(self, t):
        OSPEntity(**entity(type=t))

    def test_invalid_type_rejected(self):
        with pytest.raises(ValidationError, match="entity type"):
            OSPEntity(**entity(type="COMPANY"))

    def test_sentiment_positive_boundary(self):
        OSPEntity(**entity(sentiment=1.0))

    def test_sentiment_negative_boundary(self):
        OSPEntity(**entity(sentiment=-1.0))

    def test_sentiment_over_range(self):
        with pytest.raises(ValidationError):
            OSPEntity(**entity(sentiment=1.1))

    def test_sentiment_under_range(self):
        with pytest.raises(ValidationError):
            OSPEntity(**entity(sentiment=-1.1))

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            OSPEntity(**entity(), extra_field="bad")


# ─── OSPSentimentDistribution ─────────────────────────────────────────────────

class TestOSPSentimentDistribution:
    def test_valid_distribution(self):
        d = OSPSentimentDistribution(positive=0.5, neutral=0.3, negative=0.2)
        assert d.positive == 0.5

    def test_sums_to_exactly_one(self):
        OSPSentimentDistribution(positive=1/3, neutral=1/3, negative=1/3)

    def test_tolerance_boundary_pass(self):
        # 0.005 + 0.005 = 0.01 tolerance: total = 1.005 should pass
        OSPSentimentDistribution(positive=0.995, neutral=0.005, negative=0.005)

    def test_sum_over_tolerance_fails(self):
        with pytest.raises(ValidationError, match="sum to 1.0"):
            OSPSentimentDistribution(positive=0.8, neutral=0.3, negative=0.2)

    def test_sum_under_tolerance_fails(self):
        with pytest.raises(ValidationError, match="sum to 1.0"):
            OSPSentimentDistribution(positive=0.1, neutral=0.1, negative=0.1)

    def test_negative_value_rejected(self):
        with pytest.raises(ValidationError):
            OSPSentimentDistribution(positive=-0.1, neutral=0.6, negative=0.5)


# ─── OSPRelationship ──────────────────────────────────────────────────────────

class TestOSPRelationship:
    def _rel(self, **kw) -> dict:
        base = {
            "source_entity_id": "ent_001",
            "target_entity_id": "ent_002",
            "type":             "CO_MENTION",
            "strength":         0.7,
            "context":          ["finance"],
        }
        base.update(kw)
        return base

    @pytest.mark.parametrize("t", ["CO_MENTION", "CAUSAL", "ADVERSARIAL", "COLLABORATIVE"])
    def test_all_valid_types(self, t):
        OSPRelationship(**self._rel(type=t))

    def test_invalid_type_rejected(self):
        with pytest.raises(ValidationError, match="relationship type"):
            OSPRelationship(**self._rel(type="ALLIED"))

    def test_source_equals_target_rejected(self):
        with pytest.raises(ValidationError, match="must differ"):
            OSPRelationship(**self._rel(source_entity_id="ent_001", target_entity_id="ent_001"))

    def test_strength_boundary_zero(self):
        OSPRelationship(**self._rel(strength=0.0))

    def test_strength_boundary_one(self):
        OSPRelationship(**self._rel(strength=1.0))

    def test_strength_over_range(self):
        with pytest.raises(ValidationError):
            OSPRelationship(**self._rel(strength=1.1))

    def test_empty_context_list_ok(self):
        OSPRelationship(**self._rel(context=[]))


# ─── OSPNarrativeThread ───────────────────────────────────────────────────────

class TestOSPNarrativeThread:
    def test_valid_thread(self):
        t = OSPNarrativeThread(id="t1", label="Thread", entity_ids=["e1"], velocity=0.5)
        assert t.velocity == 0.5

    def test_velocity_zero(self):
        OSPNarrativeThread(id="t1", label="Thread", entity_ids=[], velocity=0.0)

    def test_velocity_one(self):
        OSPNarrativeThread(id="t1", label="Thread", entity_ids=[], velocity=1.0)

    def test_velocity_out_of_range(self):
        with pytest.raises(ValidationError):
            OSPNarrativeThread(id="t1", label="Thread", entity_ids=[], velocity=1.5)


# ─── OSPCorePayload ───────────────────────────────────────────────────────────

class TestOSPCorePayload:
    def test_valid_core(self):
        p = OSPCorePayload(**core_payload())
        assert p.confidence == 0.9
        assert len(p.entities) == 1

    def test_standard_event_types_accepted(self):
        for et in [
            "story.created", "story.updated", "story.state_changed",
            "entity.spike", "entity.sentiment_shift", "trend.detected",
            "relationship.new", "graph.updated", "context.refresh", "alert.custom",
        ]:
            OSPCorePayload(**core_payload(event_type=et))

    def test_custom_event_type_accepted(self):
        OSPCorePayload(**core_payload(event_type="custom.my_feed.alert"))

    def test_invalid_event_type_rejected(self):
        with pytest.raises(ValidationError, match="event_type"):
            OSPCorePayload(**core_payload(event_type="bad.event"))

    def test_confidence_zero(self):
        OSPCorePayload(**core_payload(confidence=0.0))

    def test_confidence_one(self):
        OSPCorePayload(**core_payload(confidence=1.0))

    def test_confidence_over_range(self):
        with pytest.raises(ValidationError):
            OSPCorePayload(**core_payload(confidence=1.01))

    def test_empty_entities_ok(self):
        OSPCorePayload(**core_payload(entities=[]))

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            OSPCorePayload(**core_payload(unknown_field="x"))

    def test_dump_exclude_none(self):
        p = OSPCorePayload(**core_payload())
        d = p.model_dump(exclude_none=True)
        assert "entities" in d
        # No None values
        assert all(v is not None for v in d.values())


# ─── OSPExtendedPayload ───────────────────────────────────────────────────────

class TestOSPExtendedPayload:
    def test_all_optional_fields_absent(self):
        # Only core fields — should be valid
        p = OSPExtendedPayload(**core_payload())
        assert p.narrative_state is None
        assert p.cluster_id is None

    def test_full_extended(self):
        p = OSPExtendedPayload(**extended_payload())
        assert p.narrative_state == "DEVELOPING"
        assert p.source_count == 5
        assert p.geographic_focus == ["US", "GB"]

    @pytest.mark.parametrize("state", [
        "EMERGING", "DEVELOPING", "PEAK", "DECLINING", "CONCLUDED", "CONTESTED"
    ])
    def test_all_narrative_states_accepted(self, state):
        OSPExtendedPayload(**extended_payload(narrative_state=state))

    def test_invalid_narrative_state_rejected(self):
        with pytest.raises(ValidationError, match="narrative_state"):
            OSPExtendedPayload(**extended_payload(narrative_state="UNKNOWN"))

    def test_iso_code_validation_passes(self):
        OSPExtendedPayload(**extended_payload(geographic_focus=["US", "DE", "JP"]))

    def test_iso_code_lowercase_rejected(self):
        with pytest.raises(ValidationError, match="ISO 3166"):
            OSPExtendedPayload(**extended_payload(geographic_focus=["us"]))

    def test_iso_code_three_chars_rejected(self):
        with pytest.raises(ValidationError, match="ISO 3166"):
            OSPExtendedPayload(**extended_payload(geographic_focus=["USA"]))

    def test_inherits_core_fields(self):
        p = OSPExtendedPayload(**extended_payload())
        assert p.event_type == "story.created"
        assert p.confidence == 0.9


# ─── OSPGraphPayload ──────────────────────────────────────────────────────────

class TestOSPGraphPayload:
    def test_valid_graph_payload(self):
        p = OSPGraphPayload(**graph_payload())
        assert len(p.relationships) == 1
        assert len(p.narrative_threads) == 1

    def test_contested_without_counter_narrative_sets_flag(self):
        p = OSPGraphPayload(**graph_payload(narrative_state="CONTESTED"))
        assert p._contested_without_counter_narrative is True

    def test_contested_with_counter_narrative_clears_flag(self):
        data = graph_payload(
            narrative_state="CONTESTED",
            counter_narrative={
                "detected":   True,
                "confidence": 0.8,
                "summary":    "Alternative framing detected.",
            },
        )
        p = OSPGraphPayload(**data)
        assert p._contested_without_counter_narrative is False

    def test_non_contested_without_counter_narrative_no_flag(self):
        p = OSPGraphPayload(**graph_payload(narrative_state="DEVELOPING"))
        assert p._contested_without_counter_narrative is False

    def test_empty_relationships_ok(self):
        p = OSPGraphPayload(**graph_payload(relationships=[]))
        assert p.relationships == []

    def test_inherits_extended_fields(self):
        p = OSPGraphPayload(**graph_payload())
        assert p.narrative_state == "DEVELOPING"
        assert p.source_count == 5


# ─── OSPLLMContextPayload ─────────────────────────────────────────────────────

class TestOSPLLMContextPayload:
    def test_valid_llm_payload(self):
        p = OSPLLMContextPayload(**llm_payload())
        assert "Polari" in p.grounding_statement
        assert len(p.key_facts) == 1
        assert len(p.open_questions) == 1
        assert len(p.provenance) == 1
        assert p.valid_until == FUTURE

    def test_valid_until_must_be_after_emitted_at(self):
        with pytest.raises(ValidationError, match="valid_until"):
            OSPLLMContextPayload(**llm_payload(valid_until=NOW - 1))

    def test_valid_until_equal_to_emitted_at_rejected(self):
        with pytest.raises(ValidationError, match="valid_until"):
            OSPLLMContextPayload(**llm_payload(valid_until=NOW))

    def test_empty_key_facts_ok(self):
        OSPLLMContextPayload(**llm_payload(key_facts=[]))

    def test_empty_open_questions_ok(self):
        OSPLLMContextPayload(**llm_payload(open_questions=[]))

    def test_empty_provenance_ok(self):
        OSPLLMContextPayload(**llm_payload(provenance=[]))

    def test_inherits_graph_fields(self):
        p = OSPLLMContextPayload(**llm_payload())
        assert len(p.relationships) == 1
        assert p.narrative_state == "DEVELOPING"

    def test_dump_has_all_tiers(self):
        p = OSPLLMContextPayload(**llm_payload())
        d = p.model_dump(exclude_none=True)
        # Core
        assert "event_type" in d
        assert "confidence" in d
        # Extended
        assert "narrative_state" in d
        assert "cluster_id" in d
        # Graph
        assert "relationships" in d
        # LLM
        assert "grounding_statement" in d
        assert "key_facts" in d
        assert "open_questions" in d
        assert "valid_until" in d


# ─── OSPAdapter ───────────────────────────────────────────────────────────────

class TestOSPAdapterTier0:
    def _full(self) -> dict:
        p = OSPLLMContextPayload(**llm_payload())
        return p.model_dump(exclude_none=True)

    def test_headline_truncated_to_160(self):
        long_headline = "A" * 200
        p = OSPCorePayload(**core_payload(headline=long_headline))
        d = p.model_dump(exclude_none=True)
        adapted = OSPAdapter.adapt(d, Tier.TIER_0)
        assert len(adapted["headline"]) == 160

    def test_short_headline_not_truncated(self):
        p = OSPCorePayload(**core_payload(headline="Short"))
        d = p.model_dump(exclude_none=True)
        adapted = OSPAdapter.adapt(d, Tier.TIER_0)
        assert adapted["headline"] == "Short"

    def test_confidence_omitted(self):
        adapted = OSPAdapter.adapt(self._full(), Tier.TIER_0)
        assert "confidence" not in adapted

    def test_narrative_state_omitted(self):
        adapted = OSPAdapter.adapt(self._full(), Tier.TIER_0)
        assert "narrative_state" not in adapted

    def test_relationships_omitted(self):
        adapted = OSPAdapter.adapt(self._full(), Tier.TIER_0)
        assert "relationships" not in adapted

    def test_only_first_entity_name(self):
        adapted = OSPAdapter.adapt(self._full(), Tier.TIER_0)
        # entity should be a single string, not a list
        assert "entity" in adapted
        assert isinstance(adapted["entity"], str)

    def test_no_entity_key_when_entities_empty(self):
        p = OSPCorePayload(**core_payload(entities=[]))
        d = p.model_dump(exclude_none=True)
        adapted = OSPAdapter.adapt(d, Tier.TIER_0)
        assert "entity" not in adapted
        assert "entities" not in adapted

    def test_adapt_to_bytes_tier0_is_plain_text(self):
        d = OSPCorePayload(**core_payload()).model_dump(exclude_none=True)
        b = OSPAdapter.adapt_to_bytes(d, Tier.TIER_0)
        assert isinstance(b, bytes)
        # Should decode as UTF-8 string (the headline)
        text = b.decode("utf-8")
        assert text == "Test headline"


class TestOSPAdapterTier1:
    def _full(self) -> dict:
        # Use 7 entities to test the cap at 5
        entities = [entity(id=f"ent_{i:03d}", name=f"Entity {i}") for i in range(7)]
        p = OSPLLMContextPayload(**llm_payload(entities=entities))
        return p.model_dump(exclude_none=True)

    def test_confidence_included(self):
        adapted = OSPAdapter.adapt(self._full(), Tier.TIER_1)
        assert "confidence" in adapted

    def test_entities_capped_at_5(self):
        adapted = OSPAdapter.adapt(self._full(), Tier.TIER_1)
        assert len(adapted["entities"]) == 5

    def test_narrative_state_omitted(self):
        adapted = OSPAdapter.adapt(self._full(), Tier.TIER_1)
        assert "narrative_state" not in adapted

    def test_relationships_omitted(self):
        adapted = OSPAdapter.adapt(self._full(), Tier.TIER_1)
        assert "relationships" not in adapted

    def test_provenance_omitted(self):
        adapted = OSPAdapter.adapt(self._full(), Tier.TIER_1)
        assert "provenance" not in adapted

    def test_entity_fields_present(self):
        adapted = OSPAdapter.adapt(self._full(), Tier.TIER_1)
        e = adapted["entities"][0]
        assert "id" in e
        assert "name" in e
        assert "sentiment" in e


class TestOSPAdapterTier2:
    def _full(self) -> dict:
        p = OSPLLMContextPayload(**llm_payload())
        return p.model_dump(exclude_none=True)

    def test_narrative_state_included(self):
        adapted = OSPAdapter.adapt(self._full(), Tier.TIER_2)
        assert "narrative_state" in adapted

    def test_sentiment_distribution_included(self):
        adapted = OSPAdapter.adapt(self._full(), Tier.TIER_2)
        assert "sentiment_distribution" in adapted

    def test_relationships_omitted(self):
        adapted = OSPAdapter.adapt(self._full(), Tier.TIER_2)
        assert "relationships" not in adapted

    def test_narrative_threads_omitted(self):
        adapted = OSPAdapter.adapt(self._full(), Tier.TIER_2)
        assert "narrative_threads" not in adapted

    def test_provenance_omitted(self):
        adapted = OSPAdapter.adapt(self._full(), Tier.TIER_2)
        assert "provenance" not in adapted

    def test_key_facts_omitted(self):
        adapted = OSPAdapter.adapt(self._full(), Tier.TIER_2)
        assert "key_facts" not in adapted

    def test_grounding_statement_omitted(self):
        adapted = OSPAdapter.adapt(self._full(), Tier.TIER_2)
        assert "grounding_statement" not in adapted


class TestOSPAdapterTier3:
    def _full(self) -> dict:
        p = OSPLLMContextPayload(**llm_payload())
        return p.model_dump(exclude_none=True)

    def test_graph_json_relationships_included(self):
        adapted = OSPAdapter.adapt(
            self._full(), Tier.TIER_3, content_type=ContentType.GRAPH_JSON
        )
        assert "relationships" in adapted

    def test_graph_json_llm_fields_omitted(self):
        adapted = OSPAdapter.adapt(
            self._full(), Tier.TIER_3, content_type=ContentType.GRAPH_JSON
        )
        assert "grounding_statement" not in adapted
        assert "key_facts" not in adapted
        assert "provenance" not in adapted

    def test_llm_context_all_fields_included(self):
        adapted = OSPAdapter.adapt(
            self._full(), Tier.TIER_3, content_type=ContentType.LLM_CONTEXT
        )
        assert "relationships" in adapted
        assert "grounding_statement" in adapted
        assert "key_facts" in adapted
        assert "open_questions" in adapted
        assert "provenance" in adapted
        assert "valid_until" in adapted

    def test_adapt_to_bytes_tier3_is_json(self):
        d = self._full()
        b = OSPAdapter.adapt_to_bytes(d, Tier.TIER_3, content_type=ContentType.LLM_CONTEXT)
        parsed = json.loads(b)
        assert "grounding_statement" in parsed


class TestOSPAdapterSizeCheck:
    def test_within_tier0_limit(self):
        b = b"Short headline"
        fits, size, limit = OSPAdapter.check_size(b, Tier.TIER_0)
        assert fits
        assert size == len(b)
        assert limit == 160

    def test_over_tier0_limit(self):
        b = b"X" * 161
        fits, size, limit = OSPAdapter.check_size(b, Tier.TIER_0)
        assert not fits
        assert size == 161
        assert limit == 160

    def test_tier3_no_limit(self):
        b = b"X" * 1_000_000
        fits, size, limit = OSPAdapter.check_size(b, Tier.TIER_3)
        assert fits
        assert limit is None

    def test_within_tier2_limit(self):
        b = b"X" * 1000
        fits, size, limit = OSPAdapter.check_size(b, Tier.TIER_2)
        assert fits
        assert limit == 65536


class TestOSPAdapterDowngrade:
    """Tier 3 payload adapted to Tier 1 should produce Tier 1 profile."""

    def test_downgrade_tier3_to_tier1(self):
        p = OSPLLMContextPayload(**llm_payload())
        d = p.model_dump(exclude_none=True)
        adapted = OSPAdapter.adapt(d, Tier.TIER_1)
        # Tier 1 rules apply
        assert "confidence" in adapted
        assert "narrative_state" not in adapted
        assert "relationships" not in adapted
        assert "grounding_statement" not in adapted
