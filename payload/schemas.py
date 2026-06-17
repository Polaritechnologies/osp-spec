"""
OSP Payload Schemas
Open Smart Protocol v0.2 — Section 6

Four additive schema tiers, each a strict superset of the tier below:

    OSPCorePayload          — all tiers (Section 6.1)
    OSPExtendedPayload      — Tier 2+ (Section 6.2)
    OSPGraphPayload         — Tier 3 / osp/graph-json (Section 6.3)
    OSPLLMContextPayload    — osp/llm-context (Section 6.4)

Design notes
------------
- Every model uses model_config = ConfigDict(extra="forbid") so unknown
  fields are caught at parse time rather than silently ignored.
- All optional fields default to None rather than being absent from the
  model, which makes tier adaptation (field stripping) explicit.
- model_dump(exclude_none=True) produces clean JSON with no null fields.
- Sentiment floats are clamped to [-1.0, 1.0]; confidence/strength floats
  to [0.0, 1.0]. Validators raise ValueError on violation so bad publisher
  payloads are rejected before they reach the delivery engine.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ─── Entity types (valid values for OSPEntity.type) ──────────────────────────

ENTITY_TYPES = {"PERSON", "ORG", "GPE", "LOC", "EVENT", "CONCEPT"}

# ─── Relationship types ───────────────────────────────────────────────────────

RELATIONSHIP_TYPES = {"CO_MENTION", "CAUSAL", "ADVERSARIAL", "COLLABORATIVE"}

# ─── Narrative states ─────────────────────────────────────────────────────────

NARRATIVE_STATES = {"EMERGING", "DEVELOPING", "PEAK", "DECLINING", "CONCLUDED", "CONTESTED"}


# ─── Shared sub-models ────────────────────────────────────────────────────────

class OSPEntity(BaseModel):
    """
    Named entity present in the payload.

    Used in all tiers. At Tier 0 only the first entity name is delivered
    (adaptation handled by OSPAdapter, not here). At Tier 1+ the full
    list is included.
    """
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, description="Publisher-scoped entity identifier")
    name: str = Field(..., min_length=1, description="Canonical display name")
    type: str = Field(..., description="PERSON | ORG | GPE | LOC | EVENT | CONCEPT")
    sentiment: float = Field(
        0.0,
        ge=-1.0, le=1.0,
        description="Sentiment toward this entity (-1.0 negative → 1.0 positive)",
    )

    @field_validator("type")
    @classmethod
    def validate_entity_type(cls, v: str) -> str:
        if v not in ENTITY_TYPES:
            raise ValueError(f"entity type must be one of {ENTITY_TYPES}, got {v!r}")
        return v


class OSPSentimentDistribution(BaseModel):
    """
    Distribution of sentiment across the articles in a cluster.
    The three floats must sum to approximately 1.0 (tolerance ±0.01).
    """
    model_config = ConfigDict(extra="forbid")

    positive: float = Field(..., ge=0.0, le=1.0)
    neutral:  float = Field(..., ge=0.0, le=1.0)
    negative: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def sum_to_one(self) -> "OSPSentimentDistribution":
        total = self.positive + self.neutral + self.negative
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"sentiment_distribution must sum to 1.0 (±0.01), got {total:.4f}"
            )
        return self


class OSPRelationship(BaseModel):
    """
    A directed relationship between two entities.
    Present in Tier 3 / osp/graph-json payloads (Section 6.3).
    """
    model_config = ConfigDict(extra="forbid")

    source_entity_id: str = Field(..., min_length=1)
    target_entity_id: str = Field(..., min_length=1)
    type: str = Field(..., description="CO_MENTION | CAUSAL | ADVERSARIAL | COLLABORATIVE")
    strength: float = Field(..., ge=0.0, le=1.0, description="Relationship strength 0.0–1.0")
    context: list[str] = Field(
        default_factory=list,
        description="Contextual topic tags for this relationship",
    )

    @field_validator("type")
    @classmethod
    def validate_relationship_type(cls, v: str) -> str:
        if v not in RELATIONSHIP_TYPES:
            raise ValueError(
                f"relationship type must be one of {RELATIONSHIP_TYPES}, got {v!r}"
            )
        return v

    @field_validator("source_entity_id", "target_entity_id")
    @classmethod
    def not_same_entity(cls, v: str) -> str:
        # Individual field validation can't compare both fields;
        # cross-field check is in model_validator below
        return v

    @model_validator(mode="after")
    def source_not_target(self) -> "OSPRelationship":
        if self.source_entity_id == self.target_entity_id:
            raise ValueError(
                f"source_entity_id and target_entity_id must differ, "
                f"got {self.source_entity_id!r} for both"
            )
        return self


class OSPNarrativeThread(BaseModel):
    """
    A coherent sub-narrative within a story cluster.
    Present in Tier 3 payloads (Section 6.3).
    """
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    entity_ids: list[str] = Field(default_factory=list)
    velocity: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Rate of change 0.0 (static) → 1.0 (fast-moving)",
    )


class OSPCounterNarrative(BaseModel):
    """
    Counter-narrative signal. Present when narrative_state == CONTESTED.
    Defined in Section 6.3.
    """
    model_config = ConfigDict(extra="forbid")

    detected: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str = Field(..., min_length=1)


class OSPProvenance(BaseModel):
    """
    Source provenance entry for osp/llm-context payloads (Section 6.4).
    """
    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., min_length=1, description="Publication or feed name")
    url: str = Field(..., min_length=1)
    published_at: int = Field(..., gt=0, description="Unix timestamp of publication")
    reliability_score: float = Field(..., ge=0.0, le=1.0)


# ─── Tier schemas ─────────────────────────────────────────────────────────────

class OSPCorePayload(BaseModel):
    """
    Section 6.1 — Core fields present in all tiers.

    headline is capped at 160 chars at Tier 0; the raw value stored here
    may be longer (up to 512 chars). Tier adaptation truncates on output.
    """
    model_config = ConfigDict(extra="forbid")

    event_type:   str = Field(..., min_length=1, description="Registered event type identifier")
    publisher_id: int = Field(..., gt=0, description="Matches frame header publisher_id")
    emitted_at:   int = Field(..., gt=0, description="Unix timestamp (matches frame header)")
    headline:     str = Field(..., min_length=1, max_length=512)
    confidence:   float = Field(..., ge=0.0, le=1.0, description="Publisher confidence 0.0–1.0")
    entities:     list[OSPEntity] = Field(default_factory=list)

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        from frame.constants import EventType
        if not EventType.is_valid(v):
            raise ValueError(
                f"event_type {v!r} is not a standard OSP event type and does not "
                f"use the custom.* prefix"
            )
        return v


class OSPExtendedPayload(OSPCorePayload):
    """
    Section 6.2 — Extended fields for Tier 2+.

    All extended fields are Optional so that a partially-populated
    extended payload is still valid — publishers populate what they have.
    """
    model_config = ConfigDict(extra="forbid")

    narrative_state:         Optional[str]                       = None
    sentiment_distribution:  Optional[OSPSentimentDistribution]  = None
    source_count:            Optional[int]                       = Field(None, ge=0)
    article_count:           Optional[int]                       = Field(None, ge=0)
    geographic_focus:        Optional[list[str]]                 = None   # ISO 3166-1 alpha-2
    topics:                  Optional[list[str]]                 = None   # controlled vocabulary
    cluster_id:              Optional[str]                       = None   # publisher-scoped

    @field_validator("narrative_state")
    @classmethod
    def validate_narrative_state(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in NARRATIVE_STATES:
            raise ValueError(
                f"narrative_state must be one of {NARRATIVE_STATES}, got {v!r}"
            )
        return v

    @field_validator("geographic_focus")
    @classmethod
    def validate_iso_codes(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if v is None:
            return v
        for code in v:
            if not (len(code) == 2 and code.isupper() and code.isalpha()):
                raise ValueError(
                    f"geographic_focus entries must be ISO 3166-1 alpha-2 codes "
                    f"(2 uppercase letters), got {code!r}"
                )
        return v


class OSPGraphPayload(OSPExtendedPayload):
    """
    Section 6.3 — Graph fields for Tier 3 / osp/graph-json.

    Adds entity relationship graph, narrative threads, and counter-narrative.
    counter_narrative should be present when narrative_state == CONTESTED.
    """
    model_config = ConfigDict(extra="forbid")

    relationships:     list[OSPRelationship]        = Field(default_factory=list)
    narrative_threads: list[OSPNarrativeThread]     = Field(default_factory=list)
    counter_narrative: Optional[OSPCounterNarrative] = None

    @model_validator(mode="after")
    def counter_narrative_when_contested(self) -> "OSPGraphPayload":
        """
        Warn (not error) when narrative_state is CONTESTED but
        counter_narrative is absent. This is a soft constraint — the
        spec says counter_narrative is present 'when CONTESTED', but
        we don't want to break payloads that partially populate graph fields.
        """
        # Soft check: we log rather than raise, but track the violation
        # so callers can inspect _contested_without_counter_narrative.
        self._contested_without_counter_narrative = (
            self.narrative_state == "CONTESTED" and self.counter_narrative is None
        )
        return self


class OSPLLMContextPayload(OSPGraphPayload):
    """
    Section 6.4 — LLM context type (content type 0x0010).

    Packages OSP intelligence as a structured fragment for injection into
    a language model context window.

    Key fields:
    - grounding_statement: declarative sentence summarizing current world state
    - key_facts: ordered verifiable factual claims the model can assert
    - open_questions: what the model must NOT assert (reduces confabulation)
    - provenance: source attribution for downstream verification
    - valid_until: expiry timestamp — gateway must enforce strictly
    """
    model_config = ConfigDict(extra="forbid")

    grounding_statement: str                    = Field(..., min_length=1)
    key_facts:           list[str]              = Field(default_factory=list)
    open_questions:      list[str]              = Field(default_factory=list)
    provenance:          list[OSPProvenance]    = Field(default_factory=list)
    valid_until:         int                    = Field(
        ...,
        gt=0,
        description="Unix timestamp after which this payload must not be used for grounding",
    )

    @model_validator(mode="after")
    def valid_until_after_emitted_at(self) -> "OSPLLMContextPayload":
        if self.valid_until <= self.emitted_at:
            raise ValueError(
                f"valid_until ({self.valid_until}) must be after emitted_at ({self.emitted_at})"
            )
        return self