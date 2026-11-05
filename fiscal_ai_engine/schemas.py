"""Pydantic schemas for GovSim AI (Interactive Policy & Fiscal Budget Simulator).

Designed for an executive, English-language conversational policy-to-budget workflow
grounded in the German Federal Budget 2027 Government Draft (Regierungsentwurf 2027):
1. User enters any abstract or concrete policy proposal in Chat.
2. Gemini 3.8 Flash decomposes the idea into structured fiscal primitives
   (`BudgetLineModification`, `TransferFormulaConfig`, `TaxBracketShift`), generates
   an executive `chat_feedback` summary, and runs the deterministic microsimulation.
3. User reviews the calculation card and clicks "Apply to Budget" to commit the changes
   to the dual vertical stacked bar chart (Revenues vs. Expenditures) and the full
   Before/After comparison table.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


PolicyDomain = Literal[
    "SOCIAL_AND_TRANSFERS",
    "TAXATION_AND_REVENUE",
    "INFRASTRUCTURE_AND_CLIMATE",
    "DEFENSE_AND_SECURITY",
    "EDUCATION_AND_INNOVATION",
    "STRUCTURAL_CONSOLIDATION",
    "MULTI_DOMAIN_REFORM",
]


class TaxBracketShift(BaseModel):
    """Adjusts effective income tax rates for a specific adult population decile."""

    decile: int = Field(
        ...,
        ge=1,
        le=10,
        description="Adult income decile (1 = lowest 10%, 10 = highest 10%).",
    )
    rate_delta_pct: float = Field(
        ...,
        ge=-0.50,
        le=0.50,
        description="Change in effective income tax rate (e.g., 0.03 for +3% points, -0.02 for -2% points).",
    )


class TransferFormulaConfig(BaseModel):
    """Generic income-tested or universal per-capita transfer/dividend formula.

    Evaluates T(Y) = max(0, base_monthly_grant_eur * 12 - income_withdrawal_rate * Y)
    for deciles <= max_eligible_decile.
    """

    program_label: str = Field(
        default="Direct Household Transfer / Dividend",
        description="English display name of the new transfer or dividend mechanism.",
    )
    base_monthly_grant_eur: float = Field(
        default=0.0,
        ge=0.0,
        le=10000.0,
        description="Monthly baseline grant G in EUR per eligible adult.",
    )
    income_withdrawal_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Marginal phase-out rate t in [0.0, 1.0] per EUR of gross annual income Y (0.0 = universal flat payout).",
    )
    max_eligible_decile: int = Field(
        default=10,
        ge=1,
        le=10,
        description="Highest income decile eligible for the transfer (10 = all adults).",
    )

    @field_validator("income_withdrawal_rate", mode="before")
    @classmethod
    def normalize_rate(cls, v: float) -> float:
        if isinstance(v, (int, float)) and 1.0 < v <= 100.0:
            return float(v) / 100.0
        return float(v)


class BudgetLineModification(BaseModel):
    """Generic modification to an existing or newly created revenue/expenditure line."""

    target_side: Literal["revenue", "expenditure"] = Field(
        ...,
        description="Whether this change targets public revenues or public expenditures.",
    )
    category_key: str = Field(
        ...,
        description="Canonical budget category identifier (or new custom snake_case key).",
    )
    display_name: str = Field(
        ...,
        description="English human-readable label for the budget line.",
    )
    operation: Literal["DELTA_BN", "DELTA_PCT", "SET_ABSOLUTE_BN", "ABOLISH", "NEW_LINE_BN"] = Field(
        ...,
        description="Type of mathematical adjustment to apply to the category.",
    )
    value: float = Field(
        default=0.0,
        description=(
            "Numeric value for the operation: EUR billions for DELTA_BN / SET_ABSOLUTE_BN / NEW_LINE_BN, "
            "or decimal fraction (e.g. -0.15 for -15%, +0.20 for +20%) for DELTA_PCT."
        ),
    )
    rationale: str = Field(
        default="",
        description="Brief English explanation of how this budget line modification implements the user's idea.",
    )


class PolicyParameters(BaseModel):
    """Universal fiscal policy parameters capable of representing any abstract policy idea."""

    budget_modifications: List[BudgetLineModification] = Field(
        default_factory=list,
        description="Direct additions, cuts, abolitions, or new programs across revenue and expenditure lines.",
    )
    tax_bracket_shifts: List[TaxBracketShift] = Field(
        default_factory=list,
        description="Distributional income tax rate adjustments across adult population deciles (1..10).",
    )
    transfer_formula: Optional[TransferFormulaConfig] = Field(
        default=None,
        description="Optional per-capita household transfer or dividend formula across income deciles.",
    )


class PolicyExtractionSchema(BaseModel):
    """Structured schema extracted by Gemini from a user's conversational chat prompt."""

    policy_name: str = Field(
        ...,
        description="Concise English title summarizing the policy proposal.",
    )
    policy_domain: PolicyDomain = Field(
        ...,
        description="Primary policy domain classification.",
    )
    chat_feedback: str = Field(
        ...,
        description=(
            "Concise, professional English 1-3 sentence executive response explaining how the proposal "
            "was mapped into concrete 2027 budget adjustments and its net fiscal impact."
        ),
    )
    parameters: PolicyParameters = Field(
        ...,
        description="Quantitative fiscal primitives representing the user's proposal.",
    )
    grounding_queries: List[str] = Field(
        default_factory=list,
        description="Search queries used to ground macroeconomic and fiscal baseline figures.",
    )
    extracted_assumptions: List[str] = Field(
        default_factory=list,
        description="Key economic and demographic assumptions in English.",
    )


class GroundingCitation(BaseModel):
    """Citation metadata from Google Search grounding or official statistical baselines."""

    title: str = Field(..., description="Source document or institution title.")
    uri: str = Field(..., description="Canonical URL or reference identifier.")
    snippet: Optional[str] = Field(
        default=None,
        description="Key fiscal metric or factual excerpt supporting the simulation.",
    )
    retrieved_via: Literal["GOOGLE_SEARCH_GROUNDING", "OFFICIAL_BASELINE_REGISTRY"] = Field(
        default="OFFICIAL_BASELINE_REGISTRY",
        description="Origin of the citation.",
    )


class FiscalSummary(BaseModel):
    """Macro-fiscal aggregate summary (Before vs. After)."""

    total_revenue_bn: float = Field(
        ..., description="Total annual non-credit public revenues in EUR billions."
    )
    total_expenditure_bn: float = Field(
        ..., description="Total annual public expenditures in EUR billions."
    )
    fiscal_balance_bn: float = Field(
        ..., description="Net fiscal balance (Revenues - Expenditures) in EUR billions. Negative indicates deficit."
    )
    net_borrowing_requirement_bn: float = Field(
        ..., description="Required net new borrowing (Nettokreditaufnahme) to balance the budget in EUR billions."
    )
    revenue_breakdown_bn: Dict[str, float] = Field(
        ..., description="Revenue streams by category key in EUR billions."
    )
    expenditure_breakdown_bn: Dict[str, float] = Field(
        ..., description="Expenditure items by category key in EUR billions."
    )


class SimulationRequest(BaseModel):
    """Incoming request payload for POST /api/v1/simulate."""

    proposal_id: Optional[str] = Field(
        default=None,
        description="Optional unique proposal ID assigned by the API gateway.",
    )
    raw_prompt: str = Field(
        ...,
        min_length=3,
        max_length=4000,
        description="Freeform natural language policy proposal written by the user.",
    )
    country_code: Literal["DE"] = Field(
        default="DE",
        description="ISO 3166-1 alpha-2 country code for the baseline fiscal dataset.",
    )
    fiscal_year: int = Field(
        default=2027,
        ge=2024,
        le=2030,
        description="Baseline budget year for simulation (defaults to 2027 Government Draft).",
    )
    current_baseline_override: Optional[FiscalSummary] = Field(
        default=None,
        description="Optional active budget state when building cumulatively on previously applied proposals.",
    )
    override_parameters: Optional[PolicyParameters] = Field(
        default=None,
        description="Optional explicit parameter overrides.",
    )
    enable_search_grounding: bool = Field(
        default=False,
        description=(
            "Whether to invoke live Google Search grounding during extraction. "
            "Defaults to False for sub-5-second fast execution using the built-in 2027 Regierungsentwurf baseline."
        ),
    )


class LineItemDelta(BaseModel):
    """Individual line-item change displayed in the Calculation Card and Before/After Comparison Table."""

    side: Literal["revenue", "expenditure"] = Field(..., description="Budget side.")
    category_key: str = Field(..., description="Category identifier.")
    display_name: str = Field(..., description="English human-readable category name.")
    baseline_bn: float = Field(..., description="Before value in EUR billions.")
    proposed_bn: float = Field(..., description="After value in EUR billions.")
    delta_bn: float = Field(..., description="Absolute difference (proposed_bn - baseline_bn) in EUR billions.")
    delta_pct: float = Field(
        default=0.0,
        description="Percentage change relative to baseline_bn (e.g. 12.5 for +12.5%).",
    )


class ComparisonTableRow(BaseModel):
    """Full row in the Before / After comparison table across all budget items."""

    side: Literal["revenue", "expenditure", "financing"] = Field(..., description="Budget section.")
    category_key: str = Field(..., description="Canonical category key.")
    display_name: str = Field(..., description="English category label.")
    initial_2027_draft_bn: float = Field(
        ..., description="Official 2027 Government Draft baseline in EUR billions."
    )
    before_bn: float = Field(
        ..., description="Active budget value prior to this proposal in EUR billions."
    )
    after_bn: float = Field(
        ..., description="Proposed budget value after this reform in EUR billions."
    )
    delta_vs_before_bn: float = Field(
        ..., description="Difference (after_bn - before_bn) in EUR billions."
    )
    delta_vs_before_pct: float = Field(
        ..., description="Percentage change relative to before_bn."
    )
    delta_vs_initial_bn: float = Field(
        ..., description="Cumulative difference (after_bn - initial_2027_draft_bn) in EUR billions."
    )
    share_of_total_pct: float = Field(
        ..., description="Share of total revenues or expenditures in percent."
    )


class BarSegment(BaseModel):
    """Segment of one of the two vertical budget bars (Revenues Bar vs. Expenditures Bar)."""

    id: str = Field(..., description="Stable segment ID.")
    category_key: str = Field(..., description="Canonical category key.")
    name: str = Field(..., description="English display label.")
    side: Literal["revenue", "expenditure"] = Field(..., description="Which vertical bar this segment belongs to.")
    segment_type: Literal["regular", "deficit_borrowing", "surplus_reserve", "transfer"] = Field(
        default="regular",
        description="Semantic type used for styling and hatching.",
    )
    initial_2027_bn: float = Field(..., description="Official 2027 Government Draft value in EUR billions.")
    before_bn: float = Field(..., description="Value before current proposal in EUR billions.")
    after_bn: float = Field(..., description="Active/proposed value in EUR billions.")
    delta_bn: float = Field(..., description="Change (after_bn - before_bn) in EUR billions.")
    share_pct: float = Field(..., description="Percentage share of the bar.")


class DualBarChartPayload(BaseModel):
    """Payload for the two vertical stacked bars (Revenues vs. Expenditures) + Before/After Table."""

    revenue_bar: List[BarSegment] = Field(
        ..., description="Ordered segments for the vertical Revenues & Financing bar."
    )
    expenditure_bar: List[BarSegment] = Field(
        ..., description="Ordered segments for the vertical Expenditures & Allocations bar."
    )
    comparison_table: List[ComparisonTableRow] = Field(
        ..., description="Complete Before/After comparison table rows across all budget items."
    )
    unit: Literal["EUR_BILLIONS"] = Field(default="EUR_BILLIONS")


class DecileImpact(BaseModel):
    """Microsimulation breakdown for a single adult population income decile."""

    decile: int = Field(..., ge=1, le=10, description="Income decile (1 = lowest, 10 = highest).")
    population_millions: float = Field(
        ..., description="Adult population count in this decile (in millions)."
    )
    mean_annual_gross_income_eur: float = Field(
        ..., description="Mean annual pre-tax market income Y in EUR per adult."
    )
    annual_transfer_per_capita_eur: float = Field(
        ..., description="Evaluated annual direct transfer/dividend per capita in EUR."
    )
    total_decile_transfer_cost_bn: float = Field(
        ..., description="Aggregate annual transfer expenditure for this decile in EUR billions."
    )
    baseline_income_tax_bn: float = Field(
        ..., description="Baseline income tax contribution from this decile in EUR billions."
    )
    reformed_income_tax_bn: float = Field(
        ..., description="Proposed income tax contribution from this decile in EUR billions."
    )
    net_disposable_income_delta_eur: float = Field(
        ..., description="Average annual change in per-capita disposable income in EUR."
    )


class FiscalDeltaSummary(BaseModel):
    """Summary of net fiscal impacts displayed in the Chat Calculation Card."""

    total_revenue_delta_bn: float = Field(
        ..., description="Net change in total non-credit public revenues in EUR billions."
    )
    total_expenditure_delta_bn: float = Field(
        ..., description="Net change in total public expenditures in EUR billions."
    )
    net_budget_impact_bn: float = Field(
        ...,
        description="Net change in fiscal balance (Proposed Balance - Baseline Balance) in EUR billions.",
    )
    transfer_program_cost_bn: float = Field(
        default=0.0,
        description="Gross annual cost of any per-capita household transfer formula in EUR billions.",
    )
    line_item_deltas: List[LineItemDelta] = Field(
        default_factory=list,
        description="Changed revenue and expenditure line items for compact display next to the Apply button.",
    )


class SankeyNode(BaseModel):
    """Legacy D3.js Sankey node definition kept for OpenAPI backwards compatibility."""

    id: str
    name: str
    category: Literal["revenue", "budget_pool", "expenditure", "transfer", "deficit_surplus"]
    column: int
    baseline_value_bn: float
    reformed_value_bn: float
    delta_bn: float


class SankeyLink(BaseModel):
    """Legacy D3.js Sankey link definition kept for OpenAPI backwards compatibility."""

    source: str
    target: str
    value: float
    baseline_value: float
    delta_bn: float
    flow_type: Literal["revenue_inflow", "deficit_financing", "budget_allocation", "surplus_reserve"]
    label: str


class SankeyGraphPayload(BaseModel):
    """Legacy D3.js Sankey diagram payload."""

    nodes: List[SankeyNode]
    links: List[SankeyLink]
    unit: Literal["EUR_BILLIONS"] = Field(default="EUR_BILLIONS")


class ApplyActionMetadata(BaseModel):
    """UI descriptor for the 'Apply' button rendered next to the chat calculation card."""

    can_apply: bool = Field(
        default=True,
        description="Whether this calculation is ready to be applied to the active budget visualization.",
    )
    status: Literal["PROPOSED", "APPLIED"] = Field(
        default="PROPOSED",
        description="Lifecycle state: 'PROPOSED' when shown in chat; 'APPLIED' once transferred to the budget.",
    )
    button_label: str = Field(
        default="Apply to Budget",
        description="English UI label for the action button next to the calculation.",
    )
    apply_endpoint: str = Field(
        ...,
        description="Gateway endpoint to call when the user clicks 'Apply' (e.g., POST /api/scenarios/{proposal_id}/apply).",
    )


class SimulationMetadata(BaseModel):
    """Execution metadata, formulas, and provenance."""

    extraction_mode: Literal["GEMINI_STRUCTURED_GROUNDED", "DIRECT_OVERRIDE"] = Field(
        ..., description="How the natural language prompt was translated into policy parameters."
    )
    model_used: str = Field(
        default="gemini-3.8-flash",
        description="Gemini model identifier used for structured extraction.",
    )
    formulas_applied: List[str] = Field(
        ..., description="LaTeX / mathematical formulas evaluated by the deterministic simulator."
    )
    assumptions: List[str] = Field(
        ..., description="Modeling assumptions applied during the simulation run."
    )
    citations: List[GroundingCitation] = Field(
        ..., description="Data sources and search grounding citations."
    )
    timestamp_utc: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the calculation completed.",
    )


class SimulationResult(BaseModel):
    """Complete response payload returned by POST /api/v1/simulate."""

    proposal_id: str = Field(..., description="Unique identifier for this chat policy calculation.")
    country_code: str = Field(..., description="ISO country code of the baseline model.")
    fiscal_year: int = Field(..., description="Simulated fiscal year (2027 Government Draft).")
    raw_prompt: str = Field(..., description="Original natural language message from the user.")
    chat_feedback: str = Field(
        ...,
        description="Concise English executive explanation displayed in the chat reply alongside the calculation card.",
    )
    extracted_policy: PolicyExtractionSchema = Field(
        ..., description="Structured policy configuration extracted from the user's message."
    )
    initial_2027_summary: FiscalSummary = Field(
        ..., description="Unmodified official 2027 Government Draft baseline summary."
    )
    baseline_summary: FiscalSummary = Field(
        ..., description="Active fiscal budget summary prior to this proposal (Before state)."
    )
    reformed_summary: FiscalSummary = Field(
        ..., description="Proposed fiscal budget summary after this proposal (After state)."
    )
    fiscal_delta: FiscalDeltaSummary = Field(
        ..., description="Compact calculation breakdown displayed next to the Apply button."
    )
    bar_chart: DualBarChartPayload = Field(
        ..., description="Dual vertical stacked bar chart payload (Revenues vs. Expenditures) and Before/After comparison table."
    )
    decile_impacts: List[DecileImpact] = Field(
        ..., description="Distributional impact across all 10 adult income deciles."
    )
    sankey_graph: SankeyGraphPayload = Field(
        ..., description="Sankey diagram payload retained for API compatibility."
    )
    apply_action: ApplyActionMetadata = Field(
        ..., description="Action metadata powering the 'Apply to Budget' button next to the calculation."
    )
    metadata: SimulationMetadata = Field(
        ..., description="Audit trail of formulas, assumptions, and grounding citations."
    )


class MacroFactorEffect(BaseModel):
    """Dynamic economic channel impact (e.g. Multiplier, Elasticity, Yields)."""

    factor_name: str = Field(..., description="Name of the economic mechanism.")
    direction: Literal["positive", "negative", "neutral"] = Field(
        ..., description="Net effect on fiscal sustainability."
    )
    revenue_effect_bn: float = Field(
        default=0.0, description="Second-round dynamic effect on tax revenues in EUR billions."
    )
    expenditure_effect_bn: float = Field(
        default=0.0, description="Dynamic effect on expenditures / debt service in EUR billions."
    )
    explanation: str = Field(..., description="Concise explanation of the economic transmission channel.")


class LineCorrection(BaseModel):
    """Specific line item adjustment recommended by the macroeconomic assessment."""

    category_key: str = Field(..., description="Budget category identifier.")
    display_name: str = Field(..., description="Human-readable category label.")
    side: Literal["revenue", "expenditure"] = Field(..., description="Revenue or expenditure line.")
    original_amount_bn: float = Field(..., description="Static amount before macro adjustment.")
    corrected_amount_bn: float = Field(..., description="Dynamically corrected amount in EUR billions.")
    adjustment_delta_bn: float = Field(..., description="Adjustment (corrected - original) in EUR billions.")
    reason: str = Field(..., description="Economic justification for this line correction.")


class PostReformNetLedger(BaseModel):
    """Zero-base before-vs-after accounting ledger to prevent baseline-carryover double counting."""

    baseline_components_before_reform: str = Field(
        ...,
        description="State the pre-reform baseline components and their combined baseline total (e.g., Component A + Component B = Baseline Total).",
    )
    eliminated_or_reduced_components: str = Field(
        ...,
        description="Explicitly list every baseline component abolished or reduced by the reform package and state that its post-reform value is now 0 (or its reduced value).",
    )
    zero_base_post_reform_total: str = Field(
        ...,
        description=(
            "Calculate the post-reform total FROM ZERO by summing ONLY the surviving and newly introduced "
            "components (with all abolished components set strictly to 0). Never add a new or increased "
            "component onto the old pre-reform composite total."
        ),
    )
    net_change_vs_baseline: str = Field(
        ...,
        description=(
            "Compute (zero_base_post_reform_total MINUS baseline_components_before_reform). If an increase in "
            "one component merely replaces an abolished component of similar size, state clearly that the net "
            "combined burden is unchanged or offset, NOT increased."
        ),
    )


class MacroAssessmentCritiqueSchema(BaseModel):
    """Structured output expected from Gemini for macroeconomic critique."""

    net_accounting_ledger: PostReformNetLedger = Field(
        ...,
        description=(
            "Mandatory first step: compute the zero-base post-reform net state by setting abolished baseline "
            "components to 0 before summing surviving/new components, ensuring replaced burdens are never "
            "carried over or double-counted."
        ),
    )
    assessment_title: str = Field(
        default="Macroeconomic Reality Check & Dynamic Feedback Calibration",
        description="Title of the critical economic assessment.",
    )
    executive_critique: str = Field(
        ...,
        description="Critical, authoritative 2-4 sentence macroeconomic critique of the package (must strictly obey net_accounting_ledger.net_change_vs_baseline).",
    )
    dynamic_gdp_growth_impact_pct: float = Field(
        default=0.0,
        description="Estimated net impact on annual real GDP growth (e.g., 0.3 for +0.3%, -0.2 for -0.2%).",
    )
    factors: List[MacroFactorEffect] = Field(
        ..., description="Key economic transmission channels factored into the revision (must strictly obey net_accounting_ledger and never claim a total burden rose if an increase merely offset an abolished component)."
    )
    suggested_line_corrections: List[LineCorrection] = Field(
        ...,
        description="Specific line adjustments to existing active budget categories to correct miscalculations and reflect dynamic economic feedback.",
    )


class MacroAssessmentRequest(BaseModel):
    """Request payload to critically assess the active reform package."""

    fiscal_year: int = Field(default=2027)
    applied_proposals: List[Dict[str, Any]] = Field(default_factory=list)
    active_summary: Optional[FiscalSummary] = None


class MacroAssessmentResponse(BaseModel):
    """Response returned by POST /api/v1/assess-package."""

    assessment_title: str
    executive_critique: str
    dynamic_gdp_growth_impact_pct: float
    static_net_borrowing_bn: float
    corrected_net_borrowing_bn: float
    net_correction_impact_bn: float
    factors: List[MacroFactorEffect]
    line_corrections: List[LineCorrection]
    corrected_summary: FiscalSummary
    corrected_bar_chart: DualBarChartPayload
    assessed_at: str


MacroAssessmentRequest.model_rebuild()
MacroAssessmentResponse.model_rebuild()
MacroAssessmentCritiqueSchema.model_rebuild()
