"""FastAPI Application for GovSim AI (`fiscal_ai_engine`).

Microservice 1: Fiscal AI & Deterministic Microsimulation Engine
- Grounded in the German Federal Budget 2027 Government Draft (Regierungsentwurf 2027, BT-Drs. 21/7300).
- Uses `gemini-3.8-flash` with strict Pydantic structured outputs (`PolicyExtractionSchema`),
  optimized for low latency (`thinking_budget=0` by default) and optional Google Search grounding.
- Strictly zero heuristic/deterministic NLP fallback: all natural language policy analysis is performed
  by Gemini.
"""

from __future__ import annotations

from datetime import datetime
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    FiscalSummary,
    GroundingCitation,
    MacroAssessmentCritiqueSchema,
    MacroAssessmentRequest,
    MacroAssessmentResponse,
    PolicyExtractionSchema,
    SimulationRequest,
    SimulationResult,
)
from .simulator import GERMANY_2027_DRAFT_BASELINE, FiscalSimulator

load_dotenv()

logger = logging.getLogger("fiscal_ai_engine")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


SYSTEM_EXTRACTION_INSTRUCTION = """You are the Chief Fiscal Economist AI for GovSim AI, an institutional policy & budget microsimulation platform.
All output (including `policy_name`, `chat_feedback`, `display_name`, `rationale`, and `extracted_assumptions`) MUST be written in clear, executive **English**, even if the user writes their prompt in German or another language.

Official Baseline Dataset: **German Federal Budget 2027 Government Draft (Regierungsentwurf 2027, Drucksache 21/7300)**:
- Total Core Budget Expenditures: **€555.4B**
- Total Non-Credit Revenues: **€436.7B** (Federal Tax Revenues: €398.6B + Non-Tax/Other: €38.1B)
- Net Borrowing Requirement (Nettokreditaufnahme): **€118.7B**
- Adult Population: **68.8 million** across 10 equal income deciles (6.88M adults per decile).

Available Revenue Categories (`target_side = "revenue"`):
- `income_and_wealth_taxes`: €162.4B (Income & Solidarity Taxes, Federal Share)
- `value_added_and_consumption_taxes`: €146.8B (VAT & Import VAT, Federal Share)
- `corporate_and_business_taxes`: €44.2B (Corporate & Trade Taxes)
- `energy_and_environmental_taxes`: €45.2B (Energy, Carbon, Electricity, Aviation & Excise Taxes)
- `non_tax_and_other_revenues`: €38.1B (Bundesbank Profit, Truck Tolls, Fees & EU Inflows)

Available Expenditure Categories (`target_side = "expenditure"`, 2027 Einzelpläne):
- `labor_and_social_affairs_ep11`: €201.5B (Einzelplan 11 BMAS — Statutory Pension Subsidies ~€133B, Citizen's Benefit / Bürgergeld ~€42.6B, Labor Market Policy)
- `defense_and_security_ep14`: €109.7B (Einzelplan 14 BMVg — Federal Defense & Military Procurement, +€27.1B surge in 2027 Draft)
- `federal_debt_interest_ep32`: €43.6B (Einzelplan 32 Bundesschuld — Federal Debt Interest Service)
- `transport_and_digital_ep12`: €26.4B (Einzelplan 12 BMV — Federal Rail, Autobahn, Waterways & Digital Infrastructure)
- `research_tech_and_aerospace_ep30`: €22.0B (Einzelplan 30 BMFTR — Research, AI, Quantum, Biotech & Aerospace)
- `family_education_and_youth_ep17`: €15.5B (Einzelplan 17 BMBFSFJ — Family Allowances, Elterngeld, Youth & Education)
- `economic_affairs_energy_climate_ep09_16`: €8.3B (Einzelplan 09 BMWE €5.55B + Einzelplan 16 BMUKN €2.74B — Economic Policy, Energy & Environment)
- `general_financial_admin_and_other`: €128.4B (Einzelplan 60 General Financial Admin, EP 15 Health, EP 06 Interior, EP 23 Development BMZ €9.47B, EP 08 Finance €11.1B & Federal Subsidies)

Instructions:
1. Write a concise, authoritative 1-3 sentence `chat_feedback` in **English** explaining how the user's policy idea was translated into specific 2027 budget adjustments and its net fiscal effect.
2. Map the policy into `parameters.budget_modifications` using:
   - `DELTA_BN`: Add or subtract EUR billions from an existing category.
   - `DELTA_PCT`: Increase or decrease an existing category by a decimal percentage (e.g., `-0.10` for -10%, `0.15` for +15%).
   - `SET_ABSOLUTE_BN`: Set a category to an exact EUR billion target.
   - `ABOLISH`: Remove or reduce a sub-program (specify the EUR billion savings in `value`, or `0.0` to eliminate the entire category).
   - `NEW_LINE_BN`: Introduce a new revenue or expenditure program with `value` in EUR billions.
3. If the policy alters income tax rates across population deciles (1=lowest, 10=highest), populate `parameters.tax_bracket_shifts`.
4. If the policy introduces a per-capita household transfer, climate dividend, or negative income tax / basic grant, populate `parameters.transfer_formula`.
5. Always evaluate policies grounded in the German institutional context, factoring in the social security system, differences in net wages, and real household consumption without relying on artificial deterministic rules.
"""


class GeminiPolicyExtractor:
    """Service for fast, structured LLM policy decomposition using `gemini-3.8-flash`."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required.")

    def extract_policy(
        self, raw_prompt: str, enable_search_grounding: bool = False
    ) -> Tuple[PolicyExtractionSchema, str, List[GroundingCitation]]:
        """Extract structured policy parameters via Gemini API. Strictly no deterministic fallback."""
        return self._extract_via_gemini(raw_prompt, enable_search_grounding)

    def _extract_via_gemini(
        self, raw_prompt: str, enable_search_grounding: bool
    ) -> Tuple[PolicyExtractionSchema, str, List[GroundingCitation]]:
        """Invoke google-genai SDK with structured output schema and low-latency thinking config."""
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        tools = [types.Tool(google_search=types.GoogleSearch())] if enable_search_grounding else None

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_EXTRACTION_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=PolicyExtractionSchema,
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            tools=tools,
        )

        response = client.models.generate_content(
            model=self.model_name,
            contents=raw_prompt,
            config=config,
        )

        if not response.text:
            raise ValueError("Empty response text returned by Gemini API.")

        extracted_schema = PolicyExtractionSchema.model_validate_json(response.text)
        citations = self._parse_grounding_metadata(response)
        return extracted_schema, "GEMINI_STRUCTURED_GROUNDED", citations

    @staticmethod
    def _parse_grounding_metadata(response: Any) -> List[GroundingCitation]:
        """Extract web search citations from Gemini candidate grounding metadata."""
        citations: List[GroundingCitation] = []
        try:
            candidates = getattr(response, "candidates", None) or []
            for candidate in candidates:
                grounding_meta = getattr(candidate, "grounding_metadata", None)
                if not grounding_meta:
                    continue
                chunks = getattr(grounding_meta, "grounding_chunks", None) or []
                for chunk in chunks:
                    web = getattr(chunk, "web", None)
                    if web and getattr(web, "uri", None):
                        citations.append(
                            GroundingCitation(
                                title=getattr(web, "title", None) or "Google Search Grounded Source",
                                uri=web.uri,
                                snippet="Retrieved via Gemini Google Search grounding during policy analysis.",
                                retrieved_via="GOOGLE_SEARCH_GROUNDING",
                            )
                        )
        except Exception as exc:
            logger.debug("Could not parse grounding metadata chunks: %s", exc)
        return citations

    def assess_macro_critique(
        self,
        applied_proposals: List[Dict[str, Any]],
        active_summary: Optional[FiscalSummary],
    ) -> MacroAssessmentCritiqueSchema:
        """Call Gemini to critically evaluate the reform package and calculate dynamic macroeconomic feedback."""
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)

        proposals_text = ""
        for i, p in enumerate(applied_proposals, 1):
            name = p.get("policy_name", "Reform")
            prompt = p.get("raw_prompt", "")
            desc = p.get("short_description", "")
            rev = p.get("revenue_delta_bn", 0.0)
            exp = p.get("expenditure_delta_bn", 0.0)
            bal = p.get("net_budget_impact_bn", 0.0)
            lines = p.get("line_item_deltas", [])
            lines_desc = ", ".join(f"{it.get('display_name')}: {it.get('delta_bn', 0):+.2f}B" for it in lines)
            proposals_text += f"\nReform #{i}: {name}\n- User Prompt: \"{prompt}\"\n- Description: {desc}\n- Static Delta: Rev {rev:+.2f}B, Exp {exp:+.2f}B, Net Balance {bal:+.2f}B\n- Line Items: {lines_desc}\n"

        cur_exp = active_summary.total_expenditure_bn if active_summary else 555.4
        cur_rev = active_summary.total_revenue_bn if active_summary else 436.7
        cur_bor = active_summary.net_borrowing_requirement_bn if active_summary else 118.7

        rev_lines_str = (
            "\n".join(f"  - {k}: €{v:.2f}B" for k, v in active_summary.revenue_breakdown_bn.items())
            if active_summary
            else ""
        )
        exp_lines_str = (
            "\n".join(f"  - {k}: €{v:.2f}B" for k, v in active_summary.expenditure_breakdown_bn.items())
            if active_summary
            else ""
        )

        prompt_content = f"""Critically audit and assess this accumulated reform package modifying the German Federal Budget:

Baseline Reference & Current Active Post-Reform Budget State:
- Active Total Expenditures: €{cur_exp:.2f}B
- Active Total Non-Credit Revenues: €{cur_rev:.2f}B
- Active Net Borrowing: €{cur_bor:.2f}B

Active Revenue Categories (Post-Reform Static State):
{rev_lines_str}

Active Expenditure Categories (Post-Reform Static State):
{exp_lines_str}

Applied Reform Package to Evaluate (Chronological Sequence):
{proposals_text if proposals_text.strip() else "No specific reforms applied (Active state is at baseline)."}

Your Evaluation Focus:
1. Zero-Base Before-vs-After Component Accounting (`net_accounting_ledger`):
   - CRITICAL REQUIREMENT: To evaluate post-reform rates, burdens, and costs, you MUST calculate the post-reform total FROM ZERO by summing ONLY the surviving and newly introduced components (setting every abolished or replaced component strictly to 0).
   - NEVER commit the Baseline-Carryover Fallacy: Never add a new or increased component on top of an old pre-reform composite total that already included the abolished component!
   - If an increased or newly introduced levy/mechanism merely replaces an abolished baseline levy/mechanism of similar magnitude, the net combined burden is OFFSET (unchanged or restructured), NOT increased!
2. Audit for Miscalculations & Accounting Coherence:
   - Check whether the static numbers in the package contain arithmetic errors, unmodeled fiscal gaps, double counting, or unrealistic revenue/cost projections given the post-reform rules.
3. Social Security System, Net Wages & Household Consumption:
   - Factor in how the post-reform rules (accounting for all abolished, replaced, and newly introduced measures together) affect statutory social systems, net take-home wages across income groups, and resulting shifts in private household consumption and indirect taxes.
4. Live Empirical Grounding:
   - Use Google Search freely to verify actual institutional data, macroeconomic indicators, and empirical research.

Provide:
1. `net_accounting_ledger`: Explicit zero-base accounting ledger calculating post-reform totals by setting abolished components to 0 before summing surviving/new components.
2. `assessment_title`: An executive, analytical title.
3. `executive_critique`: A sharp, internally consistent assessment strictly aligned with `net_accounting_ledger`.
4. `dynamic_gdp_growth_impact_pct`: Realistic annual real GDP delta.
5. `factors`: Key macroeconomic transmission channels reflecting the net post-reform state.
6. `suggested_line_corrections`: Specific line adjustments (using the exact `category_key` identifiers listed above) in EUR billions to correct miscalculations and reflect dynamic economic feedback."""

        macro_instruction = """You are an Independent Senior Fiscal Economist and Macroeconomic Auditor.
Your role is to critically assess policy reform packages on the German Federal Budget.
You have access to Google Search to look up current empirical data, institutional parameters, and economic research.

Core Principles of Logical & Economic Consistency:
1. Zero-Base Accounting Integrity (No Baseline-Carryover Fallacy):
   - Always evaluate the post-reform world under the NEW rules established by the user's package.
   - When a package abolishes, replaces, or reduces an existing baseline levy, contribution, fee, or program, that baseline element is ELIMINATED (set to 0) in the post-reform world.
   - NEVER calculate post-reform marginal rates, household deductions, or budgetary costs by taking the old composite baseline and adding the replacement measure on top. Compute the post-reform burden strictly as the sum of surviving post-reform components (with abolished components equal to 0).
   - If a new or higher tax/levy replaces an abolished baseline contribution/levy, recognize that this is a substitution/shift of financing; the net burden is the difference between the two, NOT their sum.
2. Holistic Net Evaluation:
   - When a package links offsetting changes, evaluate the combined net change in household disposable income, marginal incentives, and fiscal balance across the entire package combined.
3. Check for Miscalculations & Unmodeled Feedback:
   - Rigorously audit the static figures for arithmetic discrepancies, missing funding, or unmodeled behavioral/macroeconomic effects on net wages, private consumption, social security balances, and public borrowing.
4. No Deterministic Fallacies:
   - Reason dynamically from first principles and live empirical evidence via Google Search, maintaining strict internal consistency between your ledger and your conclusions.

Output strictly valid JSON complying with the requested schema."""

        tools = [types.Tool(google_search=types.GoogleSearch())]

        config = types.GenerateContentConfig(
            system_instruction=macro_instruction,
            response_mime_type="application/json",
            response_schema=MacroAssessmentCritiqueSchema,
            temperature=0.2,
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
            tools=tools,
        )

        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt_content,
            config=config,
        )

        if not response.text:
            raise ValueError("Empty response from Gemini during macroeconomic critique.")

        return MacroAssessmentCritiqueSchema.model_validate_json(response.text)


# FastAPI Dependency Providers
def get_simulator() -> FiscalSimulator:
    return FiscalSimulator(baseline=GERMANY_2027_DRAFT_BASELINE)


def get_policy_extractor() -> GeminiPolicyExtractor:
    return GeminiPolicyExtractor()


app = FastAPI(
    title="GovSim AI — Fiscal Policy & 2027 Budget Microsimulation Engine",
    description=(
        "Microservice 1 (`fiscal_ai_engine`): Decomposes natural language policy proposals via "
        "`gemini-3.8-flash` and executes deterministic microsimulations on the German Federal Budget "
        "2027 Government Draft (Regierungsentwurf 2027)."
    ),
    version="2.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Liveness and readiness probe for the Python simulation microservice."""
    has_gemini_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    return {
        "status": "healthy",
        "service": "fiscal_ai_engine",
        "version": "2.0.0",
        "model": model_name,
        "baseline": "Germany Federal Budget 2027 Government Draft (Regierungsentwurf 2027)",
        "gemini_api_configured": has_gemini_key,
        "supported_countries": ["DE"],
    }


@app.get("/api/v1/baseline", tags=["Simulation"])
async def get_baseline_data(
    simulator: FiscalSimulator = Depends(get_simulator),
) -> Dict[str, Any]:
    """Return the initial 2027 Government Draft baseline summary, dual vertical bar chart, and comparison table."""
    initial_summary = simulator.get_initial_summary()
    initial_bar_chart = simulator.build_bar_chart_for_summary(
        initial_summary=initial_summary,
        before_summary=initial_summary,
        after_summary=initial_summary,
    )
    initial_sankey = simulator.build_sankey_for_summary(
        reference_summary=initial_summary,
        active_summary=initial_summary,
    )
    b = simulator.baseline
    return {
        "country_code": b.country_code,
        "fiscal_year": b.fiscal_year,
        "baseline_label": "German Federal Budget 2027 — Government Draft (Regierungsentwurf 2027)",
        "adult_population_millions": b.adult_population_millions,
        "baseline_summary": initial_summary,
        "bar_chart": initial_bar_chart,
        "sankey_graph": initial_sankey,
    }


@app.post(
    "/api/v1/simulate",
    response_model=SimulationResult,
    status_code=status.HTTP_200_OK,
    tags=["Simulation"],
)
async def simulate_policy(
    request: SimulationRequest,
    simulator: FiscalSimulator = Depends(get_simulator),
    extractor: GeminiPolicyExtractor = Depends(get_policy_extractor),
) -> SimulationResult:
    """Interpret a policy proposal via Gemini 3.8 Flash and compute 2027 budget impacts."""
    try:
        extracted_policy, extraction_mode, grounded_citations = extractor.extract_policy(
            raw_prompt=request.raw_prompt,
            enable_search_grounding=request.enable_search_grounding,
        )

        if request.override_parameters is not None:
            extracted_policy.parameters = request.override_parameters
            extraction_mode = "DIRECT_OVERRIDE"

        result = simulator.run_simulation(
            raw_prompt=request.raw_prompt,
            extracted_policy=extracted_policy,
            proposal_id=request.proposal_id,
            current_baseline_override=request.current_baseline_override,
            extraction_mode=extraction_mode,
            model_used=extractor.model_name,
            extra_citations=grounded_citations,
        )
        return result
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid policy simulation parameters: {val_err}",
        ) from val_err
    except Exception as exc:
        logger.exception("Unhandled error during policy simulation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation engine failure: {exc}",
        ) from exc


@app.post(
    "/api/v1/assess-package",
    response_model=MacroAssessmentResponse,
    status_code=status.HTTP_200_OK,
    tags=["Simulation"],
)
async def assess_reform_package(
    request: MacroAssessmentRequest,
    simulator: FiscalSimulator = Depends(get_simulator),
    extractor: GeminiPolicyExtractor = Depends(get_policy_extractor),
) -> MacroAssessmentResponse:
    """Critically assess the reform package, factor in economic changes, and compute dynamic corrections."""
    try:
        initial_summary = simulator.get_initial_summary()
        active_summary = request.active_summary or initial_summary

        critique = extractor.assess_macro_critique(
            applied_proposals=request.applied_proposals,
            active_summary=active_summary,
        )

        # Apply line corrections to construct the dynamically corrected summary
        corrected_rev_map = dict(active_summary.revenue_breakdown_bn)
        corrected_exp_map = dict(active_summary.expenditure_breakdown_bn)

        for corr in critique.suggested_line_corrections:
            if corr.side == "revenue":
                orig = corrected_rev_map.get(corr.category_key, corr.original_amount_bn)
                new_val = max(0.0, round(orig + corr.adjustment_delta_bn, 3))
                corr.original_amount_bn = round(orig, 3)
                corr.corrected_amount_bn = new_val
                corr.adjustment_delta_bn = round(new_val - orig, 3)
                corrected_rev_map[corr.category_key] = new_val
            elif corr.side == "expenditure":
                orig = corrected_exp_map.get(corr.category_key, corr.original_amount_bn)
                new_val = max(0.0, round(orig + corr.adjustment_delta_bn, 3))
                corr.original_amount_bn = round(orig, 3)
                corr.corrected_amount_bn = new_val
                corr.adjustment_delta_bn = round(new_val - orig, 3)
                corrected_exp_map[corr.category_key] = new_val

        total_rev = round(sum(corrected_rev_map.values()), 3)
        total_exp = round(sum(corrected_exp_map.values()), 3)
        balance = round(total_rev - total_exp, 3)
        borrowing = round(max(0.0, -balance), 3)

        corrected_summary = FiscalSummary(
            total_revenue_bn=total_rev,
            total_expenditure_bn=total_exp,
            fiscal_balance_bn=balance,
            net_borrowing_requirement_bn=borrowing,
            revenue_breakdown_bn=corrected_rev_map,
            expenditure_breakdown_bn=corrected_exp_map,
        )

        corrected_bar_chart = simulator.build_bar_chart_for_summary(
            initial_summary=initial_summary,
            before_summary=active_summary,
            after_summary=corrected_summary,
        )

        net_correction = round(corrected_summary.fiscal_balance_bn - active_summary.fiscal_balance_bn, 3)

        return MacroAssessmentResponse(
            assessment_title=critique.assessment_title,
            executive_critique=critique.executive_critique,
            dynamic_gdp_growth_impact_pct=round(critique.dynamic_gdp_growth_impact_pct, 2),
            static_net_borrowing_bn=round(active_summary.net_borrowing_requirement_bn, 2),
            corrected_net_borrowing_bn=round(corrected_summary.net_borrowing_requirement_bn, 2),
            net_correction_impact_bn=net_correction,
            factors=critique.factors,
            line_corrections=critique.suggested_line_corrections,
            corrected_summary=corrected_summary,
            corrected_bar_chart=corrected_bar_chart,
            assessed_at=datetime.utcnow().isoformat() + "Z",
        )
    except Exception as exc:
        logger.exception("Failed during macroeconomic assessment")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Macroeconomic assessment engine failure: {exc}",
        ) from exc
