"""FastAPI Application for GovSim AI (`fiscal_ai_engine`).

Microservice 1: Fiscal AI & Deterministic Microsimulation Engine
- Grounded in the German Federal Budget 2027 Government Draft (Regierungsentwurf 2027, BT-Drs. 21/7300).
- Uses `gemini-3.8-flash` with strict Pydantic structured outputs (`PolicyExtractionSchema`),
  optimized for low latency (`thinking_budget=0` by default) and optional Google Search grounding.
- Strictly zero heuristic/deterministic NLP fallback: all natural language policy analysis is performed
  by Gemini.
"""

from __future__ import annotations

import base64
from datetime import datetime
import io
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import docx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import pypdf
from pydantic import BaseModel

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


SYSTEM_EXTRACTION_INSTRUCTION = """You are the Chief Fiscal Economist AI for GovSim AI.
Speed & Precision Directive: Be decisive, mathematically exact, and institutional. All outputs must be in crisp English.

Baseline Reference: German Federal Budget 2027 Draft (Regierungsentwurf 2027, BT-Drs. 21/7300):
- Total Core Expenditures: €555.4B | Total Non-Credit Revenues: €436.7B | Net Borrowing (Deficit): €118.7B | 68.8M adults.

Revenue Categories (Federal Share):
- `income_and_wealth_taxes` (€162.4B) — Wage Tax, Assessed Income Tax, Solidarity Surcharge, Capital Withholding Tax
- `value_added_and_consumption_taxes` (€146.8B) — Federal share of VAT (Umsatzsteuer) & Import VAT
- `corporate_and_business_taxes` (€44.2B) — Corporate Tax (Körperschaftsteuer) & Trade Tax federal share
- `energy_and_environmental_taxes` (€45.2B) — Energy Tax, Electricity Tax, Aviation, Tobacco & Insurance Taxes
- `non_tax_and_other_revenues` (€38.1B) — Bundesbank profit, CO2/Tolls, administrative fees & EU inflows

Expenditure Categories (2027 Einzelpläne):
- `labor_and_social_affairs_ep11` (€201.5B) — BMAS: Statutory Pensions (~€133B), Citizen's Benefit (Bürgergeld), Basic Security
- `defense_and_security_ep14` (€109.7B) — BMVg: Federal Armed Forces (Bundeswehr), Military Procurement & NATO capabilities
- `federal_debt_interest_ep32` (€43.6B) — Bundesschuld: Federal Debt Service & Interest
- `transport_and_digital_ep12` (€26.4B) — BMV: Rail Infrastructure, Roads, Waterways & Digital Connectivity
- `research_tech_and_aerospace_ep30` (€22.0B) — BMFTR: Science, Universities, AI, Innovation, Space & Future Technologies
- `family_education_and_youth_ep17` (€15.5B) — BMBFSFJ: Child Benefits, Parental Allowance (Elterngeld), Youth & Education
- `economic_affairs_energy_climate_ep09_16` (€8.3B) — BMWE & BMUKN: Economic Transformation, Energy Grids & Climate Actions
- `general_financial_admin_and_other` (€128.4B) — EP 60, 06, 08, 15, 23: General Financial Administration, Health, Interior, Development Aid (BMZ €9.5B), Subsidies & Reserves

================================================================================
CRITICAL DIRECTIVE 1: ZERO DOUBLE-COUNTING & NO PHANTOM FINANCING
================================================================================
1. FINANCING CUTS ARE EXPENDITURE REDUCTIONS, NEVER REVENUES:
   When a proposal states: "Increase spending on X by €A, funded by cutting €B from Y":
   - Measure 1 (Expenditure Increase): target_side: "expenditure", category_key: "X", operation: "DELTA_BN", value: +A
   - Measure 2 (Expenditure Cut): target_side: "expenditure", category_key: "Y", operation: "DELTA_BN", value: -B (MUST be negative!)
   - STRICT PROHIBITION: NEVER enter a spending cut or subsidy reduction as a revenue increase (`target_side: "revenue"`).
     Cutting an expenditure decreases public spending; it does NOT generate new tax or toll revenue.
     Entering a spending cut on the revenue side creates fictitious phantom revenue and falsely doubles the net deficit reduction!

2. TAX CUTS / RELIEF ARE REVENUE REDUCTIONS, NEVER EXPENDITURES:
   When a proposal reduces taxes or offers tax relief:
   - target_side: "revenue", operation: "DELTA_BN", value: -X (MUST be a negative number).
   - NEVER model tax relief as a public expenditure.

3. DEDUPLICATION ACROSS PRIMITIVES:
   - If using `parameters.transfer_formula` for a household dividend (e.g. climate dividend, universal payout), DO NOT ALSO add a `BudgetLineModification` for the transfer cost. The simulation engine automatically evaluates the formula.
   - If using `parameters.tax_bracket_shifts` for decile tax rates, DO NOT ALSO add a `BudgetLineModification` for `income_and_wealth_taxes`.

4. STRICT SIGN CONVENTIONS:
   - `DELTA_BN`: Positive (`+10.0`) for spending increases / tax hikes. Negative (`-10.0`) for spending cuts / tax relief.
   - `DELTA_PCT`: Decimal fraction (e.g., `-0.10` for -10% cut, `+0.05` for +5% increase).
   - `NEW_LINE_BN`: Positive number for creating an entirely new distinct budget program.

================================================================================
CRITICAL DIRECTIVE 2: EXHAUSTIVE EXTRACTION (NO NEGLECTED PARTS)
================================================================================
1. In `deconstruction_ledger.raw_measures_extracted`:
   You MUST enumerate EVERY single fiscal pillar, spending allocation, tax modification, social security reform, sub-federal shift, and subsidy reduction present in the user prompt or uploaded document.
   Do NOT sample, do NOT pick favorites, and do NOT truncate.
2. In `deconstruction_ledger.accounting_validation_audit`:
   Audit each measure to verify:
   a) Every fiscal measure in `raw_measures_extracted` is accounted for in `parameters.budget_modifications` (do NOT create dummy `0.0` entries in `budget_modifications` for purely non-fiscal regulatory clauses; instead, consolidate fiscal measures into non-zero `budget_modifications`).
   b) No spending cuts are entered as revenues.
   c) All measures from the input are covered (`all_measures_covered: true`).
3. Use `parameters.budget_modifications` as the primary authoritative vehicle for all revenue and expenditure adjustments (leave `transfer_formula` as null and `tax_bracket_shifts` as empty `[]` when `budget_modifications` already models the full income tax and transfer deltas, to avoid duplicate evaluation).

================================================================================
CRITICAL DIRECTIVE 3: COMPREHENSIVE REFORM PACKAGES & MULTI-LEVEL / SOCIAL SECURITY INTEGRATION
================================================================================
1. MODEL COMPLEX SYSTEMIC TRANSFORMATIONS ACROSS ALL LEVELS OF PUBLIC FINANCE:
   Comprehensive reform packages often overhaul the entire German public sector — integrating **Social Security (RV, ALV, GKV/PV)** and **Sub-Federal Levels (Länder & Kommunen)** with the **Federal Budget (Bund)**.
   - STRICT PROHIBITION ON `0.0` FOR SUB-FEDERAL OR SOCIAL SECURITY REFORMS:
     NEVER assign `0.0` to a fiscal measure simply because it targets Social Security (statutory pensions RV, unemployment ALV, healthcare GKV/PV) or sub-federal levels (Länder/Kommunen taxes like Erbschaftsteuer, Grunderwerbsteuer, Bodenwertsteuer, Gewerbesteuer, or municipal outlays like Kosten der Unterkunft KdU, Hilfe zur Pflege, schools/universities)!
     Under German constitutional fiscal federalism (*Finanzausgleich* Art. 106/107 GG & *Konnexitätsprinzip*) and federal social security guarantees, all public finance pillars are interconnected. You MUST model their real, non-zero EUR billion fiscal impacts mapped onto the canonical revenue and expenditure categories!

2. SOCIAL SECURITY (PARA-FISCI: RV, ALV, GKV/PV) TRANSMISSION & ABSORPTION RULES:
   - Status-Quo Context: Statutory Pension Insurance (RV, 18.6% rate, ~€400B+ spending), Unemployment Insurance (ALV, 2.6% rate, ~€45B spending), and Health/Long-Term Care (GKV/PV, ~20.8% combined rate, ~€370B+ spending) are separate from the core federal budget, connected via federal grants (*Bundeszuschüsse*: ~€133B for pensions in `labor_and_social_affairs_ep11`, ~€14.5B for the health fund in `general_financial_admin_and_other`) and tax-funded *Bürgergeld* (~€45-50B in `labor_and_social_affairs_ep11`).
   - Full Absorption of RV / ALV into the Tax-Transfer System:
     When a reform abolishes RV (18.6%) and ALV (2.6%) payroll contributions (e.g., via a 50:50 employer wage split `New Gross = 1.053 × Old Gross` and -4.4% employer labor cost reduction) and finances all pensions ("Altrenten / Mindestrente") and basic security / Negative Income Tax ("NES-Sockel") through general taxes:
     a) Tax Base Expansion: Lowering/abolishing deductible social contributions (§ 10 EStG) and raising gross wages by +5.3% (plus taxing pensions above the basic allowance/Sockel) massively expands taxable income (`zvE = 0.91 × Brutto`). Model this net revenue expansion in `income_and_wealth_taxes`, and model the corporate profit tax gain from lower employer labor costs in `corporate_and_business_taxes`.
     b) Absorbed Expenditure Obligations: Because RV/ALV payroll contributions (~€345B) are abolished, the tax budget must now finance not merely the baseline €133B federal pension grant and €50B Bürgergeld, but the *entire net statutory pension and basic income/NES obligation*! Model the net change in pension, NES/transfer, and family benefit outlays in `labor_and_social_affairs_ep11` and `family_education_and_youth_ep17`.
   - Off-Budget Healthcare (GKV/PV / Gesundheitsfonds) Reforms:
     When a reform alters GKV/PV contribution rates (e.g., employee 9.0% / employer 10.3%), raises the assessment ceiling (BBG, e.g., to €7,550/mo), introduces a private-insurance (PKV) solidarity levy (e.g., 3.5%), and claims the Gesundheitsfonds is self-financed without tax grants:
     a) Model the elimination of the Federal Grant to the Gesundheitsfonds (~-€14.5B expenditure reduction in `general_financial_admin_and_other`).
     b) Independently verify whether the rate cut vs. base broadening truly leaves the Gesundheitsfonds in surplus or creates an uncovered structural deficit that requires federal liquidity backstopping!

3. SUB-FEDERAL LEVELS (LÄNDER & KOMMUNEN) & VERTICAL FISCAL EQUALIZATION (*FINANZAUSGLEICH*):
   - Upward & Downward Expenditure Shifts (*Konnexität*):
     * When a federalized transfer (e.g. NES-Sockel with regional housing zones) relieves Municipalities (Kommunen) of housing costs (*Kosten der Unterkunft - KdU*, ~€22B) and care assistance (*Hilfe zur Pflege*, ~€6B), or when the 50:50 payroll split and digital registry automation reduce Länder/Kommunen administrative/personnel outlays, those sub-federal savings free up shared tax capacity under the *Finanzausgleich* and offset the federalized transfer cost!
     * Conversely, new federal-state/municipal investment commitments (e.g., a €30B Municipal Housing Construction Fund, Art. 104c GG school renovation/canteen/free-Kita pact, permanent €39 Deutschlandticket, multi-year Rail Fund) must be modeled as concrete annual expenditure additions in the appropriate categories (`general_financial_admin_and_other`, `family_education_and_youth_ep17`, `transport_and_digital_ep12`).
   - Consolidated Tax Attribution (Preventing Artificial Vertical Asymmetry):
     * When a reform package restructures the entire *Gesamtstaat* (Bund, Länder, Kommunen, Social Security) or shifts major nationwide tasks onto the central tax-transfer budget, do NOT dump 100% of newly absorbed nationwide pension/NES/family expenditures onto the Federal Budget while leaving 57.5% of the replacement income tax or 100% of inheritance/local tax changes outside the simulation!
     * Under constitutional *Konnexität* and *Finanzausgleich*, channel the net consolidated revenue deltas (including progressive *Erbschaftsteuer* in `income_and_wealth_taxes`, *Körperschaft-/Gewerbesteuer* in `corporate_and_business_taxes`, *Grunderwerbsteuer* exemptions and *Bodenwertsteuer* net shifts, and earmarked university tuition fees in `non_tax_and_other_revenues`) alongside the consolidated net expenditure deltas, so that the simulated net budget impact (`net_budget_impact_bn`) reflects the **true, integrated fiscal balance change** of the reform package!
     * For standalone targeted reforms (e.g., a simple federal income or VAT tweak without task reassignment), apply standard constitutional federal shares (Income Tax: 42.5% Bund; Solidarity Surcharge: 100% Bund; VAT: ~53% Bund; Corporate Tax: 50% Bund; Energy/Electricity/Tobacco/Aviation: 100% Bund).

================================================================================
CRITICAL DIRECTIVE 4: INDEPENDENT ECONOMETRIC CRITICAL THINKING (ANTI-BIAS MANDATE — ALWAYS THINK YOURSELF)
================================================================================
1. NEVER BLINDLY COPY AUTHOR-CLAIMED OUTCOMES OR DEFICIT NUMBERS:
   - Political manifestos, party programs, and reform white papers frequently contain optimistic self-financing claims, accounting tricks, or understated transfer costs (e.g., claiming a specific start-year deficit like -€147.1B, claiming a +€42B surplus by Year 5, claiming €48B/yr from inheritance tax, or listing €40B of subsidy cuts as revenue).
   - You are an **Independent Chief Fiscal Economist**. You must **ALWAYS THINK FOR YOURSELF** and independently calculate the realistic 2027 fiscal impact of every measure from first principles using empirical German macroeconomic benchmarks (Nominal GDP ~€4,400B; Gross Wage Bill ~€2,000B; 68.8M adults, ~46M employed, ~21.5M pensioners, ~14.5M children).
   - Where your independent calculation differs from the document's stated figures, **you MUST enter your own independent, economically defensible estimates into `parameters.budget_modifications`** and explicitly explain the discrepancy in `deconstruction_ledger` and `chat_feedback`!

2. MANDATORY INDEPENDENT REALITY CHECKS:
   a) NEGATIVE INCOME TAX (NES) / BASIC ALLOWANCE PHASE-OUT MATH:
      - In a Negative Income Tax or flat-tax-with-Sockel system (e.g. effective tax rate t_eff = 36.4% on gross income up to BBG, guaranteed Sockel S = €1,100/mo = €13,200/yr, tax-free break-even Y_BE = S / t_eff = €3,022/mo):
      - By mathematical identity, every earner above the break-even point has their gross tax reduced by the full Sockel credit (Net_Tax = t_eff * Y_i - S), while every eligible adult below break-even receives a net transfer payout (S - t_eff * Y_i).
      - Audit whether the proposal's claimed "Net Income Tax Revenue" and "NES Cash Payouts" properly account for the ~€13,200/yr tax credit across all ~46M employed workers and taxed pensioners, or whether the author overestimated net tax revenue / underestimated transfer phase-out costs!
   b) TAX BASE REALISM, DEFERRALS & BEHAVIORAL ELASTICITY:
      - *Inheritance Tax (Erbschaftsteuer)*: If a proposal claims €48B/yr (4x Germany's current ~€11-12B yield) while granting a €2M personal allowance per heir, exempting owner-occupied homes, and allowing **20-year cashflow installment payments for corporate assets**, recognize that 20-year deferral delays ~90-95% of corporate inheritance tax cash inflows in Year 1! Independently haircut the Year-1 revenue gain to a realistic level (e.g. +€8B to +€15B net increase, not +€36B).
      - *Corporate & Electricity Tax Cuts*: Cutting corporate tax on reinvested profits to 5% (and 15% on distributions) and lowering electricity tax to the EU minimum (0.05 ct/kWh, ~-€6.5B) creates immediate static revenue losses in Year 1 before dynamic investment effects mature.
      - *Capital Gains & Anti-"Buy-Borrow-Die" Rule*: Raising capital gains tax to 30% and taxing Lombard loans backed by unrealized equity gains is partially offset by a generous €10,000/yr regular allowance (€30,000/yr for retirees) and behavioral realization avoidance; apply a realistic net yield.
   c) SUBSIDY ABOLITION (SUBVENTIONSABBAU) — NO DOUBLE-COUNTING & ELASTICITY HAIRCUT:
      - If a document lists "Subventionsabbau (e.g. €40B)" as a *Revenue* line while also cutting core ministry budgets, NEVER count it twice!
      - Separate genuine tax-privilege abolitions (company car privilege, kerosene tax exemption, agricultural diesel — modeled on `revenue` with a 20-30% behavioral elasticity haircut as flights/car leases adjust) from direct budget subsidy cuts (KfW programs, ministry grants — modeled as negative `DELTA_BN` on `expenditure`).
   d) OFF-BUDGET HEALTH FUND (GKV/PV) AUDIT:
      - Critically verify claims that cutting employee GKV/PV rates to 9.0% and employer rates to 10.3% (19.3% combined vs ~20.8% status quo) while abolishing the €14.5B federal grant still produces a "+€16.5B structural surplus" from raising the BBG to €7,550, taxing retiree income above the Sockel, and a 3.5% PKV solidarity levy. Point out if the rate cut + lost federal grant largely consumes the base-broadening revenues.
   e) MULTI-YEAR CONSOLIDATION SKEPTICISM:
      - Never bake speculative Year-5 dynamic growth miracles (e.g. claiming a -€147.1B deficit turns into a +€42B surplus in 5 years) into the static 2027 budget simulation. Model the realistic 2027 Year-1 structural impact and critique the 5-year assumptions in `chat_feedback`.

Respond strictly in valid JSON complying with the schema."""


MACRO_AUDIT_INSTRUCTION = """You are an Independent Senior Fiscal Economist and Macroeconomic Auditor.
Assess the reform package on the German Federal Budget and the broader German economy (including Länder, Kommunen, and Social Security funds). Be sharp, analytically rigorous, and completely independent — never parrot the reform author's self-serving claims; always think for yourself.

Core Principles:
1. Zero-Base Accounting Integrity: Sum ONLY surviving/new components (abolished components strictly 0). Never add replacement levies on top of pre-reform baselines.
2. Multi-Level & Social Security Transmission: Explicitly evaluate impacts across the Federal Budget, Länder/Kommunen (KdU relief, local taxes, Konnexität), and Social Security (RV/ALV abolition or rate shifts, GKV/PV self-financing stability, employer labor cost wedge).
3. Independent Econometric Critique: Audit labor supply incentives (marginal tax-and-transfer withdrawal rates), capital formation (reinvestment tax rates vs. inheritance/capital levies), disposable household purchasing power, and realistic fiscal deficits (exposing any optimistic revenue assumptions or NIT phase-out gaps).
4. Length: Keep `executive_critique` strictly to 2-3 concise, high-level sentences. Keep each factor to 1 sentence.
Output strictly valid JSON complying with the requested schema."""


def _openai_chat_kwargs(
    model_name: str,
    desired_temperature: Optional[float] = None,
    reasoning_effort: str = "low",
) -> Dict[str, Any]:
    """
    Build kwargs for OpenAI completions.
    Models like GPT-6-Luna, o1, o3, and next-gen reasoning models reject custom temperature
    values (OpenAI returns 400). For those models, omit temperature and pass reasoning_effort
    and high max_completion_tokens.
    """
    m = (model_name or "").lower()
    kwargs: Dict[str, Any] = {}
    if any(k in m for k in ("gpt-6", "o1", "o3", "reasoning", "luna")):
        kwargs["reasoning_effort"] = reasoning_effort if reasoning_effort in ("low", "medium", "high") else "low"
        kwargs["max_completion_tokens"] = 16384
        return kwargs
    if desired_temperature is not None:
        kwargs["temperature"] = desired_temperature
    return kwargs


class UnifiedPolicyExtractor:
    """Multi-provider LLM extractor supporting OpenAI (GPT-6-Luna, GPT-4o, etc.),
    Google Gemini (gemini-3.8-flash, gemini-2.5-flash), and GLM models."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = (
            model_name
            or os.getenv("ACTIVE_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "gpt-6-luna"
        )
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL")

    def get_provider(self, model_name: Optional[str] = None) -> str:
        m = (model_name or self.model_name).lower()
        if m.startswith("gemini-"):
            return "google"
        elif m.startswith("glm-"):
            return "zhipu"
        else:
            return "openai"

    def set_model(self, model_name: str, api_key: Optional[str] = None) -> None:
        self.model_name = model_name
        if api_key:
            if self.get_provider() == "google":
                self.gemini_api_key = api_key
                os.environ["GEMINI_API_KEY"] = api_key
            else:
                self.openai_api_key = api_key
                os.environ["OPENAI_API_KEY"] = api_key
        os.environ["ACTIVE_MODEL"] = model_name

    def _build_user_turn_prompt(
        self,
        raw_prompt: str,
        current_baseline: Optional[FiscalSummary] = None,
    ) -> str:
        cur_exp = current_baseline.total_expenditure_bn if current_baseline else 555.4
        cur_rev = current_baseline.total_revenue_bn if current_baseline else 436.7
        cur_bor = current_baseline.net_borrowing_requirement_bn if current_baseline else 118.7

        rev_lines = ""
        exp_lines = ""
        if current_baseline:
            rev_lines = "\n".join(
                f"  - `{k}`: €{v:.1f}B" for k, v in current_baseline.revenue_breakdown_bn.items()
            )
            exp_lines = "\n".join(
                f"  - `{k}`: €{v:.1f}B" for k, v in current_baseline.expenditure_breakdown_bn.items()
            )

        context_str = f"""CURRENT ACTIVE BUDGET STATE (Prior to Applying This Proposal):
- Total Core Expenditures: €{cur_exp:.1f}B
- Total Non-Credit Revenues: €{cur_rev:.1f}B
- Net Borrowing Requirement (Deficit): €{cur_bor:.1f}B"""
        if rev_lines and exp_lines:
            context_str += f"""\nActive Revenues:\n{rev_lines}\nActive Expenditures:\n{exp_lines}"""

        return f"""Analyze and model the following policy proposal against the German Federal Budget.

{context_str}

USER POLICY PROPOSAL / LEGISLATIVE REFORM PACKAGE:
\"\"\"
{raw_prompt}
\"\"\"

EXECUTION DIRECTIVES:
1. EXHAUSTIVE DECONSTRUCTION: Enumerate EVERY distinct fiscal pillar, social security reform, sub-federal shift, and policy measure into `deconstruction_ledger.raw_measures_extracted`. Do not omit or truncate any part.
2. SUB-FEDERAL (LÄNDER/KOMMUNEN) & SOCIAL SECURITY (RV/ALV/GKV) INTEGRATION: Never set a fiscal measure to `0.0` just because it targets Social Security or Länder/Kommunen! When a reform integrates RV/ALV into the tax-transfer system, alters GKV/PV financing, relieves Kommunen of KdU/Pflege via a federal NES-Sockel, or modifies inheritance/corporate/property/income taxes across levels, model the consolidated fiscal flows across the canonical revenue and expenditure categories via Vertical Fiscal Equalization (*Finanzausgleich* & *Konnexität*) and omit purely non-fiscal regulatory clauses from `budget_modifications`.
3. INDEPENDENT ECONOMETRIC THINKING (ZERO AUTHOR BIAS): Always think for yourself! Never blindly copy the author's claimed revenue, cost, or deficit numbers (such as -€147.1B start-year deficit, +€42B Year-5 surplus, €48B inheritance tax despite €2M allowances and 20-year corporate deferrals, €40B subsidy revenue, or +€16.5B health fund surplus). Independently verify Negative Income Tax (NES) break-even tax-credit costs across all earners, apply realistic behavioral/deferral haircuts, and put YOUR independent estimates into `parameters.budget_modifications` (leaving `transfer_formula` null and `tax_bracket_shifts` empty `[]` when `budget_modifications` models them).
4. ACCOUNTING INTEGRITY: Certify in `deconstruction_ledger.accounting_validation_audit` that spending cuts are strictly modeled as expenditure reductions (never phantom revenues), no measure is double-counted, and explain where your independent econometric estimates correct the author's stated figures.
5. EXECUTIVE FEEDBACK: Provide a sharp, authoritative 2-4 sentence institutional executive assessment in `chat_feedback` explaining the package's multi-level/social-security structure, contrasting the document's claimed balance with your independent econometric calculation, and stating the net impact on the 2027 budget baseline."""

    def extract_policy(
        self,
        raw_prompt: str,
        enable_search_grounding: bool = False,
        reasoning_effort: str = "low",
        current_baseline: Optional[FiscalSummary] = None,
        model_override: Optional[str] = None,
    ) -> Tuple[PolicyExtractionSchema, str, List[GroundingCitation]]:
        """Extract structured policy parameters via the active LLM provider or model override."""
        active_model = model_override or self.model_name
        provider = self.get_provider(active_model)
        if provider == "google":
            return self._extract_via_gemini(
                raw_prompt,
                enable_search_grounding,
                current_baseline=current_baseline,
                model_name=active_model,
            )
        else:
            return self._extract_via_openai(
                raw_prompt,
                enable_search_grounding,
                reasoning_effort,
                current_baseline=current_baseline,
                model_name=active_model,
            )

    def _extract_via_openai(
        self,
        raw_prompt: str,
        enable_search_grounding: bool = False,
        reasoning_effort: str = "low",
        current_baseline: Optional[FiscalSummary] = None,
        model_name: Optional[str] = None,
    ) -> Tuple[PolicyExtractionSchema, str, List[GroundingCitation]]:
        """Extract policy using OpenAI model (e.g. GPT-6.1-Terra, GPT-6-Luna). Supports live OpenAI web search."""
        import json
        from openai import OpenAI

        active_model = model_name or self.model_name
        api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                f"OPENAI_API_KEY is required to run simulations with OpenAI model '{active_model}'. "
                "Please configure OPENAI_API_KEY in your .env or the admin panel."
            )

        client = OpenAI(api_key=api_key, base_url=self.openai_base_url)
        effort = reasoning_effort if reasoning_effort in ("low", "medium", "high") else "low"
        user_prompt = self._build_user_turn_prompt(raw_prompt, current_baseline)

        # 1. Live Web Search via OpenAI Responses API (web_search)
        if enable_search_grounding:
            try:
                schema_json = json.dumps(PolicyExtractionSchema.model_json_schema())
                resp = client.responses.create(
                    model=active_model,
                    input=user_prompt,
                    instructions=(
                        SYSTEM_EXTRACTION_INSTRUCTION
                        + f"\n\nCRITICAL: Respond ONLY with a valid JSON object complying exactly with this schema:\n{schema_json}"
                    ),
                    tools=[{"type": "web_search_preview"}],
                    reasoning={"effort": effort},
                )
                output_text = getattr(resp, "output_text", "") or ""
                citations: List[GroundingCitation] = []
                for item in getattr(resp, "output", []) or []:
                    if getattr(item, "type", None) == "web_search_call":
                        action = getattr(item, "action", None)
                        if action:
                            for src in getattr(action, "sources", []) or []:
                                url = getattr(src, "url", None)
                                if url and not any(c.uri == url for c in citations):
                                    citations.append(
                                        GroundingCitation(
                                            title="OpenAI Web Search Source",
                                            uri=url,
                                            snippet=f"Retrieved via OpenAI web search for: {getattr(action, 'query', '') or raw_prompt}",
                                            retrieved_via="OPENAI_WEB_SEARCH",
                                        )
                                    )
                            open_url = getattr(action, "url", None)
                            if open_url and not any(c.uri == open_url for c in citations):
                                citations.append(
                                    GroundingCitation(
                                        title="OpenAI Web Search Source",
                                        uri=open_url,
                                        snippet="Retrieved via OpenAI web search.",
                                        retrieved_via="OPENAI_WEB_SEARCH",
                                    )
                                )

                clean_json = output_text.strip()
                if clean_json.startswith("```"):
                    clean_json = clean_json.split("\n", 1)[1]
                    if clean_json.endswith("```"):
                        clean_json = clean_json.rsplit("```", 1)[0]
                    clean_json = clean_json.strip()
                parsed = PolicyExtractionSchema.model_validate_json(clean_json)
                return parsed, f"OPENAI_WEB_SEARCH_{active_model.upper()}", citations
            except Exception as search_err:
                logger.warning(
                    "OpenAI web_search response failed (%s), falling back to standard completion", search_err
                )

        # 2. Standard Structured Completion (no web search or fallback)
        call_kwargs = _openai_chat_kwargs(active_model, desired_temperature=0.1, reasoning_effort=effort)

        try:
            completion = client.beta.chat.completions.parse(
                model=active_model,
                messages=[
                    {"role": "system", "content": SYSTEM_EXTRACTION_INSTRUCTION},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=PolicyExtractionSchema,
                **call_kwargs,
            )
            parsed = completion.choices[0].message.parsed
            if not parsed:
                raise ValueError("Parsed output was null.")
            return parsed, f"OPENAI_STRUCTURED_{active_model.upper()}", []
        except Exception as e:
            err_str = str(e).lower()
            if "temperature" in err_str:
                logger.warning("Retrying beta parse for %s without temperature due to: %s", active_model, e)
                try:
                    completion = client.beta.chat.completions.parse(
                        model=active_model,
                        messages=[
                            {"role": "system", "content": SYSTEM_EXTRACTION_INSTRUCTION},
                            {"role": "user", "content": user_prompt},
                        ],
                        response_format=PolicyExtractionSchema,
                    )
                    parsed = completion.choices[0].message.parsed
                    if parsed:
                        return parsed, f"OPENAI_STRUCTURED_{active_model.upper()}", []
                except Exception as retry_err:
                    logger.warning("Retry without temperature also failed: %s", retry_err)

            logger.warning("Beta parse failed for %s (%s), falling back to json_object", active_model, e)
            fallback_kwargs = {} if "temperature" in err_str else call_kwargs
            completion = client.chat.completions.create(
                model=active_model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_EXTRACTION_INSTRUCTION
                        + "\n\nCRITICAL: Respond ONLY with a valid JSON object matching the PolicyExtractionSchema schema.",
                    },
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                **fallback_kwargs,
            )
            content = completion.choices[0].message.content or "{}"
            parsed = PolicyExtractionSchema.model_validate_json(content)
            return parsed, f"OPENAI_JSON_{active_model.upper()}", []

    def _extract_via_gemini(
        self,
        raw_prompt: str,
        enable_search_grounding: bool,
        current_baseline: Optional[FiscalSummary] = None,
        model_name: Optional[str] = None,
    ) -> Tuple[PolicyExtractionSchema, str, List[GroundingCitation]]:
        """Invoke google-genai SDK with structured output schema and low-latency thinking config."""
        from google import genai
        from google.genai import types

        active_model = model_name or self.model_name
        api_key = self.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini models.")

        client = genai.Client(api_key=api_key)
        tools = [types.Tool(google_search=types.GoogleSearch())] if enable_search_grounding else None
        user_prompt = self._build_user_turn_prompt(raw_prompt, current_baseline)

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_EXTRACTION_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=PolicyExtractionSchema,
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            tools=tools,
        )

        response = client.models.generate_content(
            model=active_model,
            contents=user_prompt,
            config=config,
        )

        if not response.text:
            raise ValueError("Empty response text returned by Gemini API.")

        extracted_schema = PolicyExtractionSchema.model_validate_json(response.text)
        citations = self._parse_grounding_metadata(response)
        return extracted_schema, "GEMINI_STRUCTURED_GROUNDED", citations

    @staticmethod
    def _parse_grounding_metadata(response: Any) -> List[GroundingCitation]:
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
        """Call active LLM to critically evaluate the reform package."""
        provider = self.get_provider()
        if provider == "google":
            return self._assess_via_gemini(applied_proposals, active_summary)
        else:
            return self._assess_via_openai(applied_proposals, active_summary)

    def _build_macro_prompt(
        self,
        applied_proposals: List[Dict[str, Any]],
        active_summary: Optional[FiscalSummary],
    ) -> str:
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

        return f"""Critically audit and assess this accumulated reform package modifying the German Federal Budget:

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
   - Factor in actual institutional parameters, macroeconomic indicators, and empirical research.

Provide:
1. `net_accounting_ledger`: Explicit zero-base accounting ledger calculating post-reform totals by setting abolished components to 0 before summing surviving/new components.
2. `assessment_title`: An executive, analytical title.
3. `executive_critique`: A sharp, internally consistent assessment strictly aligned with `net_accounting_ledger`.
4. `dynamic_gdp_growth_impact_pct`: Realistic annual real GDP delta.
5. `factors`: Key macroeconomic transmission channels reflecting the net post-reform state.
6. `suggested_line_corrections`: Specific line adjustments (using the exact `category_key` identifiers listed above) in EUR billions to correct miscalculations and reflect dynamic economic feedback."""

    def _assess_via_openai(
        self,
        applied_proposals: List[Dict[str, Any]],
        active_summary: Optional[FiscalSummary],
    ) -> MacroAssessmentCritiqueSchema:
        from openai import OpenAI

        api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                f"OPENAI_API_KEY is required for macroeconomic assessment with OpenAI model '{self.model_name}'."
            )

        client = OpenAI(api_key=api_key, base_url=self.openai_base_url)
        prompt_content = self._build_macro_prompt(applied_proposals, active_summary)
        call_kwargs = _openai_chat_kwargs(self.model_name, desired_temperature=0.2)

        try:
            completion = client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": MACRO_AUDIT_INSTRUCTION},
                    {"role": "user", "content": prompt_content},
                ],
                response_format=MacroAssessmentCritiqueSchema,
                **call_kwargs,
            )
            parsed = completion.choices[0].message.parsed
            if not parsed:
                raise ValueError("Parsed critique was null.")
            return parsed
        except Exception as e:
            err_str = str(e).lower()
            if "temperature" in err_str:
                logger.warning("Retrying macro critique for %s without temperature due to: %s", self.model_name, e)
                try:
                    completion = client.beta.chat.completions.parse(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": MACRO_AUDIT_INSTRUCTION},
                            {"role": "user", "content": prompt_content},
                        ],
                        response_format=MacroAssessmentCritiqueSchema,
                    )
                    parsed = completion.choices[0].message.parsed
                    if parsed:
                        return parsed
                except Exception as retry_err:
                    logger.warning("Retry macro critique without temperature also failed: %s", retry_err)

            logger.warning("Beta parse failed for macro critique (%s), falling back to json_object", e)
            fallback_kwargs = {} if "temperature" in err_str else call_kwargs
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": MACRO_AUDIT_INSTRUCTION
                        + "\n\nRespond ONLY with a valid JSON object complying with the MacroAssessmentCritiqueSchema schema.",
                    },
                    {"role": "user", "content": prompt_content},
                ],
                response_format={"type": "json_object"},
                **fallback_kwargs,
            )
            content = completion.choices[0].message.content or "{}"
            return MacroAssessmentCritiqueSchema.model_validate_json(content)

    def _assess_via_gemini(
        self,
        applied_proposals: List[Dict[str, Any]],
        active_summary: Optional[FiscalSummary],
    ) -> MacroAssessmentCritiqueSchema:
        from google import genai
        from google.genai import types

        api_key = self.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini models.")

        client = genai.Client(api_key=api_key)
        prompt_content = self._build_macro_prompt(applied_proposals, active_summary)
        tools = [types.Tool(google_search=types.GoogleSearch())]

        config = types.GenerateContentConfig(
            system_instruction=MACRO_AUDIT_INSTRUCTION,
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


# Singleton extractor instance to allow dynamic model switching across requests
_extractor_instance = UnifiedPolicyExtractor()


# FastAPI Dependency Providers
def get_simulator() -> FiscalSimulator:
    return FiscalSimulator(baseline=GERMANY_2027_DRAFT_BASELINE)


def get_policy_extractor() -> UnifiedPolicyExtractor:
    return _extractor_instance


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


class DocumentExtractRequest(BaseModel):
    filename: str
    content_base64: str


def extract_text_from_document(filename: str, content_bytes: bytes) -> str:
    """Extract readable text from PDF, DOCX, DOC, or TXT."""
    name_lower = (filename or "").lower()
    if name_lower.endswith(".pdf"):
        reader = pypdf.PdfReader(io.BytesIO(content_bytes))
        pages_text: List[str] = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages_text.append(t)
        return "\n\n".join(pages_text)
    elif name_lower.endswith(".docx") or name_lower.endswith(".doc"):
        doc = docx.Document(io.BytesIO(content_bytes))
        paragraphs: List[str] = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    paragraphs.append(row_text)
        return "\n\n".join(paragraphs)
    else:
        try:
            return content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return content_bytes.decode("latin-1", errors="replace")


@app.post("/api/v1/extract-document", tags=["Document"])
async def extract_document_endpoint(req: DocumentExtractRequest) -> Dict[str, Any]:
    """Extract plain text from an uploaded PDF, DOCX, or TXT file."""
    try:
        content_bytes = base64.b64decode(req.content_base64)
        extracted = extract_text_from_document(req.filename, content_bytes)
        clean = extracted.strip()
        if not clean:
            raise ValueError("No readable text found in uploaded document.")
        return {
            "status": "SUCCESS",
            "filename": req.filename,
            "text": clean,
            "char_count": len(clean),
        }
    except Exception as exc:
        logger.warning("Document extraction failed for '%s': %s", req.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to extract document: {exc}",
        ) from exc


class ModelConfigRequest(BaseModel):
    model_name: str
    api_key: Optional[str] = None
    provider: Optional[str] = None


@app.get("/health", tags=["System"])
async def health_check(extractor: UnifiedPolicyExtractor = Depends(get_policy_extractor)) -> Dict[str, Any]:
    """Liveness and readiness probe for the Python simulation microservice."""
    has_gemini_key = bool(extractor.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    has_openai_key = bool(extractor.openai_api_key or os.getenv("OPENAI_API_KEY"))
    return {
        "status": "healthy",
        "service": "fiscal_ai_engine",
        "version": "2.0.0",
        "model": extractor.model_name,
        "provider": extractor.get_provider(),
        "baseline": "Germany Federal Budget 2027 Government Draft (Regierungsentwurf 2027)",
        "openai_api_configured": has_openai_key,
        "gemini_api_configured": has_gemini_key,
        "supported_countries": ["DE"],
    }


@app.get("/api/v1/model", tags=["Model"])
async def get_model_info(
    extractor: UnifiedPolicyExtractor = Depends(get_policy_extractor),
) -> Dict[str, Any]:
    """Return active model configuration, provider, and available selectable models."""
    return {
        "active_model": extractor.model_name,
        "provider": extractor.get_provider(),
        "available_models": [
            {
                "id": "gpt-6.1-terra",
                "name": "GPT-6.1-Terra (OpenAI)",
                "provider": "openai",
                "description": "Frontier high-capacity document processing model specialized in multi-page packages",
                "recommended": True,
            },
            {
                "id": "gpt-6-luna",
                "name": "GPT-6-Luna (OpenAI)",
                "provider": "openai",
                "description": "Standard European-compliant reasoning model with OpenAI web search",
                "recommended": False,
            },
            {
                "id": "gemini-3.8-flash",
                "name": "Gemini 3.8 Flash (Google)",
                "provider": "google",
                "description": "Google DeepMind structured outputs with live Google Search grounding",
                "recommended": False,
            },
        ],
        "has_openai_key": bool(extractor.openai_api_key or os.getenv("OPENAI_API_KEY")),
        "has_gemini_key": bool(extractor.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
    }


@app.post("/api/v1/model", tags=["Model"])
async def set_model_info(
    req: ModelConfigRequest,
    extractor: UnifiedPolicyExtractor = Depends(get_policy_extractor),
) -> Dict[str, Any]:
    """Dynamically switch the active LLM model and optional API credentials."""
    extractor.set_model(req.model_name, req.api_key)
    logger.info("Switched active model to %s (%s)", extractor.model_name, extractor.get_provider())
    return {
        "status": "UPDATED",
        "active_model": extractor.model_name,
        "provider": extractor.get_provider(),
        "has_openai_key": bool(extractor.openai_api_key or os.getenv("OPENAI_API_KEY")),
        "has_gemini_key": bool(extractor.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
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
    """Interpret a policy proposal and compute 2027 budget impacts."""
    try:
        model_to_use = request.model_override or extractor.model_name
        extracted_policy, extraction_mode, grounded_citations = extractor.extract_policy(
            raw_prompt=request.raw_prompt,
            enable_search_grounding=request.enable_search_grounding,
            reasoning_effort=request.reasoning_effort or "low",
            current_baseline=request.current_baseline_override,
            model_override=model_to_use,
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
            model_used=model_to_use,
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
