"""Deterministic Microsimulation Core for GovSim AI.

Grounded in the German Federal Budget 2027 Government Draft (Regierungsentwurf 2027,
Bundestags-Drucksache 21/7300):
- Total Core Expenditures: €555.4B
- Total Non-Credit Revenues: €436.7B (including €398.6B federal tax revenues)
- Net Borrowing Requirement (Nettokreditaufnahme): €118.7B

Evaluates policy proposals decomposed into universal fiscal primitives
(`BudgetLineModification`, `TaxBracketShift`, `TransferFormulaConfig`),
computes dual vertical stacked bar payloads (Revenues vs. Expenditures),
and generates a complete Before/After comparison table.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Dict, List, Optional, Tuple
import uuid
import numpy as np

logger = logging.getLogger("fiscal_simulator")

from .schemas import (
    ApplyActionMetadata,
    BarSegment,
    ComparisonTableRow,
    DecileImpact,
    DualBarChartPayload,
    FiscalDeltaSummary,
    FiscalSummary,
    GroundingCitation,
    LineItemDelta,
    PolicyExtractionSchema,
    SankeyGraphPayload,
    SankeyLink,
    SankeyNode,
    SimulationMetadata,
    SimulationResult,
    TransferFormulaConfig,
)


@dataclass(frozen=True)
class CountryFiscalBaseline:
    """Macroeconomic and distributional baseline data for a specific country & year."""

    country_code: str
    fiscal_year: int
    adult_population_millions: float
    # 10 deciles: mean annual gross market income per adult in EUR
    decile_mean_incomes_eur: np.ndarray
    # 10 deciles: effective baseline income tax rates
    decile_effective_tax_rates: np.ndarray
    # Baseline federal revenue categories in EUR billions
    revenues_bn: Dict[str, float]
    # Baseline federal expenditure categories in EUR billions
    expenditures_bn: Dict[str, float]


# Official German Federal Budget 2027 Government Draft (Regierungsentwurf 2027)
# Total Expenditures: €555.4B | Non-Credit Revenues: €436.7B | Net Borrowing: €118.7B
GERMANY_2027_DRAFT_BASELINE = CountryFiscalBaseline(
    country_code="DE",
    fiscal_year=2027,
    adult_population_millions=68.8,  # ~68.8M adults in 2027 projection
    decile_mean_incomes_eur=np.array(
        [
            5_100.0,    # Decile 1: lowest 10%
            12_200.0,   # Decile 2
            19_400.0,   # Decile 3
            26_100.0,   # Decile 4
            32_800.0,   # Decile 5
            40_900.0,   # Decile 6
            50_100.0,   # Decile 7
            62_400.0,   # Decile 8
            81_200.0,   # Decile 9
            142_500.0,  # Decile 10: top 10%
        ],
        dtype=np.float64,
    ),
    decile_effective_tax_rates=np.array(
        [0.00, 0.015, 0.046, 0.084, 0.118, 0.151, 0.185, 0.222, 0.268, 0.345],
        dtype=np.float64,
    ),
    # Total Non-Credit Revenues = €436.7B (Tax Revenues: €398.6B + Non-Tax Revenues: €38.1B)
    revenues_bn={
        "income_and_wealth_taxes": 162.4,         # Federal share of Wage, Assessed Income, Solidarity & Withholding Taxes
        "value_added_and_consumption_taxes": 146.8, # Federal share of VAT (Umsatzsteuer) & Import VAT
        "corporate_and_business_taxes": 44.2,     # Federal share of Corporate Tax (Körperschaftsteuer) & Trade Tax
        "energy_and_environmental_taxes": 45.2,   # Energy Tax, Electricity Tax, Aviation, Tobacco & Insurance Taxes
        "non_tax_and_other_revenues": 38.1,       # Bundesbank profit, CO2/Highway tolls, administrative fees & EU inflows
    },
    # Total Core Budget Expenditures (Regierungsentwurf 2027) = €555.4B
    expenditures_bn={
        "labor_and_social_affairs_ep11": 201.5,   # Einzelplan 11 (BMAS): Pensions (~€133B), Citizen's Benefit (Bürgergeld), Labor Market
        "defense_and_security_ep14": 109.7,       # Einzelplan 14 (BMVg): Federal Defense Budget (+€27.1B surge in 2027 Draft)
        "federal_debt_interest_ep32": 43.6,       # Einzelplan 32 (Bundesschuld): Federal Debt Service & Interest Payments
        "transport_and_digital_ep12": 26.4,       # Einzelplan 12 (BMV): Federal Rail, Roads & Digital Infrastructure (Core Budget)
        "research_tech_and_aerospace_ep30": 22.0, # Einzelplan 30 (BMFTR): Science, AI, Innovation & Aerospace
        "family_education_and_youth_ep17": 15.5,  # Einzelplan 17 (BMBFSFJ): Family Benefits, Elterngeld, Youth & Education
        "economic_affairs_energy_climate_ep09_16": 8.3, # Einzelplan 09 & 16 (BMWE + BMUKN): Economic Policy, Energy & Environment
        "general_financial_admin_and_other": 128.4, # Einzelplan 60, 06, 08, 15, 23: General Financial Admin, Health, Interior, Development (BMZ €9.5B) & Subsidies
    },
)

# Backwards-compatible alias
GERMANY_2025_BASELINE = GERMANY_2027_DRAFT_BASELINE


CATEGORY_DISPLAY_NAMES: Dict[str, str] = {
    # Revenues
    "income_and_wealth_taxes": "Income & Solidarity Taxes (Federal Share)",
    "value_added_and_consumption_taxes": "VAT & Consumption Taxes",
    "corporate_and_business_taxes": "Corporate & Business Taxes",
    "energy_and_environmental_taxes": "Energy, Carbon & Excise Taxes",
    "non_tax_and_other_revenues": "Non-Tax, Tolls & Administrative Revenues",
    # Expenditures (2027 Government Draft Einzelpläne)
    "labor_and_social_affairs_ep11": "Labor & Social Affairs — Pensions & Welfare (EP 11)",
    "defense_and_security_ep14": "Defense & Military Procurement (EP 14)",
    "federal_debt_interest_ep32": "Federal Debt Interest Service (EP 32)",
    "transport_and_digital_ep12": "Transport, Rail & Digital Infrastructure (EP 12)",
    "research_tech_and_aerospace_ep30": "Research, AI, Tech & Aerospace (EP 30)",
    "family_education_and_youth_ep17": "Family, Education, Seniors & Youth (EP 17)",
    "economic_affairs_energy_climate_ep09_16": "Economic Affairs, Energy & Environment (EP 09/16)",
    "general_administration_and_other": "General Financial Admin, Health, Interior & Subsidies",
    "general_financial_admin_and_other": "General Financial Admin, Health, Interior & Subsidies",
}


class FiscalSimulator:
    """Deterministic microsimulation engine for the 2027 German Federal Budget Draft."""

    def __init__(self, baseline: CountryFiscalBaseline = GERMANY_2027_DRAFT_BASELINE) -> None:
        self.baseline = baseline

    def get_initial_summary(self) -> FiscalSummary:
        """Construct the default un-modified 2027 Government Draft FiscalSummary."""
        rev_map = {k: round(v, 3) for k, v in self.baseline.revenues_bn.items()}
        exp_map = {k: round(v, 3) for k, v in self.baseline.expenditures_bn.items()}
        total_rev = round(sum(rev_map.values()), 3)
        total_exp = round(sum(exp_map.values()), 3)
        balance = round(total_rev - total_exp, 3)
        borrowing = round(max(0.0, -balance), 3)
        return FiscalSummary(
            total_revenue_bn=total_rev,
            total_expenditure_bn=total_exp,
            fiscal_balance_bn=balance,
            net_borrowing_requirement_bn=borrowing,
            revenue_breakdown_bn=rev_map,
            expenditure_breakdown_bn=exp_map,
        )

    def build_bar_chart_for_summary(
        self,
        initial_summary: FiscalSummary,
        before_summary: FiscalSummary,
        after_summary: FiscalSummary,
        custom_labels: Optional[Dict[str, str]] = None,
    ) -> DualBarChartPayload:
        """Build the Dual Vertical Bar Chart payload (Revenues vs. Expenditures) and Before/After Table."""
        labels = dict(CATEGORY_DISPLAY_NAMES)
        if custom_labels:
            labels.update(custom_labels)

        revenue_bar: List[BarSegment] = []
        expenditure_bar: List[BarSegment] = []
        comparison_table: List[ComparisonTableRow] = []

        # Total heights for percentage share calculations
        total_rev_bar_height = max(
            after_summary.total_revenue_bn + after_summary.net_borrowing_requirement_bn, 1e-6
        )
        ref_surplus = max(0.0, after_summary.fiscal_balance_bn)
        total_exp_bar_height = max(after_summary.total_expenditure_bn + ref_surplus, 1e-6)

        # 1. Revenue Categories
        all_rev_keys = list(
            dict.fromkeys(
                list(initial_summary.revenue_breakdown_bn.keys())
                + list(before_summary.revenue_breakdown_bn.keys())
                + list(after_summary.revenue_breakdown_bn.keys())
            )
        )
        for rev_key in all_rev_keys:
            init_val = round(initial_summary.revenue_breakdown_bn.get(rev_key, 0.0), 3)
            bef_val = round(before_summary.revenue_breakdown_bn.get(rev_key, 0.0), 3)
            aft_val = round(after_summary.revenue_breakdown_bn.get(rev_key, 0.0), 3)
            if init_val <= 0.0 and bef_val <= 0.0 and aft_val <= 0.0:
                continue

            delta_bef = round(aft_val - bef_val, 3)
            delta_bef_pct = round((delta_bef / bef_val) * 100.0, 2) if bef_val > 0 else 0.0
            delta_init = round(aft_val - init_val, 3)
            share_pct = round((aft_val / total_rev_bar_height) * 100.0, 2)
            name = labels.get(rev_key, rev_key.replace("_", " ").title())

            revenue_bar.append(
                BarSegment(
                    id=f"rev_{rev_key}",
                    category_key=rev_key,
                    name=name,
                    side="revenue",
                    segment_type="regular",
                    initial_2027_bn=init_val,
                    before_bn=bef_val,
                    after_bn=aft_val,
                    delta_bn=delta_bef,
                    share_pct=share_pct,
                )
            )
            comparison_table.append(
                ComparisonTableRow(
                    side="revenue",
                    category_key=rev_key,
                    display_name=name,
                    initial_2027_draft_bn=init_val,
                    before_bn=bef_val,
                    after_bn=aft_val,
                    delta_vs_before_bn=delta_bef,
                    delta_vs_before_pct=delta_bef_pct,
                    delta_vs_initial_bn=delta_init,
                    share_of_total_pct=share_pct,
                )
            )

        # Net Borrowing (Deficit Financing) on the Revenue Bar so both bars balance visually
        init_borrow = round(initial_summary.net_borrowing_requirement_bn, 3)
        bef_borrow = round(before_summary.net_borrowing_requirement_bn, 3)
        aft_borrow = round(after_summary.net_borrowing_requirement_bn, 3)
        if init_borrow > 0.0 or bef_borrow > 0.0 or aft_borrow > 0.0:
            delta_borrow = round(aft_borrow - bef_borrow, 3)
            delta_borrow_pct = (
                round((delta_borrow / bef_borrow) * 100.0, 2) if bef_borrow > 0 else 0.0
            )
            share_borrow = round((aft_borrow / total_rev_bar_height) * 100.0, 2)
            revenue_bar.append(
                BarSegment(
                    id="rev_net_borrowing_deficit",
                    category_key="net_borrowing_deficit",
                    name="Net Borrowing / Fiscal Deficit (Nettokreditaufnahme)",
                    side="revenue",
                    segment_type="deficit_borrowing",
                    initial_2027_bn=init_borrow,
                    before_bn=bef_borrow,
                    after_bn=aft_borrow,
                    delta_bn=delta_borrow,
                    share_pct=share_borrow,
                )
            )
            comparison_table.append(
                ComparisonTableRow(
                    side="financing",
                    category_key="net_borrowing_deficit",
                    display_name="Net Borrowing / Fiscal Deficit (Nettokreditaufnahme)",
                    initial_2027_draft_bn=init_borrow,
                    before_bn=bef_borrow,
                    after_bn=aft_borrow,
                    delta_vs_before_bn=delta_borrow,
                    delta_vs_before_pct=delta_borrow_pct,
                    delta_vs_initial_bn=round(aft_borrow - init_borrow, 3),
                    share_of_total_pct=share_borrow,
                )
            )

        # 2. Expenditure Categories
        all_exp_keys = list(
            dict.fromkeys(
                list(initial_summary.expenditure_breakdown_bn.keys())
                + list(before_summary.expenditure_breakdown_bn.keys())
                + list(after_summary.expenditure_breakdown_bn.keys())
            )
        )
        for exp_key in all_exp_keys:
            init_val = round(initial_summary.expenditure_breakdown_bn.get(exp_key, 0.0), 3)
            bef_val = round(before_summary.expenditure_breakdown_bn.get(exp_key, 0.0), 3)
            aft_val = round(after_summary.expenditure_breakdown_bn.get(exp_key, 0.0), 3)
            if init_val <= 0.0 and bef_val <= 0.0 and aft_val <= 0.0:
                continue

            delta_bef = round(aft_val - bef_val, 3)
            delta_bef_pct = round((delta_bef / bef_val) * 100.0, 2) if bef_val > 0 else 0.0
            delta_init = round(aft_val - init_val, 3)
            share_pct = round((aft_val / total_exp_bar_height) * 100.0, 2)
            name = labels.get(exp_key, exp_key.replace("_", " ").title())
            seg_type = (
                "transfer"
                if any(
                    k in exp_key
                    for k in ("social", "pension", "welfare", "transfer", "dividend", "grant", "bonus")
                )
                else "regular"
            )

            expenditure_bar.append(
                BarSegment(
                    id=f"exp_{exp_key}",
                    category_key=exp_key,
                    name=name,
                    side="expenditure",
                    segment_type=seg_type,  # type: ignore[arg-type]
                    initial_2027_bn=init_val,
                    before_bn=bef_val,
                    after_bn=aft_val,
                    delta_bn=delta_bef,
                    share_pct=share_pct,
                )
            )
            comparison_table.append(
                ComparisonTableRow(
                    side="expenditure",
                    category_key=exp_key,
                    display_name=name,
                    initial_2027_draft_bn=init_val,
                    before_bn=bef_val,
                    after_bn=aft_val,
                    delta_vs_before_bn=delta_bef,
                    delta_vs_before_pct=delta_bef_pct,
                    delta_vs_initial_bn=delta_init,
                    share_of_total_pct=share_pct,
                )
            )

        # Fiscal Surplus segment if Revenues > Expenditures
        init_surplus = round(max(0.0, initial_summary.fiscal_balance_bn), 3)
        bef_surplus = round(max(0.0, before_summary.fiscal_balance_bn), 3)
        aft_surplus = round(max(0.0, after_summary.fiscal_balance_bn), 3)
        if init_surplus > 0.0 or bef_surplus > 0.0 or aft_surplus > 0.0:
            delta_surp = round(aft_surplus - bef_surplus, 3)
            delta_surp_pct = (
                round((delta_surp / bef_surplus) * 100.0, 2) if bef_surplus > 0 else 0.0
            )
            share_surp = round((aft_surplus / total_exp_bar_height) * 100.0, 2)
            expenditure_bar.append(
                BarSegment(
                    id="exp_fiscal_surplus_reserve",
                    category_key="fiscal_surplus_reserve",
                    name="Fiscal Surplus / Debt Repayment",
                    side="expenditure",
                    segment_type="surplus_reserve",
                    initial_2027_bn=init_surplus,
                    before_bn=bef_surplus,
                    after_bn=aft_surplus,
                    delta_bn=delta_surp,
                    share_pct=share_surp,
                )
            )
            comparison_table.append(
                ComparisonTableRow(
                    side="financing",
                    category_key="fiscal_surplus_reserve",
                    display_name="Fiscal Surplus / Debt Repayment",
                    initial_2027_draft_bn=init_surplus,
                    before_bn=bef_surplus,
                    after_bn=aft_surplus,
                    delta_vs_before_bn=delta_surp,
                    delta_vs_before_pct=delta_surp_pct,
                    delta_vs_initial_bn=round(aft_surplus - init_surplus, 3),
                    share_of_total_pct=share_surp,
                )
            )

        return DualBarChartPayload(
            revenue_bar=revenue_bar,
            expenditure_bar=expenditure_bar,
            comparison_table=comparison_table,
        )

    def build_sankey_for_summary(
        self,
        reference_summary: FiscalSummary,
        active_summary: FiscalSummary,
        custom_labels: Optional[Dict[str, str]] = None,
    ) -> SankeyGraphPayload:
        """Public helper to build a D3.js Sankey diagram payload."""
        return self._build_sankey_payload(
            baseline_summary=reference_summary,
            reformed_summary=active_summary,
            custom_labels=custom_labels or {},
        )

    def evaluate_decile_distribution(
        self,
        transfer_formula: Optional[TransferFormulaConfig],
        tax_rate_deltas: Dict[int, float],
        current_income_tax_total_bn: float,
    ) -> Tuple[List[DecileImpact], float, float]:
        """Evaluate household transfers/dividends and income tax rate shifts across all 10 deciles."""
        decile_pop_millions = self.baseline.adult_population_millions / 10.0
        incomes_y = self.baseline.decile_mean_incomes_eur
        base_tax_rates = self.baseline.decile_effective_tax_rates

        per_capita_transfers = np.zeros_like(incomes_y)
        if transfer_formula and transfer_formula.base_monthly_grant_eur > 0.0:
            g_annual = 12.0 * transfer_formula.base_monthly_grant_eur
            t_rate = float(np.clip(transfer_formula.income_withdrawal_rate, 0.0, 1.0))
            raw_transfers = np.maximum(0.0, g_annual - t_rate * incomes_y)
            for idx in range(10):
                if (idx + 1) <= transfer_formula.max_eligible_decile:
                    per_capita_transfers[idx] = raw_transfers[idx]

        decile_transfer_costs_bn = (per_capita_transfers * decile_pop_millions) / 1_000.0

        reformed_tax_rates = base_tax_rates.copy()
        for decile_idx, delta_pct in tax_rate_deltas.items():
            if 1 <= decile_idx <= 10:
                reformed_tax_rates[decile_idx - 1] = np.clip(
                    reformed_tax_rates[decile_idx - 1] + delta_pct, 0.0, 0.75
                )

        raw_base_taxes_bn = (incomes_y * base_tax_rates * decile_pop_millions) / 1_000.0
        scale_factor = (
            current_income_tax_total_bn / float(np.sum(raw_base_taxes_bn))
            if float(np.sum(raw_base_taxes_bn)) > 0
            else 1.0
        )
        baseline_decile_taxes_bn = raw_base_taxes_bn * scale_factor
        reformed_decile_taxes_bn = (
            (incomes_y * reformed_tax_rates * decile_pop_millions) / 1_000.0
        ) * scale_factor

        decile_impacts: List[DecileImpact] = []
        for idx in range(10):
            decile_num = idx + 1
            transfer_per_cap = float(per_capita_transfers[idx])
            tax_delta_bn = float(reformed_decile_taxes_bn[idx] - baseline_decile_taxes_bn[idx])
            tax_delta_per_cap = (tax_delta_bn * 1_000.0) / decile_pop_millions
            net_disp_delta = transfer_per_cap - tax_delta_per_cap

            decile_impacts.append(
                DecileImpact(
                    decile=decile_num,
                    population_millions=round(decile_pop_millions, 3),
                    mean_annual_gross_income_eur=round(float(incomes_y[idx]), 2),
                    annual_transfer_per_capita_eur=round(transfer_per_cap, 2),
                    total_decile_transfer_cost_bn=round(float(decile_transfer_costs_bn[idx]), 3),
                    baseline_income_tax_bn=round(float(baseline_decile_taxes_bn[idx]), 3),
                    reformed_income_tax_bn=round(float(reformed_decile_taxes_bn[idx]), 3),
                    net_disposable_income_delta_eur=round(float(net_disp_delta), 2),
                )
            )

        total_transfer_cost_bn = float(np.sum(decile_transfer_costs_bn))
        total_tax_delta_bn = float(
            np.sum(reformed_decile_taxes_bn) - np.sum(baseline_decile_taxes_bn)
        )
        return (
            decile_impacts,
            round(total_transfer_cost_bn, 3),
            round(total_tax_delta_bn, 3),
        )

    def run_simulation(
        self,
        raw_prompt: str,
        extracted_policy: PolicyExtractionSchema,
        proposal_id: Optional[str] = None,
        current_baseline_override: Optional[FiscalSummary] = None,
        extraction_mode: str = "GEMINI_STRUCTURED_GROUNDED",
        model_used: str = "gemini-3.8-flash",
        extra_citations: Optional[List[GroundingCitation]] = None,
    ) -> SimulationResult:
        """Execute deterministic calculation for a chat proposal and prepare 'Apply' payload."""
        active_proposal_id = proposal_id or f"prop_{uuid.uuid4().hex[:12]}"
        params = extracted_policy.parameters

        # 1. Establish Initial 2027 Draft & Current Active Baseline (Before state)
        initial_2027_summary = self.get_initial_summary()
        baseline_summary = (
            current_baseline_override.model_copy(deep=True)
            if current_baseline_override is not None
            else initial_2027_summary.model_copy(deep=True)
        )

        reformed_rev_map = dict(baseline_summary.revenue_breakdown_bn)
        reformed_exp_map = dict(baseline_summary.expenditure_breakdown_bn)
        custom_labels: Dict[str, str] = dict(CATEGORY_DISPLAY_NAMES)

        # 2. Evaluate Decile Transfers & Income Tax Rate Shifts
        tax_rate_deltas = {
            shift.decile: shift.rate_delta_pct for shift in params.tax_bracket_shifts
        }
        current_inc_tax = reformed_rev_map.get("income_and_wealth_taxes", 162.4)
        (
            decile_impacts,
            transfer_program_cost_bn,
            decile_tax_delta_bn,
        ) = self.evaluate_decile_distribution(
            transfer_formula=params.transfer_formula,
            tax_rate_deltas=tax_rate_deltas,
            current_income_tax_total_bn=current_inc_tax,
        )

        if abs(decile_tax_delta_bn) > 1e-4:
            reformed_rev_map["income_and_wealth_taxes"] = round(
                max(0.0, reformed_rev_map.get("income_and_wealth_taxes", 0.0) + decile_tax_delta_bn),
                3,
            )

        if transfer_program_cost_bn > 0.0 and params.transfer_formula is not None:
            transfer_key = self._slugify_key(params.transfer_formula.program_label)
            custom_labels[transfer_key] = params.transfer_formula.program_label
            reformed_exp_map[transfer_key] = round(
                reformed_exp_map.get(transfer_key, 0.0) + transfer_program_cost_bn, 3
            )

        # 3. Defensive Deduplication & Normalization of Budget Line Modifications
        formulas_applied: List[str] = []
        has_transfer_formula = (
            params.transfer_formula is not None
            and params.transfer_formula.base_monthly_grant_eur > 0.0
            and transfer_program_cost_bn > 0.0
        )
        transfer_slug = (
            self._slugify_key(params.transfer_formula.program_label)
            if has_transfer_formula and params.transfer_formula is not None
            else ""
        )
        has_tax_shifts = bool(params.tax_bracket_shifts and abs(decile_tax_delta_bn) > 1e-4)

        sanitized_modifications = []
        seen_measure_fingerprints = set()

        for raw_mod in params.budget_modifications:
            mod = raw_mod.model_copy()
            mod_slug = self._slugify_key(mod.category_key)
            mod_disp_slug = self._slugify_key(mod.display_name or "")
            text_desc = f"{mod.display_name} {mod.rationale} {mod.category_key}".lower()

            # A. Deduplicate against transfer_formula (avoid charging transfer costs twice)
            if has_transfer_formula and (
                mod_slug == transfer_slug
                or mod_disp_slug == transfer_slug
                or ("transfer" in mod_slug and "transfer" in transfer_slug)
                or ("dividend" in mod_slug and "dividend" in transfer_slug)
                or ("buergergeld" in mod_slug and "buergergeld" in transfer_slug)
            ):
                logger.info("Deduplicating redundant BudgetLineModification for transfer formula: %s", mod.display_name)
                continue

            # B. Deduplicate against tax_bracket_shifts (only if exact duplicate of decile_tax_delta_bn)
            if has_tax_shifts and mod.category_key in ("income_and_wealth_taxes", "income_taxes", "wealth_taxes"):
                if abs(mod.value - decile_tax_delta_bn) < 0.75:
                    logger.info("Deduplicating redundant BudgetLineModification for income tax bracket shifts: %s", mod.display_name)
                    continue

            # C. Correct Phantom Revenues: pure spending/admin cuts misclassified as positive public revenue
            disp_lower = (mod.display_name or "").lower()
            if mod.target_side == "revenue" and mod.value > 0:
                is_tax_Measure = any(
                    tk in disp_lower or tk in text_desc
                    for tk in (
                        "tax",
                        "steuer",
                        "exemption",
                        "kerosin",
                        "kerosene",
                        "dienstwagen",
                        "company car",
                        "agrardiesel",
                        "diesel",
                        "allowance",
                        "freibetrag",
                        "loophole",
                        "privilege",
                    )
                )
                is_pure_spending_cut = not is_tax_Measure and any(
                    k in disp_lower
                    for k in (
                        "cut subsidies",
                        "subsidy cut",
                        "subsidies cut",
                        "admin cut",
                        "overhead trim",
                        "spending cut",
                        "ausgabenkürzung",
                        "verwaltungskürzung",
                    )
                )
                if is_pure_spending_cut:
                    logger.info("Correcting misclassified phantom revenue to expenditure cut: %s", mod.display_name)
                    mod.target_side = "expenditure"
                    mod.category_key = "general_financial_admin_and_other"
                    mod.value = -abs(mod.value)

            # D. Sign Normalization: Only fix positive values when display_name unambiguously denotes a cut/relief
            # Never flip an intentional negative value (mod.value < 0) to positive!
            if mod.operation == "DELTA_BN" and mod.value > 0:
                has_increase_word = any(
                    k in disp_lower
                    for k in (
                        "increase",
                        "boost",
                        "expand",
                        "funding",
                        "invest",
                        "raise",
                        "hike",
                        "levy",
                        "surcharge",
                        "net",
                        "new",
                        "addition",
                        "reallocation",
                        "ausbau",
                        "erhöhung",
                        "investition",
                        "abgabe",
                        "mehr",
                    )
                )
                if not has_increase_word:
                    if mod.target_side == "expenditure":
                        is_cut_title = any(
                            k in disp_lower
                            for k in ("cut", "reduction", "savings", "trim", "curtail", "slash", "abolish", "kürzung", "einsparung", "abbau")
                        )
                        if is_cut_title:
                            logger.info("Normalizing positive expenditure cut value to negative: %s (%s -> %s)", mod.display_name, mod.value, -mod.value)
                            mod.value = -abs(mod.value)
                    elif mod.target_side == "revenue":
                        is_relief_title = any(
                            k in disp_lower
                            for k in ("tax cut", "tax relief", "rate cut", "rate reduction", "steuersenkung", "steuerentlastung")
                        )
                        if is_relief_title:
                            logger.info("Normalizing positive tax relief value to negative: %s (%s -> %s)", mod.display_name, mod.value, -mod.value)
                            mod.value = -abs(mod.value)

            # E. Deduplicate identical duplicate line modifications
            target_map_check = reformed_rev_map if mod.target_side == "revenue" else reformed_exp_map
            resolved_temp_key = self._resolve_category_key(mod.category_key, target_map_check)
            fingerprint = f"{mod.target_side}:{resolved_temp_key}:{round(mod.value, 2)}:{mod.operation}"
            if fingerprint in seen_measure_fingerprints:
                logger.info("Deduplicating identical duplicate modification: %s", fingerprint)
                continue
            seen_measure_fingerprints.add(fingerprint)

            sanitized_modifications.append(mod)

        # Apply sanitized modifications (order-independent accumulation before non-negative clamping)
        for mod in sanitized_modifications:
            target_map = (
                reformed_rev_map if mod.target_side == "revenue" else reformed_exp_map
            )
            resolved_key = self._resolve_category_key(mod.category_key, target_map)
            if mod.display_name:
                custom_labels.setdefault(resolved_key, mod.display_name)

            prior_val = target_map.get(resolved_key, 0.0)
            if mod.operation == "DELTA_BN":
                new_val = prior_val + mod.value
                formulas_applied.append(
                    f"\\text{{{custom_labels.get(resolved_key, resolved_key)}}}: {prior_val:.1f}\\text{{B}} + ({mod.value:+.1f}\\text{{B}}) = {new_val:.1f}\\text{{B}}"
                )
            elif mod.operation == "DELTA_PCT":
                delta_bn = prior_val * mod.value
                new_val = prior_val + delta_bn
                formulas_applied.append(
                    f"\\text{{{custom_labels.get(resolved_key, resolved_key)}}}: {prior_val:.1f}\\text{{B}} \\times (1 + {mod.value:+.1%}) = {new_val:.1f}\\text{{B}}"
                )
            elif mod.operation == "SET_ABSOLUTE_BN":
                new_val = max(0.0, mod.value)
                formulas_applied.append(
                    f"\\text{{{custom_labels.get(resolved_key, resolved_key)}}} \\leftarrow {new_val:.1f}\\text{{B}}"
                )
            elif mod.operation == "ABOLISH":
                new_val = max(0.0, prior_val - abs(mod.value)) if mod.value != 0.0 else 0.0
                formulas_applied.append(
                    f"\\text{{Abolish/Reduce {custom_labels.get(resolved_key, resolved_key)}}}: {prior_val:.1f}\\text{{B}} \\rightarrow {new_val:.1f}\\text{{B}}"
                )
            elif mod.operation == "NEW_LINE_BN":
                new_val = prior_val + abs(mod.value)
                custom_labels[resolved_key] = mod.display_name or resolved_key.replace("_", " ").title()
                formulas_applied.append(
                    f"\\text{{New Program ({custom_labels[resolved_key]})}}: +€{abs(mod.value):.1f}\\text{{B/yr}}"
                )
            else:
                new_val = prior_val

            target_map[resolved_key] = round(new_val, 3)

        # Clamp final category totals to non-negative after all modifications have accumulated
        for k in list(reformed_rev_map.keys()):
            reformed_rev_map[k] = round(max(0.0, reformed_rev_map[k]), 3)
        for k in list(reformed_exp_map.keys()):
            reformed_exp_map[k] = round(max(0.0, reformed_exp_map[k]), 3)

        if transfer_program_cost_bn > 0.0 and params.transfer_formula is not None:
            g_ann = params.transfer_formula.base_monthly_grant_eur * 12.0
            formulas_applied.append(
                f"T(Y_d) = \\max\\left(0, \\text{{€}}{g_ann:,.0f} - {params.transfer_formula.income_withdrawal_rate:.2f} \\cdot Y_d\\right) \\implies \\text{{€}}{transfer_program_cost_bn:.1f}\\text{{B}}"
            )

        if abs(decile_tax_delta_bn) > 1e-4:
            formulas_applied.append(
                f"\\Delta R_{{\\text{{income tax}}}} = \\sum_{{d=1}}^{{10}} N_d \\cdot Y_d \\cdot \\Delta \\tau_d = {decile_tax_delta_bn:+.2f}\\text{{B}}"
            )

        # 4. Compute Reformed Fiscal Summary (After state) & Line Item Deltas
        reformed_total_rev = round(sum(reformed_rev_map.values()), 3)
        reformed_total_exp = round(sum(reformed_exp_map.values()), 3)
        reformed_balance = round(reformed_total_rev - reformed_total_exp, 3)
        reformed_borrowing = round(max(0.0, -reformed_balance), 3)

        reformed_summary = FiscalSummary(
            total_revenue_bn=reformed_total_rev,
            total_expenditure_bn=reformed_total_exp,
            fiscal_balance_bn=reformed_balance,
            net_borrowing_requirement_bn=reformed_borrowing,
            revenue_breakdown_bn=reformed_rev_map,
            expenditure_breakdown_bn=reformed_exp_map,
        )

        line_item_deltas: List[LineItemDelta] = []
        for k in dict.fromkeys(list(baseline_summary.revenue_breakdown_bn.keys()) + list(reformed_rev_map.keys())):
            b_val = baseline_summary.revenue_breakdown_bn.get(k, 0.0)
            r_val = reformed_rev_map.get(k, 0.0)
            if abs(r_val - b_val) > 1e-3:
                d_bn = round(r_val - b_val, 3)
                d_pct = round((d_bn / b_val) * 100.0, 2) if b_val > 0 else 0.0
                line_item_deltas.append(
                    LineItemDelta(
                        side="revenue",
                        category_key=k,
                        display_name=custom_labels.get(k, k.replace("_", " ").title()),
                        baseline_bn=round(b_val, 3),
                        proposed_bn=round(r_val, 3),
                        delta_bn=d_bn,
                        delta_pct=d_pct,
                    )
                )

        for k in dict.fromkeys(list(baseline_summary.expenditure_breakdown_bn.keys()) + list(reformed_exp_map.keys())):
            b_val = baseline_summary.expenditure_breakdown_bn.get(k, 0.0)
            r_val = reformed_exp_map.get(k, 0.0)
            if abs(r_val - b_val) > 1e-3:
                d_bn = round(r_val - b_val, 3)
                d_pct = round((d_bn / b_val) * 100.0, 2) if b_val > 0 else 0.0
                line_item_deltas.append(
                    LineItemDelta(
                        side="expenditure",
                        category_key=k,
                        display_name=custom_labels.get(k, k.replace("_", " ").title()),
                        baseline_bn=round(b_val, 3),
                        proposed_bn=round(r_val, 3),
                        delta_bn=d_bn,
                        delta_pct=d_pct,
                    )
                )

        rev_delta_bn = round(reformed_total_rev - baseline_summary.total_revenue_bn, 3)
        exp_delta_bn = round(reformed_total_exp - baseline_summary.total_expenditure_bn, 3)
        net_budget_impact_bn = round(reformed_balance - baseline_summary.fiscal_balance_bn, 3)

        formulas_applied.append(
            f"\\Delta B = \\Delta R ({rev_delta_bn:+.1f}\\text{{B}}) - \\Delta E ({exp_delta_bn:+.1f}\\text{{B}}) = {net_budget_impact_bn:+.1f}\\text{{B}}"
        )

        fiscal_delta = FiscalDeltaSummary(
            total_revenue_delta_bn=rev_delta_bn,
            total_expenditure_delta_bn=exp_delta_bn,
            net_budget_impact_bn=net_budget_impact_bn,
            transfer_program_cost_bn=transfer_program_cost_bn,
            line_item_deltas=line_item_deltas,
        )

        # 5. Build Dual Vertical Bar Chart & Before/After Comparison Table
        bar_chart = self.build_bar_chart_for_summary(
            initial_summary=initial_2027_summary,
            before_summary=baseline_summary,
            after_summary=reformed_summary,
            custom_labels=custom_labels,
        )

        # 6. Build Sankey Payload for API compatibility
        sankey_graph = self._build_sankey_payload(
            baseline_summary=baseline_summary,
            reformed_summary=reformed_summary,
            custom_labels=custom_labels,
        )

        chat_feedback = extracted_policy.chat_feedback
        if not chat_feedback:
            direction = "reduces the fiscal deficit" if net_budget_impact_bn >= 0 else "increases the net borrowing requirement"
            chat_feedback = (
                f"Modeled **{extracted_policy.policy_name}** against the 2027 Federal Budget Draft: "
                f"revenues shift by €{rev_delta_bn:+.1f}B and expenditures by €{exp_delta_bn:+.1f}B, which {direction} by €{abs(net_budget_impact_bn):.1f}B/year."
            )

        apply_action = ApplyActionMetadata(
            can_apply=True,
            status="PROPOSED",
            button_label="Apply to Budget",
            apply_endpoint=f"/api/scenarios/{active_proposal_id}/apply",
        )

        assumptions = list(extracted_policy.extracted_assumptions)
        assumptions.append(
            f"Evaluated against the German Federal Budget 2027 Government Draft (Regierungsentwurf 2027: €555.4B core expenditures, €118.7B net borrowing, {self.baseline.adult_population_millions:.1f}M adults)."
        )

        metadata = SimulationMetadata(
            extraction_mode=extraction_mode,  # type: ignore[arg-type]
            model_used=model_used,
            formulas_applied=formulas_applied,
            assumptions=assumptions,
            citations=self._compile_citations(extra_citations),
        )

        return SimulationResult(
            proposal_id=active_proposal_id,
            country_code=self.baseline.country_code,
            fiscal_year=self.baseline.fiscal_year,
            raw_prompt=raw_prompt,
            chat_feedback=chat_feedback,
            extracted_policy=extracted_policy,
            initial_2027_summary=initial_2027_summary,
            baseline_summary=baseline_summary,
            reformed_summary=reformed_summary,
            fiscal_delta=fiscal_delta,
            bar_chart=bar_chart,
            decile_impacts=decile_impacts,
            sankey_graph=sankey_graph,
            apply_action=apply_action,
            metadata=metadata,
        )

    @staticmethod
    def _slugify_key(label: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
        return slug or "custom_policy_line"

    @classmethod
    def _resolve_category_key(cls, raw_key: str, existing_map: Dict[str, float]) -> str:
        """Match an AI-supplied category key to an existing 2027 budget category or create a clean key."""
        slug = cls._slugify_key(raw_key)
        if slug in existing_map:
            return slug

        keyword_aliases = {
            "income": "income_and_wealth_taxes",
            "wealth": "income_and_wealth_taxes",
            "solidarity": "income_and_wealth_taxes",
            "vat": "value_added_and_consumption_taxes",
            "consumption": "value_added_and_consumption_taxes",
            "sales": "value_added_and_consumption_taxes",
            "corporate": "corporate_and_business_taxes",
            "business": "corporate_and_business_taxes",
            "carbon": "energy_and_environmental_taxes",
            "energy": "energy_and_environmental_taxes",
            "environmental": "energy_and_environmental_taxes",
            "excise": "energy_and_environmental_taxes",
            "non_tax": "non_tax_and_other_revenues",
            "toll": "non_tax_and_other_revenues",
            "ep11": "labor_and_social_affairs_ep11",
            "pension": "labor_and_social_affairs_ep11",
            "labor": "labor_and_social_affairs_ep11",
            "social": "labor_and_social_affairs_ep11",
            "welfare": "labor_and_social_affairs_ep11",
            "buergergeld": "labor_and_social_affairs_ep11",
            "ep14": "defense_and_security_ep14",
            "defense": "defense_and_security_ep14",
            "defence": "defense_and_security_ep14",
            "military": "defense_and_security_ep14",
            "bundeswehr": "defense_and_security_ep14",
            "ep32": "federal_debt_interest_ep32",
            "interest": "federal_debt_interest_ep32",
            "debt": "federal_debt_interest_ep32",
            "ep12": "transport_and_digital_ep12",
            "transport": "transport_and_digital_ep12",
            "rail": "transport_and_digital_ep12",
            "digital": "transport_and_digital_ep12",
            "infrastructure": "transport_and_digital_ep12",
            "ep30": "research_tech_and_aerospace_ep30",
            "research": "research_tech_and_aerospace_ep30",
            "science": "research_tech_and_aerospace_ep30",
            "tech": "research_tech_and_aerospace_ep30",
            "ai": "research_tech_and_aerospace_ep30",
            "innovation": "research_tech_and_aerospace_ep30",
            "aerospace": "research_tech_and_aerospace_ep30",
            "ep17": "family_education_and_youth_ep17",
            "family": "family_education_and_youth_ep17",
            "education": "family_education_and_youth_ep17",
            "youth": "family_education_and_youth_ep17",
            "child": "family_education_and_youth_ep17",
            "ep09": "economic_affairs_energy_climate_ep09_16",
            "ep16": "economic_affairs_energy_climate_ep09_16",
            "economic": "economic_affairs_energy_climate_ep09_16",
            "climate": "economic_affairs_energy_climate_ep09_16",
            "green": "economic_affairs_energy_climate_ep09_16",
            "admin": "general_financial_admin_and_other",
            "bureaucracy": "general_financial_admin_and_other",
            "subsid": "general_financial_admin_and_other",
            "development": "general_financial_admin_and_other",
            "health": "general_financial_admin_and_other",
        }
        for kw, target in keyword_aliases.items():
            if kw in slug and target in existing_map:
                return target
        return slug

    def _build_sankey_payload(
        self,
        baseline_summary: FiscalSummary,
        reformed_summary: FiscalSummary,
        custom_labels: Dict[str, str],
    ) -> SankeyGraphPayload:
        """Construct a balanced D3.js Sankey diagram payload."""
        nodes: List[SankeyNode] = []
        links: List[SankeyLink] = []

        pool_id = "federal_budget_pool"
        base_pool_total = max(
            baseline_summary.total_revenue_bn, baseline_summary.total_expenditure_bn
        )
        reformed_pool_total = max(
            reformed_summary.total_revenue_bn, reformed_summary.total_expenditure_bn
        )

        all_rev_keys = list(
            dict.fromkeys(
                list(baseline_summary.revenue_breakdown_bn.keys())
                + list(reformed_summary.revenue_breakdown_bn.keys())
            )
        )
        for rev_key in all_rev_keys:
            base_val = baseline_summary.revenue_breakdown_bn.get(rev_key, 0.0)
            reformed_val = reformed_summary.revenue_breakdown_bn.get(rev_key, 0.0)
            if reformed_val <= 0.0 and base_val <= 0.0:
                continue
            node_id = f"rev_{rev_key}"
            label = custom_labels.get(rev_key, rev_key.replace("_", " ").title())
            delta = round(reformed_val - base_val, 3)

            nodes.append(
                SankeyNode(
                    id=node_id,
                    name=label,
                    category="revenue",
                    column=0,
                    baseline_value_bn=round(base_val, 3),
                    reformed_value_bn=round(reformed_val, 3),
                    delta_bn=delta,
                )
            )
            if reformed_val > 0.0:
                links.append(
                    SankeyLink(
                        source=node_id,
                        target=pool_id,
                        value=round(reformed_val, 3),
                        baseline_value=round(base_val, 3),
                        delta_bn=delta,
                        flow_type="revenue_inflow",
                        label=f"{label} → Federal Budget (€{reformed_val:.1f}B)",
                    )
                )

        if (
            reformed_summary.net_borrowing_requirement_bn > 0.0
            or baseline_summary.net_borrowing_requirement_bn > 0.0
        ):
            base_borrow = baseline_summary.net_borrowing_requirement_bn
            ref_borrow = reformed_summary.net_borrowing_requirement_bn
            delta_borrow = round(ref_borrow - base_borrow, 3)
            borrow_id = "rev_net_borrowing_deficit"
            nodes.append(
                SankeyNode(
                    id=borrow_id,
                    name="Net Borrowing / Fiscal Deficit",
                    category="deficit_surplus",
                    column=0,
                    baseline_value_bn=round(base_borrow, 3),
                    reformed_value_bn=round(ref_borrow, 3),
                    delta_bn=delta_borrow,
                )
            )
            if ref_borrow > 0.0:
                links.append(
                    SankeyLink(
                        source=borrow_id,
                        target=pool_id,
                        value=round(ref_borrow, 3),
                        baseline_value=round(base_borrow, 3),
                        delta_bn=delta_borrow,
                        flow_type="deficit_financing",
                        label=f"Net Borrowing → Federal Budget (€{ref_borrow:.1f}B)",
                    )
                )

        nodes.append(
            SankeyNode(
                id=pool_id,
                name="2027 Federal Budget Pool",
                category="budget_pool",
                column=1,
                baseline_value_bn=round(base_pool_total, 3),
                reformed_value_bn=round(reformed_pool_total, 3),
                delta_bn=round(reformed_pool_total - base_pool_total, 3),
            )
        )

        all_exp_keys = list(
            dict.fromkeys(
                list(baseline_summary.expenditure_breakdown_bn.keys())
                + list(reformed_summary.expenditure_breakdown_bn.keys())
            )
        )
        for exp_key in all_exp_keys:
            base_val = baseline_summary.expenditure_breakdown_bn.get(exp_key, 0.0)
            reformed_val = reformed_summary.expenditure_breakdown_bn.get(exp_key, 0.0)
            if reformed_val <= 0.0 and base_val <= 0.0:
                continue

            node_id = f"exp_{exp_key}"
            label = custom_labels.get(exp_key, exp_key.replace("_", " ").title())
            delta = round(reformed_val - base_val, 3)
            node_cat = (
                "transfer"
                if any(
                    k in exp_key
                    for k in ("social", "pension", "welfare", "transfer", "dividend", "grant", "bonus")
                )
                else "expenditure"
            )

            nodes.append(
                SankeyNode(
                    id=node_id,
                    name=label,
                    category=node_cat,  # type: ignore[arg-type]
                    column=2,
                    baseline_value_bn=round(base_val, 3),
                    reformed_value_bn=round(reformed_val, 3),
                    delta_bn=delta,
                )
            )
            if reformed_val > 0.0:
                links.append(
                    SankeyLink(
                        source=pool_id,
                        target=node_id,
                        value=round(reformed_val, 3),
                        baseline_value=round(base_val, 3),
                        delta_bn=delta,
                        flow_type="budget_allocation",
                        label=f"Federal Budget → {label} (€{reformed_val:.1f}B)",
                    )
                )

        base_surplus = max(0.0, baseline_summary.fiscal_balance_bn)
        ref_surplus = max(0.0, reformed_summary.fiscal_balance_bn)
        if ref_surplus > 0.0 or base_surplus > 0.0:
            surplus_id = "exp_fiscal_surplus_reserve"
            delta_surplus = round(ref_surplus - base_surplus, 3)
            nodes.append(
                SankeyNode(
                    id=surplus_id,
                    name="Fiscal Surplus / Debt Repayment",
                    category="deficit_surplus",
                    column=2,
                    baseline_value_bn=round(base_surplus, 3),
                    reformed_value_bn=round(ref_surplus, 3),
                    delta_bn=delta_surplus,
                )
            )
            if ref_surplus > 0.0:
                links.append(
                    SankeyLink(
                        source=pool_id,
                        target=surplus_id,
                        value=round(ref_surplus, 3),
                        baseline_value=round(base_surplus, 3),
                        delta_bn=delta_surplus,
                        flow_type="surplus_reserve",
                        label=f"Federal Budget → Fiscal Surplus (€{ref_surplus:.1f}B)",
                    )
                )

        return SankeyGraphPayload(nodes=nodes, links=links)

    @staticmethod
    def _compile_citations(
        extra_citations: Optional[List[GroundingCitation]] = None,
    ) -> List[GroundingCitation]:
        """Return official 2027 Government Draft baseline citations merged with live search grounding citations."""
        baseline_citations = [
            GroundingCitation(
                title="German Federal Ministry of Finance (BMF) – Government Draft for the 2027 Federal Budget (Regierungsentwurf 2027, Drucksache 21/7300)",
                uri="https://www.bundeshaushalt.de/",
                snippet="Core budget volume of €555.4B, non-credit revenues of €436.7B (tax revenues €398.6B), and net borrowing requirement of €118.7B across Einzelpläne 11, 14, 32, 12, 30, 17, 09/16.",
                retrieved_via="OFFICIAL_BASELINE_REGISTRY",
            ),
            GroundingCitation(
                title="Federal Statistical Office (Destatis) – Adult Income Distribution & Tax Burden Deciles",
                uri="https://www.destatis.de/EN/Themes/Society-Environment/Income-Consumption-Living-Conditions/_node.html",
                snippet="German adult population (~68.8M) gross market income deciles and effective income tax rates.",
                retrieved_via="OFFICIAL_BASELINE_REGISTRY",
            ),
        ]
        if extra_citations:
            seen_uris = {c.uri for c in baseline_citations}
            for citation in extra_citations:
                if citation.uri not in seen_uris:
                    baseline_citations.append(citation)
                    seen_uris.add(citation.uri)
        return baseline_citations
