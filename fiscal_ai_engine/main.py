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
Speed & Precision Directive: Be concise, decisive, and institutional. All outputs must be in crisp English.

Baseline: German Federal Budget 2027 Draft (Regierungsentwurf, €555.4B Exp / €436.7B Rev / €118.7B Deficit / 68.8M adults).

Revenue Categories:
- `income_and_wealth_taxes` (€162.4B) | `value_added_and_consumption_taxes` (€146.8B) | `corporate_and_business_taxes` (€44.2B) | `energy_and_environmental_taxes` (€45.2B) | `non_tax_and_other_revenues` (€38.1B)

Expenditure Categories (2027 Einzelpläne):
- `labor_and_social_affairs_ep11` (€201.5B) | `defense_and_security_ep14` (€109.7B) | `federal_debt_interest_ep32` (€43.6B) | `transport_and_digital_ep12` (€26.4B) | `research_tech_and_aerospace_ep30` (€22.0B) | `family_education_and_youth_ep17` (€15.5B) | `economic_affairs_energy_climate_ep09_16` (€8.3B) | `general_financial_admin_and_other` (€128.4B)

Extraction Rules (High-Speed Execution):
1. `chat_feedback`: Strictly 1-2 concise, executive sentences. State the mapped budget action and resulting net borrowing effect. No conversational filler.
2. `policy_name`: Punchy title under 5 words.
3. `budget_modifications`: Output ONLY the modified line items using:
   - `DELTA_BN` (+/- EUR billions), `DELTA_PCT` (decimal percent, e.g. -0.05), `SET_ABSOLUTE_BN`, `ABOLISH` (value in savings), or `NEW_LINE_BN`.
4. `rationale` & `extracted_assumptions`: 1 short sentence each.
5. If decile tax rates shift (1-10), populate `parameters.tax_bracket_shifts`. If per-capita transfer/dividend, populate `parameters.transfer_formula`.
Respond strictly in valid JSON complying with the schema."""


MACRO_AUDIT_INSTRUCTION = """You are an Independent Senior Fiscal Economist and Macroeconomic Auditor.
Assess the reform package on the German Federal Budget. Be sharp, analytically rigorous, and concise.

Core Principles:
1. Zero-Base Accounting Integrity: Sum ONLY surviving/new components (abolished components strictly 0). Never add replacement levies on top of pre-reform baselines.
2. Macro Transmission: Focus on disposable wages, domestic consumption, and dynamic feedback.
3. Length: Keep `executive_critique` strictly to 2-3 concise, high-level sentences. Keep each factor to 1 sentence.
Output strictly valid JSON complying with the requested schema."""


def _openai_chat_kwargs(model_name: str, desired_temperature: Optional[float] = None) -> Dict[str, Any]:
    """
    Build kwargs for OpenAI completions.
    Models like GPT-6-Luna, o1, o3, and next-gen reasoning models reject custom temperature
    values (OpenAI returns 400). For those models, omit temperature and pass reasoning_effort="low"
    to maximize response speed.
    """
    m = (model_name or "").lower()
    kwargs: Dict[str, Any] = {}
    if any(k in m for k in ("gpt-6", "o1", "o3", "reasoning", "luna")):
        kwargs["reasoning_effort"] = "low"
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

    def get_provider(self) -> str:
        m = self.model_name.lower()
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

    def extract_policy(
        self, raw_prompt: str, enable_search_grounding: bool = False
    ) -> Tuple[PolicyExtractionSchema, str, List[GroundingCitation]]:
        """Extract structured policy parameters via the active LLM provider."""
        provider = self.get_provider()
        if provider == "google":
            return self._extract_via_gemini(raw_prompt, enable_search_grounding)
        else:
            return self._extract_via_openai(raw_prompt, enable_search_grounding)

    def _extract_via_openai(
        self, raw_prompt: str, enable_search_grounding: bool = False
    ) -> Tuple[PolicyExtractionSchema, str, List[GroundingCitation]]:
        """Extract policy using OpenAI model (e.g. GPT-6-Luna). Supports live OpenAI web search."""
        import json
        from openai import OpenAI

        api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                f"OPENAI_API_KEY is required to run simulations with OpenAI model '{self.model_name}'. "
                "Please configure OPENAI_API_KEY in your .env or the admin panel."
            )

        client = OpenAI(api_key=api_key, base_url=self.openai_base_url)

        # 1. Live Web Search via OpenAI Responses API (web_search)
        if enable_search_grounding:
            try:
                schema_json = json.dumps(PolicyExtractionSchema.model_json_schema())
                resp = client.responses.create(
                    model=self.model_name,
                    input=raw_prompt,
                    instructions=(
                        SYSTEM_EXTRACTION_INSTRUCTION
                        + f"\n\nCRITICAL: Respond ONLY with a valid JSON object complying exactly with this schema:\n{schema_json}"
                    ),
                    tools=[{"type": "web_search_preview"}],
                    reasoning={"effort": "low"},
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
                return parsed, f"OPENAI_WEB_SEARCH_{self.model_name.upper()}", citations
            except Exception as search_err:
                logger.warning(
                    "OpenAI web_search response failed (%s), falling back to standard completion", search_err
                )

        # 2. Standard Structured Completion (no web search or fallback)
        call_kwargs = _openai_chat_kwargs(self.model_name, desired_temperature=0.1)

        try:
            completion = client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_EXTRACTION_INSTRUCTION},
                    {"role": "user", "content": raw_prompt},
                ],
                response_format=PolicyExtractionSchema,
                **call_kwargs,
            )
            parsed = completion.choices[0].message.parsed
            if not parsed:
                raise ValueError("Parsed output was null.")
            return parsed, f"OPENAI_STRUCTURED_{self.model_name.upper()}", []
        except Exception as e:
            err_str = str(e).lower()
            if "temperature" in err_str:
                logger.warning("Retrying beta parse for %s without temperature due to: %s", self.model_name, e)
                try:
                    completion = client.beta.chat.completions.parse(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": SYSTEM_EXTRACTION_INSTRUCTION},
                            {"role": "user", "content": raw_prompt},
                        ],
                        response_format=PolicyExtractionSchema,
                    )
                    parsed = completion.choices[0].message.parsed
                    if parsed:
                        return parsed, f"OPENAI_STRUCTURED_{self.model_name.upper()}", []
                except Exception as retry_err:
                    logger.warning("Retry without temperature also failed: %s", retry_err)

            logger.warning("Beta parse failed for %s (%s), falling back to json_object", self.model_name, e)
            fallback_kwargs = {} if "temperature" in err_str else call_kwargs
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_EXTRACTION_INSTRUCTION
                        + "\n\nCRITICAL: Respond ONLY with a valid JSON object matching the PolicyExtractionSchema schema.",
                    },
                    {"role": "user", "content": raw_prompt},
                ],
                response_format={"type": "json_object"},
                **fallback_kwargs,
            )
            content = completion.choices[0].message.content or "{}"
            parsed = PolicyExtractionSchema.model_validate_json(content)
            return parsed, f"OPENAI_JSON_{self.model_name.upper()}", []

    def _extract_via_gemini(
        self, raw_prompt: str, enable_search_grounding: bool
    ) -> Tuple[PolicyExtractionSchema, str, List[GroundingCitation]]:
        """Invoke google-genai SDK with structured output schema and low-latency thinking config."""
        from google import genai
        from google.genai import types

        api_key = self.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini models.")

        client = genai.Client(api_key=api_key)
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
                "id": "gpt-6-luna",
                "name": "GPT-6-Luna (OpenAI)",
                "provider": "openai",
                "description": "High-knowledge European-compliant reasoning model with OpenAI web search",
                "recommended": True,
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
