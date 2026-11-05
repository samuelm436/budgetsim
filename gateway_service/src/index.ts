/**
 * GovSim AI — API Gateway & Scenario Orchestrator (`gateway_service`)
 *
 * Microservice 2: Node.js + TypeScript Fastify Gateway
 * Orchestrates the English-language Policy Chat -> Calculation Preview -> "Apply to Budget"
 * workflow against the Python `fiscal_ai_engine` (German Federal Budget 2027 Government Draft).
 */

import crypto from "crypto";
import path from "path";
import Fastify, { FastifyInstance, FastifyReply, FastifyRequest } from "fastify";
import cors from "@fastify/cors";
import fastifyStatic from "@fastify/static";

// ============================================================================
// Typed Interfaces (Mirrors Python OpenAPI / Pydantic Schemas)
// ============================================================================

export type PolicyDomain =
  | "SOCIAL_AND_TRANSFERS"
  | "TAXATION_AND_REVENUE"
  | "INFRASTRUCTURE_AND_CLIMATE"
  | "DEFENSE_AND_SECURITY"
  | "EDUCATION_AND_INNOVATION"
  | "STRUCTURAL_CONSOLIDATION"
  | "MULTI_DOMAIN_REFORM";

export interface TaxBracketShift {
  decile: number;
  rate_delta_pct: number;
}

export interface TransferFormulaConfig {
  program_label: string;
  base_monthly_grant_eur: number;
  income_withdrawal_rate: number;
  max_eligible_decile: number;
}

export interface BudgetLineModification {
  target_side: "revenue" | "expenditure";
  category_key: string;
  display_name: string;
  operation:
    | "DELTA_BN"
    | "DELTA_PCT"
    | "SET_ABSOLUTE_BN"
    | "ABOLISH"
    | "NEW_LINE_BN";
  value: number;
  rationale?: string;
}

export interface PolicyParameters {
  budget_modifications?: BudgetLineModification[];
  tax_bracket_shifts?: TaxBracketShift[];
  transfer_formula?: TransferFormulaConfig | null;
}

export interface ScenarioRunRequest {
  raw_prompt: string;
  country_code?: "DE";
  fiscal_year?: number;
  build_on_active_diagram?: boolean;
  override_parameters?: PolicyParameters;
  enable_search_grounding?: boolean;
}

export interface FiscalSummary {
  total_revenue_bn: number;
  total_expenditure_bn: number;
  fiscal_balance_bn: number;
  net_borrowing_requirement_bn: number;
  revenue_breakdown_bn: Record<string, number>;
  expenditure_breakdown_bn: Record<string, number>;
}

export interface LineItemDelta {
  side: "revenue" | "expenditure";
  category_key: string;
  display_name: string;
  baseline_bn: number;
  proposed_bn: number;
  delta_bn: number;
  delta_pct: number;
}

export interface ComparisonTableRow {
  side: "revenue" | "expenditure" | "financing";
  category_key: string;
  display_name: string;
  initial_2027_draft_bn: number;
  before_bn: number;
  after_bn: number;
  delta_vs_before_bn: number;
  delta_vs_before_pct: number;
  delta_vs_initial_bn: number;
  share_of_total_pct: number;
}

export interface BarSegment {
  id: string;
  category_key: string;
  name: string;
  side: "revenue" | "expenditure";
  segment_type: "regular" | "deficit_borrowing" | "surplus_reserve" | "transfer";
  initial_2027_bn: number;
  before_bn: number;
  after_bn: number;
  delta_bn: number;
  share_pct: number;
}

export interface DualBarChartPayload {
  revenue_bar: BarSegment[];
  expenditure_bar: BarSegment[];
  comparison_table: ComparisonTableRow[];
  unit: "EUR_BILLIONS";
}

export interface FiscalDeltaSummary {
  total_revenue_delta_bn: number;
  total_expenditure_delta_bn: number;
  net_budget_impact_bn: number;
  transfer_program_cost_bn: number;
  line_item_deltas: LineItemDelta[];
}

export interface DecileImpact {
  decile: number;
  population_millions: number;
  mean_annual_gross_income_eur: number;
  annual_transfer_per_capita_eur: number;
  total_decile_transfer_cost_bn: number;
  baseline_income_tax_bn: number;
  reformed_income_tax_bn: number;
  net_disposable_income_delta_eur: number;
}

export interface GroundingCitation {
  title: string;
  uri: string;
  snippet?: string | null;
  retrieved_via: "GOOGLE_SEARCH_GROUNDING" | "OFFICIAL_BASELINE_REGISTRY";
}

export interface ApplyActionMetadata {
  can_apply: boolean;
  status: "PROPOSED" | "APPLIED";
  button_label: string;
  apply_endpoint: string;
}

export interface SankeyGraphPayload {
  nodes: unknown[];
  links: unknown[];
  unit: "EUR_BILLIONS";
}

export interface SimulationResult {
  proposal_id: string;
  country_code: string;
  fiscal_year: number;
  raw_prompt: string;
  chat_feedback: string;
  extracted_policy: {
    policy_name: string;
    policy_domain: PolicyDomain;
    chat_feedback: string;
    parameters: PolicyParameters;
    grounding_queries: string[];
    extracted_assumptions: string[];
  };
  initial_2027_summary: FiscalSummary;
  baseline_summary: FiscalSummary;
  reformed_summary: FiscalSummary;
  fiscal_delta: FiscalDeltaSummary;
  bar_chart: DualBarChartPayload;
  decile_impacts: DecileImpact[];
  sankey_graph: SankeyGraphPayload;
  apply_action: ApplyActionMetadata;
  metadata: {
    extraction_mode: "GEMINI_STRUCTURED_GROUNDED" | "DIRECT_OVERRIDE";
    model_used: string;
    formulas_applied: string[];
    assumptions: string[];
    citations: GroundingCitation[];
    timestamp_utc: string;
  };
}

export interface ChatCalculationCard {
  proposal_id: string;
  title: string;
  policy_domain: PolicyDomain;
  chat_feedback: string;
  total_revenue_delta_bn: number;
  total_expenditure_delta_bn: number;
  net_budget_impact_bn: number;
  line_item_deltas: LineItemDelta[];
  apply_button: ApplyActionMetadata;
}

export interface GatewayProposalEnvelope {
  proposal_id: string;
  status: "PROPOSED" | "APPLIED";
  gateway_received_at: string;
  applied_at?: string;
  execution_time_ms: number;
  chat_reply: {
    feedback_text: string;
    calculation_card: ChatCalculationCard;
  };
  simulation: SimulationResult;
}

export interface AppliedProposalRecord {
  proposal_id: string;
  policy_name: string;
  policy_domain: PolicyDomain;
  raw_prompt: string;
  short_description: string;
  revenue_delta_bn: number;
  expenditure_delta_bn: number;
  net_budget_impact_bn: number;
  line_item_deltas: LineItemDelta[];
  applied_at: string;
  calculation_card?: ChatCalculationCard;
}

export interface ActiveBudgetState {
  country_code: string;
  fiscal_year: number;
  baseline_label: string;
  initial_2027_summary: FiscalSummary | null;
  active_summary: FiscalSummary | null;
  active_bar_chart: DualBarChartPayload | null;
  active_sankey_graph: SankeyGraphPayload | null;
  applied_proposals: AppliedProposalRecord[];
}

export interface ApiErrorResponse {
  error: {
    code: string;
    message: string;
    proposal_id?: string;
    details?: unknown;
  };
}

// ============================================================================
// Gateway Configuration & State Registry
// ============================================================================

const PORT = Number(process.env.PORT ?? 3000);
const FISCAL_ENGINE_URL =
  process.env.FISCAL_ENGINE_URL ?? "http://localhost:8000";
const UPSTREAM_TIMEOUT_MS = Number(process.env.UPSTREAM_TIMEOUT_MS ?? 120000);

const proposalRegistry = new Map<string, GatewayProposalEnvelope>();

const activeBudgetState: ActiveBudgetState = {
  country_code: "DE",
  fiscal_year: 2027,
  baseline_label:
    "German Federal Budget 2027 — Government Draft (Regierungsentwurf 2027)",
  initial_2027_summary: null,
  active_summary: null,
  active_bar_chart: null,
  active_sankey_graph: null,
  applied_proposals: [],
};

// ============================================================================
// Fastify Server Instance
// ============================================================================

export const server: FastifyInstance = Fastify({
  logger: process.env.NODE_ENV === "production",
  bodyLimit: 1048576, // 1MB
});

// Alias for backwards compatibility
export const app = server;

const publicDir = path.join(__dirname, "..", "public");

// ============================================================================
// Helper Utilities & Errors
// ============================================================================

class GatewayHttpError extends Error {
  public readonly statusCode: number;
  public readonly errorCode: string;
  public readonly details?: unknown;

  constructor(
    statusCode: number,
    errorCode: string,
    message: string,
    details?: unknown
  ) {
    super(message);
    this.statusCode = statusCode;
    this.errorCode = errorCode;
    this.details = details;
  }
}

function generateProposalId(): string {
  const randomSuffix = crypto.randomBytes(6).toString("hex");
  return `prop_${Date.now().toString(36)}_${randomSuffix}`;
}

function validateScenarioRequest(body: unknown): ScenarioRunRequest {
  if (!body || typeof body !== "object") {
    throw new GatewayHttpError(
      400,
      "INVALID_PAYLOAD",
      "Request body must be a valid JSON object."
    );
  }

  const candidate = body as Record<string, unknown>;
  const rawPrompt = candidate.raw_prompt;

  if (typeof rawPrompt !== "string" || rawPrompt.trim().length < 3) {
    throw new GatewayHttpError(
      400,
      "VALIDATION_ERROR",
      "Field 'raw_prompt' is required and must be at least 3 characters."
    );
  }

  if (rawPrompt.length > 4000) {
    throw new GatewayHttpError(
      400,
      "VALIDATION_ERROR",
      "Field 'raw_prompt' must not exceed 4,000 characters."
    );
  }

  const countryCode =
    candidate.country_code !== undefined ? candidate.country_code : "DE";
  if (countryCode !== "DE") {
    throw new GatewayHttpError(
      400,
      "UNSUPPORTED_COUNTRY",
      "Only country_code 'DE' (Germany) is supported in the 2027 baseline."
    );
  }

  const fiscalYear =
    candidate.fiscal_year !== undefined ? Number(candidate.fiscal_year) : 2027;
  if (
    !Number.isInteger(fiscalYear) ||
    fiscalYear < 2024 ||
    fiscalYear > 2030
  ) {
    throw new GatewayHttpError(
      400,
      "INVALID_FISCAL_YEAR",
      "Field 'fiscal_year' must be an integer between 2024 and 2030."
    );
  }

  return {
    raw_prompt: rawPrompt.trim(),
    country_code: "DE",
    fiscal_year: fiscalYear,
    build_on_active_diagram:
      typeof candidate.build_on_active_diagram === "boolean"
        ? candidate.build_on_active_diagram
        : true,
    override_parameters: candidate.override_parameters as
      | PolicyParameters
      | undefined,
    enable_search_grounding:
      typeof candidate.enable_search_grounding === "boolean"
        ? candidate.enable_search_grounding
        : false,
  };
}

async function ensureInitialBudgetLoaded(): Promise<void> {
  if (
    activeBudgetState.active_summary &&
    activeBudgetState.active_bar_chart
  ) {
    return;
  }
  const res = await fetch(`${FISCAL_ENGINE_URL}/api/v1/baseline`);
  if (!res.ok) {
    throw new GatewayHttpError(
      502,
      "BASELINE_FETCH_FAILED",
      `Failed to fetch 2027 baseline from ${FISCAL_ENGINE_URL}/api/v1/baseline`
    );
  }
  const data = (await res.json()) as {
    baseline_label: string;
    baseline_summary: FiscalSummary;
    bar_chart: DualBarChartPayload;
    sankey_graph: SankeyGraphPayload;
  };
  activeBudgetState.baseline_label = data.baseline_label;
  activeBudgetState.initial_2027_summary = data.baseline_summary;
  activeBudgetState.active_summary = data.baseline_summary;
  activeBudgetState.active_bar_chart = data.bar_chart;
  activeBudgetState.active_sankey_graph = data.sankey_graph;
}

// ============================================================================
// Route Handlers
// ============================================================================

/**
 * GET /health
 * Liveness & upstream microservice health check.
 */
server.get("/health", async (_request: FastifyRequest, reply: FastifyReply) => {
  let upstreamStatus = "unreachable";
  let upstreamDetails: unknown = null;

  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 3000);
    const upstreamRes = await fetch(`${FISCAL_ENGINE_URL}/health`, {
      signal: controller.signal,
    });
    clearTimeout(timer);

    if (upstreamRes.ok) {
      upstreamStatus = "healthy";
      upstreamDetails = await upstreamRes.json();
    } else {
      upstreamStatus = `degraded (HTTP ${upstreamRes.status})`;
    }
  } catch (err) {
    upstreamStatus = "unreachable";
    upstreamDetails = err instanceof Error ? err.message : String(err);
  }

  return reply.status(200).send({
    status: "healthy",
    service: "gateway_service",
    version: "2.0.0",
    architecture: "Microservices (Fastify Gateway -> Python FastAPI Fiscal Engine)",
    stored_proposals_count: proposalRegistry.size,
    applied_changes_count: activeBudgetState.applied_proposals.length,
    upstream_engine: {
      url: FISCAL_ENGINE_URL,
      status: upstreamStatus,
      details: upstreamDetails,
    },
  });
});

/**
 * POST /api/scenarios/run (and POST /api/chat/propose)
 * Step 1: User submits a policy reform idea in Chat.
 * Forwards to `fiscal_ai_engine` and returns executive feedback + calculation card + Apply button.
 */
async function handleProposeScenario(
  request: FastifyRequest,
  reply: FastifyReply
): Promise<void> {
  const startTimeMs = Date.now();
  const gatewayReceivedAt = new Date().toISOString();
  const proposalId = generateProposalId();

  await ensureInitialBudgetLoaded();
  const validatedPayload = validateScenarioRequest(request.body);

  const currentBaselineOverride =
    validatedPayload.build_on_active_diagram &&
    activeBudgetState.active_summary
      ? activeBudgetState.active_summary
      : undefined;

  const controller = new AbortController();
  const timeoutHandle = setTimeout(
    () => controller.abort(),
    UPSTREAM_TIMEOUT_MS
  );

  let upstreamResponse: globalThis.Response;
  try {
    upstreamResponse = await fetch(`${FISCAL_ENGINE_URL}/api/v1/simulate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Proposal-ID": proposalId,
      },
      body: JSON.stringify({
        proposal_id: proposalId,
        raw_prompt: validatedPayload.raw_prompt,
        country_code: validatedPayload.country_code,
        fiscal_year: validatedPayload.fiscal_year,
        current_baseline_override: currentBaselineOverride,
        override_parameters: validatedPayload.override_parameters,
        enable_search_grounding: validatedPayload.enable_search_grounding,
      }),
      signal: controller.signal,
    });
  } catch (fetchErr) {
    clearTimeout(timeoutHandle);
    const isAbort =
      fetchErr instanceof Error && fetchErr.name === "AbortError";
    throw new GatewayHttpError(
      isAbort ? 504 : 502,
      isAbort ? "UPSTREAM_TIMEOUT" : "UPSTREAM_UNREACHABLE",
      isAbort
        ? `Simulation engine timed out after ${UPSTREAM_TIMEOUT_MS}ms.`
        : `Failed to reach Python simulation engine at ${FISCAL_ENGINE_URL}.`,
      fetchErr instanceof Error ? fetchErr.message : String(fetchErr)
    );
  } finally {
    clearTimeout(timeoutHandle);
  }

  if (!upstreamResponse.ok) {
    let errorBody: unknown;
    try {
      errorBody = await upstreamResponse.json();
    } catch {
      errorBody = await upstreamResponse.text();
    }
    throw new GatewayHttpError(
      upstreamResponse.status,
      "SIMULATION_ENGINE_ERROR",
      `Python simulation engine returned HTTP ${upstreamResponse.status}.`,
      errorBody
    );
  }

  const simulationResult =
    (await upstreamResponse.json()) as SimulationResult;
  const executionTimeMs = Date.now() - startTimeMs;

  const calculationCard: ChatCalculationCard = {
    proposal_id: proposalId,
    title: simulationResult.extracted_policy.policy_name,
    policy_domain: simulationResult.extracted_policy.policy_domain,
    chat_feedback: simulationResult.chat_feedback,
    total_revenue_delta_bn:
      simulationResult.fiscal_delta.total_revenue_delta_bn,
    total_expenditure_delta_bn:
      simulationResult.fiscal_delta.total_expenditure_delta_bn,
    net_budget_impact_bn: simulationResult.fiscal_delta.net_budget_impact_bn,
    line_item_deltas: simulationResult.fiscal_delta.line_item_deltas,
    apply_button: simulationResult.apply_action,
  };

  const envelope: GatewayProposalEnvelope = {
    proposal_id: proposalId,
    status: "PROPOSED",
    gateway_received_at: gatewayReceivedAt,
    execution_time_ms: executionTimeMs,
    chat_reply: {
      feedback_text: simulationResult.chat_feedback,
      calculation_card: calculationCard,
    },
    simulation: simulationResult,
  };

  proposalRegistry.set(proposalId, envelope);

  return reply.status(200).send(envelope);
}

server.post("/api/scenarios/run", handleProposeScenario);
server.post("/api/chat/propose", handleProposeScenario);

/**
 * POST /api/scenarios/:proposalId/apply
 * Step 2: User clicks "Apply to Budget" next to the calculation in Chat.
 * Transfers the proposed 2027 budget bars and Before/After table to the active state.
 */
server.post<{ Params: { proposalId: string } }>(
  "/api/scenarios/:proposalId/apply",
  async (request, reply) => {
    const { proposalId } = request.params;
    const proposal = proposalRegistry.get(proposalId);

    if (!proposal) {
      throw new GatewayHttpError(
        404,
        "PROPOSAL_NOT_FOUND",
        `Calculation proposal '${proposalId}' was not found.`
      );
    }

    const appliedAt = new Date().toISOString();
    proposal.status = "APPLIED";
    proposal.applied_at = appliedAt;
    proposal.simulation.apply_action.status = "APPLIED";
    proposal.simulation.apply_action.button_label = "Applied to Budget";
    proposal.chat_reply.calculation_card.apply_button =
      proposal.simulation.apply_action;

    // Commit calculation to active budget state
    activeBudgetState.active_summary = proposal.simulation.reformed_summary;
    activeBudgetState.active_bar_chart = proposal.simulation.bar_chart;
    activeBudgetState.active_sankey_graph = proposal.simulation.sankey_graph;
    activeBudgetState.applied_proposals.push({
      proposal_id: proposal.proposal_id,
      policy_name: proposal.simulation.extracted_policy.policy_name,
      policy_domain: proposal.simulation.extracted_policy.policy_domain,
      raw_prompt: proposal.simulation.raw_prompt,
      short_description:
        proposal.chat_reply.feedback_text ||
        proposal.simulation.chat_feedback ||
        "",
      revenue_delta_bn:
        proposal.simulation.fiscal_delta.total_revenue_delta_bn,
      expenditure_delta_bn:
        proposal.simulation.fiscal_delta.total_expenditure_delta_bn,
      net_budget_impact_bn:
        proposal.simulation.fiscal_delta.net_budget_impact_bn,
      line_item_deltas: proposal.simulation.fiscal_delta.line_item_deltas,
      applied_at: appliedAt,
      calculation_card: proposal.chat_reply.calculation_card,
    });

    return reply.status(200).send({
      status: "APPLIED",
      proposal_id: proposal.proposal_id,
      applied_at: appliedAt,
      message: `Applied '${proposal.simulation.extracted_policy.policy_name}' to the 2027 Federal Budget.`,
      active_diagram: {
        initial_2027_summary: activeBudgetState.initial_2027_summary,
        summary: activeBudgetState.active_summary,
        bar_chart: activeBudgetState.active_bar_chart,
        sankey_graph: activeBudgetState.active_sankey_graph,
        applied_proposals: activeBudgetState.applied_proposals,
      },
    });
  }
);

/**
 * GET /api/diagram/state
 * Returns the active 2027 budget summary, dual vertical bar chart, and comparison table.
 */
server.get("/api/diagram/state", async (_request: FastifyRequest, reply: FastifyReply) => {
  await ensureInitialBudgetLoaded();
  return reply.status(200).send(activeBudgetState);
});

/**
 * POST /api/diagram/reset
 * Resets the active budget back to the un-modified 2027 Government Draft baseline.
 */
server.post("/api/diagram/reset", async (_request: FastifyRequest, reply: FastifyReply) => {
  activeBudgetState.active_summary = null;
  activeBudgetState.active_bar_chart = null;
  activeBudgetState.active_sankey_graph = null;
  activeBudgetState.applied_proposals = [];
  await ensureInitialBudgetLoaded();
  return reply.status(200).send({
    status: "RESET",
    message:
      "Active budget reset to the 2027 Government Draft baseline (Regierungsentwurf 2027).",
    active_diagram: activeBudgetState,
  });
});

/**
 * GET /api/packages/export
 * Exports the active reform package with all applied reform widgets, short descriptions, and state.
 */
server.get("/api/packages/export", async (_request: FastifyRequest, reply: FastifyReply) => {
  await ensureInitialBudgetLoaded();
  const exportData = {
    package_title: "GovSim 2027 Fiscal Reform Package",
    version: "1.0.0",
    exported_at: new Date().toISOString(),
    baseline_reference: activeBudgetState.baseline_label,
    fiscal_year: activeBudgetState.fiscal_year,
    country_code: activeBudgetState.country_code,
    applied_reforms_count: activeBudgetState.applied_proposals.length,
    active_summary: activeBudgetState.active_summary,
    active_bar_chart: activeBudgetState.active_bar_chart,
    applied_proposals: activeBudgetState.applied_proposals,
  };
  return reply
    .header("Content-Disposition", 'attachment; filename="govsim_reform_package_2027.json"')
    .header("Content-Type", "application/json")
    .status(200)
    .send(exportData);
});

/**
 * POST /api/packages/import
 * Imports a reform package JSON file, restoring all reform widgets and recalculating/updating active state.
 */
server.post("/api/packages/import", async (request: FastifyRequest, reply: FastifyReply) => {
  await ensureInitialBudgetLoaded();
  const pkg = request.body as Record<string, unknown>;
  if (!pkg || typeof pkg !== "object" || !Array.isArray(pkg.applied_proposals)) {
    throw new GatewayHttpError(
      400,
      "INVALID_REFORM_PACKAGE",
      "The uploaded JSON file is not a valid GovSim Reform Package. Expected 'applied_proposals' array."
    );
  }

  activeBudgetState.applied_proposals = pkg.applied_proposals as AppliedProposalRecord[];
  if (pkg.active_summary) {
    activeBudgetState.active_summary = pkg.active_summary as FiscalSummary;
  }
  if (pkg.active_bar_chart) {
    activeBudgetState.active_bar_chart = pkg.active_bar_chart as DualBarChartPayload;
  }

  // Register any imported proposals into proposalRegistry so client lookups continue working
  (pkg.applied_proposals as AppliedProposalRecord[]).forEach((p: AppliedProposalRecord) => {
    if (p.proposal_id && !proposalRegistry.has(p.proposal_id)) {
      const dummyEnvelope: GatewayProposalEnvelope = {
        proposal_id: p.proposal_id,
        status: "APPLIED",
        gateway_received_at: p.applied_at || new Date().toISOString(),
        applied_at: p.applied_at,
        execution_time_ms: 0,
        chat_reply: {
          feedback_text: p.short_description || "",
          calculation_card: p.calculation_card || {
            proposal_id: p.proposal_id,
            title: p.policy_name,
            policy_domain: p.policy_domain,
            chat_feedback: p.short_description,
            total_revenue_delta_bn: p.revenue_delta_bn ?? 0,
            total_expenditure_delta_bn: p.expenditure_delta_bn ?? 0,
            net_budget_impact_bn: p.net_budget_impact_bn ?? 0,
            line_item_deltas: p.line_item_deltas ?? [],
            apply_button: {
              can_apply: false,
              status: "APPLIED",
              button_label: "Applied to Budget",
              apply_endpoint: `/api/scenarios/${p.proposal_id}/apply`,
            },
          },
        },
        simulation: {} as SimulationResult,
      };
      proposalRegistry.set(p.proposal_id, dummyEnvelope);
    }
  });

  return reply.status(200).send({
    status: "IMPORTED",
    message: `Successfully imported reform package with ${(pkg.applied_proposals as AppliedProposalRecord[]).length} policy reform(s).`,
    active_diagram: activeBudgetState,
  });
});

/**
 * POST /api/packages/assess
 * Critically assesses the active reform package via the Python engine, factoring in
 * dynamic macroeconomic feedback loops, growth multipliers, and debt yield risks.
 */
server.post("/api/packages/assess", async (request: FastifyRequest, reply: FastifyReply) => {
  await ensureInitialBudgetLoaded();
  const body = (request.body as Record<string, unknown>) || {};
  const proposals =
    Array.isArray(body.applied_proposals) && body.applied_proposals.length > 0
      ? body.applied_proposals
      : activeBudgetState.applied_proposals;

  const summary = (body.active_summary as FiscalSummary) || activeBudgetState.active_summary;

  const payload = {
    fiscal_year: activeBudgetState.fiscal_year,
    applied_proposals: proposals,
    active_summary: summary,
  };

  const controller = new AbortController();
  const timeoutHandle = setTimeout(
    () => controller.abort(),
    UPSTREAM_TIMEOUT_MS
  );

  let upstreamRes: globalThis.Response;
  try {
    upstreamRes = await fetch(`${FISCAL_ENGINE_URL}/api/v1/assess-package`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timeoutHandle);
  }

  if (!upstreamRes.ok) {
    let errBody: unknown;
    try {
      errBody = await upstreamRes.json();
    } catch {
      errBody = await upstreamRes.text();
    }
    throw new GatewayHttpError(
      upstreamRes.status,
      "ASSESSMENT_ERROR",
      "Python fiscal engine failed to critically assess reform package.",
      errBody
    );
  }

  const assessmentData = await upstreamRes.json();
  return reply.status(200).send(assessmentData);
});

/**
 * POST /api/packages/apply-corrections
 * Commits the dynamically corrected summary & dual-bar chart into active budget state.
 */
server.post("/api/packages/apply-corrections", async (request: FastifyRequest, reply: FastifyReply) => {
  await ensureInitialBudgetLoaded();
  const body = (request.body as Record<string, unknown>) || {};
  const assessment = body.assessment as Record<string, unknown>;
  if (!assessment || !assessment.corrected_summary || !assessment.corrected_bar_chart) {
    throw new GatewayHttpError(
      400,
      "INVALID_CORRECTION_PAYLOAD",
      "Expected 'assessment' object with corrected_summary and corrected_bar_chart."
    );
  }

  const appliedAt = new Date().toISOString();
  const correctionId = `macro_corr_${Date.now().toString(36)}`;

  // Line item deltas for the correction card
  const lineDeltas: LineItemDelta[] = ((assessment.line_corrections as Record<string, unknown>[]) || []).map(
    (c: Record<string, unknown>) => ({
      side: c.side as "revenue" | "expenditure",
      category_key: c.category_key as string,
      display_name: c.display_name as string,
      baseline_bn: Number(c.original_amount_bn),
      proposed_bn: Number(c.corrected_amount_bn),
      delta_bn: Number(c.adjustment_delta_bn),
      delta_pct:
        Number(c.original_amount_bn) > 0
          ? (Number(c.adjustment_delta_bn) / Number(c.original_amount_bn)) * 100
          : 0,
    })
  );

  // Commit to active state
  activeBudgetState.active_summary = assessment.corrected_summary as FiscalSummary;
  activeBudgetState.active_bar_chart = assessment.corrected_bar_chart as DualBarChartPayload;

  // Register as an applied reform record
  activeBudgetState.applied_proposals.push({
    proposal_id: correctionId,
    policy_name: (assessment.assessment_title as string) || "Dynamic Macroeconomic Feedback Calibration",
    policy_domain: "STRUCTURAL_CONSOLIDATION",
    raw_prompt: "Macroeconomic Reality Check & Dynamic Feedback Calibration",
    short_description: (assessment.executive_critique as string) || "Calibrated numbers with growth multipliers, tax elasticity, and debt interest risk.",
    revenue_delta_bn: (assessment.corrected_summary as FiscalSummary).total_revenue_bn - (activeBudgetState.initial_2027_summary?.total_revenue_bn ?? 436.7),
    expenditure_delta_bn: (assessment.corrected_summary as FiscalSummary).total_expenditure_bn - (activeBudgetState.initial_2027_summary?.total_expenditure_bn ?? 555.4),
    net_budget_impact_bn: Number(assessment.net_correction_impact_bn ?? 0),
    line_item_deltas: lineDeltas,
    applied_at: appliedAt,
  });

  return reply.status(200).send({
    status: "CORRECTIONS_APPLIED",
    message: "Successfully applied dynamic macroeconomic feedback corrections to the 2027 budget.",
    active_diagram: activeBudgetState,
  });
});

/**
 * GET /api/scenarios/:proposalId
 */
server.get<{ Params: { proposalId: string } }>(
  "/api/scenarios/:proposalId",
  async (request, reply) => {
    const { proposalId } = request.params;
    const stored = proposalRegistry.get(proposalId);
    if (!stored) {
      throw new GatewayHttpError(
        404,
        "PROPOSAL_NOT_FOUND",
        `Proposal '${proposalId}' was not found in the gateway registry.`
      );
    }
    return reply.status(200).send(stored);
  }
);

/**
 * GET /api/scenarios
 */
server.get("/api/scenarios", async (_request: FastifyRequest, reply: FastifyReply) => {
  const summaries = Array.from(proposalRegistry.values()).map((item) => ({
    proposal_id: item.proposal_id,
    status: item.status,
    gateway_received_at: item.gateway_received_at,
    applied_at: item.applied_at ?? null,
    execution_time_ms: item.execution_time_ms,
    raw_prompt: item.simulation.raw_prompt,
    chat_feedback: item.simulation.chat_feedback,
    policy_name: item.simulation.extracted_policy.policy_name,
    net_budget_impact_bn: item.simulation.fiscal_delta.net_budget_impact_bn,
    apply_button: item.chat_reply.calculation_card.apply_button,
  }));
  return reply.status(200).send({ count: summaries.length, proposals: summaries });
});

// ============================================================================
// Standard Error Handling in Fastify
// ============================================================================

server.setErrorHandler((error: any, _request, reply) => {
  if (error instanceof GatewayHttpError) {
    return reply.status(error.statusCode).send({
      error: {
        code: error.errorCode,
        message: error.message,
        details: error.details,
      },
    });
  }

  const statusCode =
    typeof error?.statusCode === "number" &&
    error.statusCode >= 400 &&
    error.statusCode < 600
      ? error.statusCode
      : 500;

  return reply.status(statusCode).send({
    error: {
      code: error?.code || "INTERNAL_GATEWAY_ERROR",
      message: error?.message || "Unexpected gateway error.",
    },
  });
});

// ============================================================================
// Server Bootstrap
// ============================================================================

export async function startServer(): Promise<FastifyInstance> {
  await server.register(cors, { origin: true });
  await server.register(fastifyStatic, {
    root: publicDir,
    prefix: "/",
  });

  await server.listen({ port: PORT, host: "0.0.0.0" });
  console.log(
    `[gateway_service] Listening on http://0.0.0.0:${PORT} -> forwarding to ${FISCAL_ENGINE_URL}`
  );
  return server;
}

if (require.main === module) {
  startServer().catch((err) => {
    console.error("Failed to start gateway server:", err);
    process.exit(1);
  });
}
