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
import jwt from "jsonwebtoken";
import {
  db,
  findUserByEmail,
  findUserById,
  createUser,
  updateUserTier,
  checkAndIncrementUsage,
  getTodayUsage,
  transferGuestUsageToUser,
  cleanClientIp,
  listUserScenarios,
  getScenarioById,
  saveNewScenario,
  updateScenario,
  deleteScenario,
  isAdminEmail,
  getSetting,
  setSetting,
  UserRecord,
  UserTier,
  normalizeTier,
  TIER_LIMITS,
} from "./db";
import {
  optionalAuthHook,
  requireAuthHook,
  requireAdminHook,
  signUserToken,
  verifyPassword,
  hashPassword,
  syncExternalUser,
} from "./auth";

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
  reasoning_effort?: "low" | "medium" | "high";
  model_name?: string;
  is_package_upload?: boolean;
  query_cost?: number;
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
    deconstruction_ledger?: {
      raw_measures_extracted?: Array<{
        clause_id: number;
        raw_text_excerpt: string;
        policy_action: string;
        target_budget_category: string;
        estimated_annual_bn: number;
        reasoning: string;
      }>;
      zero_double_counting_audit?: {
        checked_for_duplicate_subsidies: boolean;
        checked_for_phantom_revenues: boolean;
        all_measures_distinct: boolean;
        audit_notes: string;
      };
    };
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
  deconstruction_ledger?: {
    raw_measures_extracted?: Array<{
      clause_id: number;
      raw_text_excerpt: string;
      policy_action: string;
      target_budget_category: string;
      estimated_annual_bn: number;
      reasoning: string;
    }>;
    zero_double_counting_audit?: {
      checked_for_duplicate_subsidies: boolean;
      checked_for_phantom_revenues: boolean;
      all_measures_distinct: boolean;
      audit_notes: string;
    };
  };
}

export interface GatewayProposalEnvelope {
  proposal_id: string;
  status: "PROPOSED" | "APPLIED";
  model_used?: string;
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
  bodyLimit: 25 * 1024 * 1024, // 25MB for document packages (PDF, DOCX, TXT)
});

// Safely handle empty JSON bodies on DELETE / POST requests
server.addContentTypeParser(
  "application/json",
  { parseAs: "string" },
  (_req, body: string, done) => {
    if (!body || body.trim() === "") {
      done(null, {});
      return;
    }
    try {
      done(null, JSON.parse(body));
    } catch (err: any) {
      err.statusCode = 400;
      done(err, undefined);
    }
  }
);

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

  if (rawPrompt.length > 150000) {
    throw new GatewayHttpError(
      400,
      "VALIDATION_ERROR",
      "Field 'raw_prompt' must not exceed 150,000 characters."
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

  const reasoningEffort =
    candidate.reasoning_effort === "medium" || candidate.reasoning_effort === "high"
      ? (candidate.reasoning_effort as "medium" | "high")
      : "low";

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
    reasoning_effort: reasoningEffort,
    model_name: typeof candidate.model_name === "string" ? candidate.model_name : undefined,
    is_package_upload: Boolean(candidate.is_package_upload),
    query_cost: typeof candidate.query_cost === "number" ? candidate.query_cost : undefined,
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

  // Tier limit check & rate limiting
  await optionalAuthHook(request, reply);
  const user = request.user;
  const tier: UserTier = user ? user.tier : "guest";
  const identifier = request.clientIdentifier || "guest_ip:127.0.0.1";

  await ensureInitialBudgetLoaded();
  const validatedPayload = validateScenarioRequest(request.body);

  const isPackageUpload = Boolean(validatedPayload.is_package_upload);
  const isProQuery =
    isPackageUpload ||
    validatedPayload.reasoning_effort === "medium" ||
    validatedPayload.reasoning_effort === "high" ||
    Boolean(validatedPayload.enable_search_grounding);

  if (isProQuery && (tier === "guest" || tier === "free")) {
    return reply.status(403).send({
      error: {
        code: "PRO_QUERY_RESTRICTED",
        message:
          "Pro queries (Medium/High reasoning, Web Search, or Document Upload) require a Pro or Expert account. Please switch to Low reasoning without Web Search in Settings or upgrade your account.",
        tier,
      },
    });
  }

  const queryCost = isPackageUpload ? 3 : (validatedPayload.query_cost || 1);
  const usageCheck = checkAndIncrementUsage(identifier, tier, isProQuery, queryCost);
  if (!usageCheck.allowed) {
    return reply.status(429).send({
      error: {
        code: "TIER_LIMIT_REACHED",
        message: usageCheck.message,
        tier: usageCheck.tier,
        is_pro_query: usageCheck.is_pro_query,
        current_count: usageCheck.current_count,
        max_limit: usageCheck.max_limit,
        pro_queries_used: usageCheck.pro_queries_used,
        pro_queries_max: usageCheck.pro_queries_max,
        pro_queries_remaining: usageCheck.pro_queries_remaining,
        normal_queries_used: usageCheck.normal_queries_used,
        normal_queries_max: usageCheck.normal_queries_max,
        normal_queries_remaining: usageCheck.normal_queries_remaining,
        remaining: usageCheck.remaining,
        reset_in_hours: usageCheck.reset_in_hours,
      },
    });
  }

  // Set rate-limiting & tier headers
  reply.header("X-GovSim-Tier", usageCheck.tier);
  reply.header("X-GovSim-Queries-Remaining", usageCheck.remaining.toString());
  reply.header("X-GovSim-Queries-Limit", usageCheck.max_limit.toString());

  const currentBaselineOverride =
    validatedPayload.build_on_active_diagram &&
    activeBudgetState.active_summary
      ? activeBudgetState.active_summary
      : undefined;

  const modelToUse =
    validatedPayload.model_name ||
    (isPackageUpload
      ? "gpt-6.1-terra"
      : getSetting("active_model", "gpt-6-luna"));

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
        reasoning_effort: validatedPayload.reasoning_effort || "low",
        model_override: modelToUse,
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
    deconstruction_ledger: simulationResult.extracted_policy?.deconstruction_ledger,
  };

  const envelope: GatewayProposalEnvelope & { usage?: any } = {
    proposal_id: proposalId,
    status: "PROPOSED",
    model_used: simulationResult.metadata?.model_used || modelToUse,
    gateway_received_at: gatewayReceivedAt,
    execution_time_ms: executionTimeMs,
    chat_reply: {
      feedback_text: simulationResult.chat_feedback,
      calculation_card: calculationCard,
    },
    simulation: simulationResult,
    usage: {
      tier: usageCheck.tier,
      is_pro_query: usageCheck.is_pro_query,
      current_count: usageCheck.current_count,
      max_limit: usageCheck.max_limit,
      pro_queries_used: usageCheck.pro_queries_used,
      pro_queries_max: usageCheck.pro_queries_max,
      pro_queries_remaining: usageCheck.pro_queries_remaining,
      normal_queries_used: usageCheck.normal_queries_used,
      normal_queries_max: usageCheck.normal_queries_max,
      normal_queries_remaining: usageCheck.normal_queries_remaining,
      remaining: usageCheck.remaining,
      reset_in_hours: usageCheck.reset_in_hours,
    },
  };

  proposalRegistry.set(proposalId, envelope);

  return reply.status(200).send(envelope);
}

server.post("/api/scenarios/run", handleProposeScenario);
server.post("/api/chat/propose", handleProposeScenario);

/**
 * POST /api/chat/upload-package
 * Business & Admin feature: Upload a PDF, DOCX, or TXT reform package document.
 * Extracts the document text and runs a comprehensive budget simulation.
 */
server.post("/api/chat/upload-package", async (request: FastifyRequest, reply: FastifyReply) => {
  await requireAuthHook(request, reply);
  const user = request.user;
  if (!user || (user.tier !== "expert" && user.role !== "admin")) {
    throw new GatewayHttpError(
      403,
      "TIER_RESTRICTED",
      "Document package upload (PDF, TXT, DOCX) is exclusively available for Expert tier and Administrators."
    );
  }

  const body = (request.body as Record<string, unknown>) || {};
  const filename = typeof body.filename === "string" ? body.filename.trim() : "";
  const contentBase64 = typeof body.content_base64 === "string" ? body.content_base64.trim() : "";

  if (!filename || !contentBase64) {
    throw new GatewayHttpError(
      400,
      "MISSING_DOCUMENT",
      "Both 'filename' and 'content_base64' are required."
    );
  }

  // Forward document to Fiscal AI Engine for extraction
  let extractedText: string;
  try {
    const upstream = await fetch(`${FISCAL_ENGINE_URL}/api/v1/extract-document`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename, content_base64: contentBase64 }),
      signal: AbortSignal.timeout(30000),
    });
    if (!upstream.ok) {
      const errData = (await upstream.json().catch(() => ({}))) as any;
      throw new GatewayHttpError(
        upstream.status,
        "DOCUMENT_EXTRACTION_FAILED",
        errData.detail || "Failed to parse document text."
      );
    }
    const data = (await upstream.json()) as any;
    extractedText = data.text;
  } catch (err: any) {
    if (err instanceof GatewayHttpError) throw err;
    throw new GatewayHttpError(
      502,
      "EXTRACTION_SERVICE_ERROR",
      `Document parsing engine unreachable: ${err.message}`
    );
  }

  // Prepend package source context for clear economist attribution, multi-level/social-security consolidation, and anti-bias audit
  const policyPrompt = `[Uploaded Legislative & Reform Document: ${filename}]\n\nMandate: Analyze this comprehensive reform package and model its full structural impact across the German public finance architecture (consolidating effects across Bund, Länder/Kommunen via Finanzausgleich/Konnexität, and Social Security RV/ALV/GKV into the canonical budget categories — never set sub-federal or social security fiscal reforms to 0.0). Always think independently: critically audit and recalculate every claimed revenue, transfer cost (especially Negative Income Tax / Sockel break-even phase-out credits), deferral (e.g. 20-year corporate inheritance tax installments), and subsidy cut from first principles rather than blindly accepting the document's self-reported outcomes:\n\n${extractedText}`;
  request.body = {
    raw_prompt: policyPrompt,
    build_on_active_diagram: body.build_on_active_diagram !== false,
    enable_search_grounding: Boolean(body.enable_search_grounding),
    reasoning_effort: body.reasoning_effort || "low",
    model_name: "gpt-6.1-terra",
    is_package_upload: true,
    query_cost: 3,
  };

  return handleProposeScenario(request, reply);
});

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
  await optionalAuthHook(request, reply);
  const user = request.user;
  const tier: UserTier = user ? user.tier : "guest";
  const identifier = request.clientIdentifier || "guest_ip:127.0.0.1";

  const body = (request.body as Record<string, unknown>) || {};
  const isProQuery =
    body.reasoning_effort === "medium" ||
    body.reasoning_effort === "high" ||
    Boolean(body.enable_search_grounding);

  // Assess Queries count as 2 queries
  const usageCheck = checkAndIncrementUsage(identifier, tier, isProQuery, 2);
  if (!usageCheck.allowed) {
    return reply.status(429).send({
      error: {
        code: "TIER_LIMIT_REACHED",
        message: usageCheck.message,
        tier: usageCheck.tier,
        is_pro_query: usageCheck.is_pro_query,
        current_count: usageCheck.current_count,
        max_limit: usageCheck.max_limit,
        pro_queries_used: usageCheck.pro_queries_used,
        pro_queries_max: usageCheck.pro_queries_max,
        pro_queries_remaining: usageCheck.pro_queries_remaining,
        normal_queries_used: usageCheck.normal_queries_used,
        normal_queries_max: usageCheck.normal_queries_max,
        normal_queries_remaining: usageCheck.normal_queries_remaining,
        remaining: usageCheck.remaining,
        reset_in_hours: usageCheck.reset_in_hours,
      },
    });
  }

  reply.header("X-GovSim-Tier", usageCheck.tier);
  reply.header("X-GovSim-Queries-Remaining", usageCheck.remaining.toString());
  reply.header("X-GovSim-Queries-Limit", usageCheck.max_limit.toString());

  await ensureInitialBudgetLoaded();
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

  const assessmentData = (await upstreamRes.json()) as any;
  assessmentData.usage = usageCheck;
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

const KEYCLOAK_URL = process.env.KEYCLOAK_URL || "http://localhost:8080";
const KEYCLOAK_REALM = process.env.KEYCLOAK_REALM || "govsim";
const KEYCLOAK_CLIENT_ID = process.env.KEYCLOAK_CLIENT_ID || "govsim-web";

/**
 * Authenticates user credentials directly against Keycloak via Direct Access Grants (password flow).
 * Returns the synced UserRecord if valid, or null if Keycloak rejected or is unreachable.
 */
async function authenticateWithKeycloak(email: string, password: string): Promise<UserRecord | null> {
  try {
    const res = await fetch(`${KEYCLOAK_URL}/realms/${KEYCLOAK_REALM}/protocol/openid-connect/token`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        grant_type: "password",
        client_id: KEYCLOAK_CLIENT_ID,
        username: email,
        password,
        scope: "openid profile email",
      }),
      signal: AbortSignal.timeout(3000),
    });

    if (!res.ok) {
      return null;
    }

    const data = (await res.json()) as any;
    const unverified = jwt.decode(data.id_token || data.access_token) as any;
    if (!unverified) return null;

    const roles: string[] = unverified?.realm_access?.roles || [];
    let tier: UserTier = "free";
    if (roles.includes("pro") || roles.includes("tier_pro") || roles.includes("journalist")) {
      tier = "pro";
    } else if (roles.includes("business") || roles.includes("enterprise") || roles.includes("expert")) {
      tier = "expert";
    }

    return syncExternalUser({
      email: unverified.email || email,
      name: unverified.name || unverified.preferred_username || email.split("@")[0],
      provider: "keycloak",
      providerId: unverified.sub || email,
      tier,
    });
  } catch {
    return null;
  }
}

/**
 * Creates a new user in Keycloak via Keycloak's Admin REST API.
 */
async function registerUserInKeycloak(email: string, name: string, password: string): Promise<boolean> {
  try {
    const adminTokenRes = await fetch(`${KEYCLOAK_URL}/realms/master/protocol/openid-connect/token`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        grant_type: "password",
        client_id: "admin-cli",
        username: process.env.KEYCLOAK_ADMIN || "admin",
        password: process.env.KEYCLOAK_ADMIN_PASSWORD || "admin",
      }),
      signal: AbortSignal.timeout(3000),
    });

    if (!adminTokenRes.ok) return false;
    const adminData = (await adminTokenRes.json()) as any;
    const adminToken = adminData.access_token;

    const createUserRes = await fetch(`${KEYCLOAK_URL}/admin/realms/${KEYCLOAK_REALM}/users`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${adminToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        username: email,
        email,
        firstName: name,
        enabled: true,
        emailVerified: true,
        credentials: [
          {
            type: "password",
            value: password,
            temporary: false,
          },
        ],
      }),
      signal: AbortSignal.timeout(3000),
    });

    return createUserRes.status === 201 || createUserRes.status === 409;
  } catch {
    return false;
  }
}

/**
 * GET /api/auth/keycloak/login
 * Initiates the standard OpenID Connect authorization code redirect to Keycloak.
 * Supports ?idp=google and ?idp=apple for direct social login forwarding.
 */
server.get<{ Querystring: { idp?: string } }>("/api/auth/keycloak/login", async (request, reply) => {
  const host = request.headers.host || "localhost:3000";
  const protocol = request.headers["x-forwarded-proto"] || "http";
  const redirectUri = `${protocol}://${host}/api/auth/keycloak/callback`;

  let authUrl = `${KEYCLOAK_URL}/realms/${KEYCLOAK_REALM}/protocol/openid-connect/auth?client_id=${KEYCLOAK_CLIENT_ID}&response_type=code&scope=openid%20profile%20email&redirect_uri=${encodeURIComponent(
    redirectUri
  )}`;

  if (request.query.idp) {
    authUrl += `&kc_idp_hint=${encodeURIComponent(request.query.idp)}`;
  }

  return reply.redirect(authUrl);
});

/**
 * Direct aliases for Google & Apple OIDC redirects via Keycloak
 */
server.get("/api/auth/login/google", async (_request, reply) => {
  return reply.redirect("/api/auth/keycloak/login?idp=google");
});

server.get("/api/auth/login/apple", async (_request, reply) => {
  return reply.redirect("/api/auth/keycloak/login?idp=apple");
});

/**
 * GET /api/auth/keycloak/callback
 * Exchanges Keycloak authorization code for JWT access token and logs user into GovSim.
 */
server.get<{ Querystring: { code?: string; error?: string } }>(
  "/api/auth/keycloak/callback",
  async (request, reply) => {
    const { code, error } = request.query;

    if (error || !code) {
      return reply.redirect(`/?auth_error=${encodeURIComponent(error || "Authorization cancelled")}`);
    }

    const host = request.headers.host || "localhost:3000";
    const protocol = request.headers["x-forwarded-proto"] || "http";
    const redirectUri = `${protocol}://${host}/api/auth/keycloak/callback`;

    try {
      const tokenRes = await fetch(
        `${KEYCLOAK_URL}/realms/${KEYCLOAK_REALM}/protocol/openid-connect/token`,
        {
          method: "POST",
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          body: new URLSearchParams({
            grant_type: "authorization_code",
            client_id: KEYCLOAK_CLIENT_ID,
            code,
            redirect_uri: redirectUri,
          }),
        }
      );

      if (!tokenRes.ok) {
        const errText = await tokenRes.text();
        console.error("[Keycloak] Token exchange failed:", errText);
        return reply.redirect(`/?auth_error=Token+exchange+failed`);
      }

      const tokenData = (await tokenRes.json()) as any;
      const idToken = tokenData.id_token || tokenData.access_token;
      const unverified = jwt.decode(idToken) as any;

      const email = unverified?.email || `${unverified?.preferred_username || unverified?.sub || "user"}@keycloak.local`;
      const name = unverified?.name || unverified?.preferred_username || "Keycloak User";
      const roles: string[] = unverified?.realm_access?.roles || [];

      let tier: UserTier = "free";
      if (roles.includes("pro") || roles.includes("tier_pro") || roles.includes("journalist")) {
        tier = "pro";
      } else if (roles.includes("business") || roles.includes("enterprise") || roles.includes("expert")) {
        tier = "expert";
      }

      const user = syncExternalUser({
        email,
        name,
        provider: "keycloak",
        providerId: unverified?.sub || `kc_${Date.now()}`,
        tier,
      });

      const sessionToken = signUserToken(user);
      const rawIp =
        (request.headers["x-forwarded-for"] as string)?.split(",")[0].trim() ||
        request.socket.remoteAddress ||
        "127.0.0.1";
      transferGuestUsageToUser(rawIp, user.id);

      return reply.redirect(`/?token=${sessionToken}`);
    } catch (err: any) {
      console.error("[Keycloak] Callback error:", err);
      return reply.redirect(`/?auth_error=${encodeURIComponent(err.message)}`);
    }
  }
);

/**
 * GET /api/auth/keycloak/status
 * Returns whether Keycloak is currently live and reachable.
 */
server.get("/api/auth/keycloak/status", async (_request: FastifyRequest, reply: FastifyReply) => {
  try {
    const res = await fetch(`${KEYCLOAK_URL}/realms/${KEYCLOAK_REALM}`, {
      method: "GET",
      signal: AbortSignal.timeout(2000),
    });
    return reply.status(200).send({
      available: res.ok,
      realm: KEYCLOAK_REALM,
      url: KEYCLOAK_URL,
    });
  } catch {
    return reply.status(200).send({
      available: false,
      realm: KEYCLOAK_REALM,
      url: KEYCLOAK_URL,
    });
  }
});

// ============================================================================
// User Authentication & Identity Endpoints (Keycloak / Google / Apple / Local)
// ============================================================================

/**
 * POST /api/auth/register
 * Creates a new user in SQLite (and Keycloak if active) and returns a JWT token.
 */
server.post("/api/auth/register", async (request: FastifyRequest, reply: FastifyReply) => {
  const body = (request.body as Record<string, unknown>) || {};
  const email = typeof body.email === "string" ? body.email.trim() : "";
  const password = typeof body.password === "string" ? body.password : "";
  const name = typeof body.name === "string" ? body.name.trim() : "";

  if (!email || !email.includes("@")) {
    throw new GatewayHttpError(400, "INVALID_EMAIL", "A valid email address is required.");
  }
  if (!password || password.length < 6) {
    throw new GatewayHttpError(400, "WEAK_PASSWORD", "Password must be at least 6 characters long.");
  }

  const existing = findUserByEmail(email);
  if (existing) {
    throw new GatewayHttpError(409, "USER_EXISTS", "An account with this email address already exists.");
  }

  // Register in Keycloak asynchronously in background if available
  registerUserInKeycloak(email, name || email.split("@")[0], password).catch((e) =>
    console.warn("[Keycloak] User registration sync skipped:", e.message)
  );

  const passwordHash = await hashPassword(password);
  const newUser = createUser({
    id: crypto.randomUUID(),
    email,
    password_hash: passwordHash,
    name: name || email.split("@")[0],
    tier: "free",
    role: "user",
    auth_provider: "local",
  });

  const token = signUserToken(newUser);
  const rawIp =
    (request.headers["x-forwarded-for"] as string)?.split(",")[0].trim() ||
    request.socket.remoteAddress ||
    "127.0.0.1";
  transferGuestUsageToUser(rawIp, newUser.id);

  return reply.status(201).send({
    status: "CREATED",
    token,
    user: {
      id: newUser.id,
      email: newUser.email,
      name: newUser.name,
      tier: newUser.tier,
      role: newUser.role,
      provider: newUser.auth_provider,
    },
    message: "Welcome to GovSim AI! You are on the Free Tier (10 simulations/day).",
  });
});

/**
 * POST /api/auth/login
 * Validates credentials via Keycloak Direct Access Grant or SQLite fallback, returning a JWT token.
 */
server.post("/api/auth/login", async (request: FastifyRequest, reply: FastifyReply) => {
  const body = (request.body as Record<string, unknown>) || {};
  const email = typeof body.email === "string" ? body.email.trim() : "";
  const password = typeof body.password === "string" ? body.password : "";

  if (!email || !password) {
    throw new GatewayHttpError(400, "MISSING_CREDENTIALS", "Email and password are required.");
  }

  // 1. Try Keycloak authentication first if Keycloak is reachable
  const kcUser = await authenticateWithKeycloak(email, password);
  if (kcUser) {
    const token = signUserToken(kcUser);
    const rawIp =
      (request.headers["x-forwarded-for"] as string)?.split(",")[0].trim() ||
      request.socket.remoteAddress ||
      "127.0.0.1";
    transferGuestUsageToUser(rawIp, kcUser.id);

    return reply.status(200).send({
      status: "AUTHENTICATED",
      token,
      user: {
        id: kcUser.id,
        email: kcUser.email,
        name: kcUser.name,
        tier: kcUser.tier,
        role: kcUser.role,
        provider: kcUser.auth_provider,
      },
    });
  }

  // 2. Fall back to local SQLite credentials
  const user = findUserByEmail(email);
  if (!user || !user.password_hash) {
    throw new GatewayHttpError(401, "INVALID_CREDENTIALS", "Invalid email or password.");
  }

  const isValid = await verifyPassword(password, user.password_hash);
  if (!isValid) {
    throw new GatewayHttpError(401, "INVALID_CREDENTIALS", "Invalid email or password.");
  }

  const token = signUserToken(user);
  const rawIp =
    (request.headers["x-forwarded-for"] as string)?.split(",")[0].trim() ||
    request.socket.remoteAddress ||
    "127.0.0.1";
  transferGuestUsageToUser(rawIp, user.id);

  return reply.status(200).send({
    status: "AUTHENTICATED",
    token,
    user: {
      id: user.id,
      email: user.email,
      name: user.name,
      tier: user.tier,
      role: user.role,
      provider: user.auth_provider,
    },
  });
});

/**
 * POST /api/auth/social
 * Handles federated authentication (Keycloak SSO, Google, Apple ID tokens).
 * Automatically provisions or links the user account in SQLite.
 */
server.post("/api/auth/social", async (request: FastifyRequest, reply: FastifyReply) => {
  const body = (request.body as Record<string, unknown>) || {};
  const provider = body.provider as "keycloak" | "google" | "apple";
  const email = typeof body.email === "string" ? body.email.trim() : "";
  const name = typeof body.name === "string" ? body.name.trim() : "";
  const providerId = (body.providerId as string) || (body.sub as string) || `prov_${Date.now()}`;
  const tier = (body.tier as UserTier) || "free";

  if (!provider || !["keycloak", "google", "apple"].includes(provider)) {
    throw new GatewayHttpError(400, "INVALID_PROVIDER", "Provider must be 'keycloak', 'google', or 'apple'.");
  }
  if (!email || !email.includes("@")) {
    throw new GatewayHttpError(400, "INVALID_EMAIL", "A valid email is required from the OAuth provider.");
  }

  const user = syncExternalUser({
    email,
    name: name || email.split("@")[0],
    provider,
    providerId,
    tier,
  });

  const token = signUserToken(user);
  const rawIp =
    (request.headers["x-forwarded-for"] as string)?.split(",")[0].trim() ||
    request.socket.remoteAddress ||
    "127.0.0.1";
  transferGuestUsageToUser(rawIp, user.id);

  return reply.status(200).send({
    status: "AUTHENTICATED",
    token,
    user: {
      id: user.id,
      email: user.email,
      name: user.name,
      tier: user.tier,
      role: user.role,
      provider: user.auth_provider,
    },
  });
});

/**
 * GET /api/auth/me
 * Returns current authenticated user and today's usage statistics / limits.
 */
server.get("/api/auth/me", async (request: FastifyRequest, reply: FastifyReply) => {
  await optionalAuthHook(request, reply);
  const user = request.user;
  const is_admin = user ? (user.role === "admin" || isAdminEmail(user.email)) : false;
  const tier: UserTier = user ? (is_admin ? "expert" : user.tier) : "guest";
  const identifier = request.clientIdentifier || "guest_ip:127.0.0.1";
  const activeModel = getSetting("active_model", "gpt-6-luna");

  if (user) {
    const rawIp =
      (request.headers["x-forwarded-for"] as string)?.split(",")[0].trim() ||
      request.socket.remoteAddress ||
      "127.0.0.1";
    transferGuestUsageToUser(rawIp, user.id);
  }

  const usage = getTodayUsage(identifier, tier);

  return reply.status(200).send({
    authenticated: !!user,
    user: user ? { ...user, role: is_admin ? "admin" : user.role, tier } : null,
    is_admin,
    tier,
    active_model: activeModel,
    usage: {
      current_count: usage.current_count,
      max_limit: usage.max_limit,
      pro_queries_used: usage.pro_queries_used,
      pro_queries_max: usage.pro_queries_max,
      pro_queries_remaining: usage.pro_queries_remaining,
      normal_queries_used: usage.normal_queries_used,
      normal_queries_max: usage.normal_queries_max,
      normal_queries_remaining: usage.normal_queries_remaining,
      remaining: usage.remaining,
      reset_in_hours: usage.reset_in_hours,
      allowed: usage.allowed,
    },
    limits: TIER_LIMITS[tier],
  });
});

/**
 * POST /api/auth/tier
 * Switch tier for testing / upgrading (Restricted to platform administrator).
 */
server.post("/api/auth/tier", async (request: FastifyRequest, reply: FastifyReply) => {
  await requireAdminHook(request, reply);
  const body = (request.body as Record<string, unknown>) || {};
  const requestedTier = body.tier as string;

  if (!requestedTier || !["free", "pro", "expert", "business"].includes(requestedTier)) {
    throw new GatewayHttpError(400, "INVALID_TIER", "Tier must be 'free', 'pro', or 'expert'.");
  }

  const normTier: UserTier = normalizeTier(requestedTier);
  updateUserTier(request.user!.id, normTier);
  const updatedUser = findUserById(request.user!.id)!;
  const newToken = signUserToken(updatedUser);

  return reply.status(200).send({
    status: "TIER_UPDATED",
    message: `Account upgraded to ${TIER_LIMITS[normTier].label}.`,
    user: {
      id: updatedUser.id,
      email: updatedUser.email,
      name: updatedUser.name,
      tier: updatedUser.tier,
      role: updatedUser.role,
      provider: updatedUser.auth_provider,
    },
    token: newToken,
    limits: TIER_LIMITS[normTier],
  });
});

/**
 * GET /api/admin/model
 * Returns the currently active LLM model and selectable options from Fiscal Engine.
 */
server.get("/api/admin/model", async (request: FastifyRequest, reply: FastifyReply) => {
  await requireAdminHook(request, reply);
  try {
    const upstream = await fetch(`${FISCAL_ENGINE_URL}/api/v1/model`, {
      signal: AbortSignal.timeout(3000),
    });
    if (upstream.ok) {
      const data = await upstream.json();
      return reply.status(200).send(data);
    }
  } catch {}

  const activeModel = getSetting("active_model", "gpt-6-luna");
  return reply.status(200).send({
    active_model: activeModel,
    provider: activeModel.startsWith("gemini-") ? "google" : "openai",
  });
});

/**
 * POST /api/admin/model
 * Dynamically switches the active AI model (e.g. GPT-6-Luna, Gemini 3.8 Flash, etc.)
 */
server.post("/api/admin/model", async (request: FastifyRequest, reply: FastifyReply) => {
  await requireAdminHook(request, reply);
  const body = (request.body as Record<string, unknown>) || {};
  const modelName = typeof body.model_name === "string" ? body.model_name.trim() : "";
  const apiKey = typeof body.api_key === "string" ? body.api_key.trim() : undefined;

  if (!modelName) {
    throw new GatewayHttpError(400, "MISSING_MODEL", "model_name is required.");
  }

  setSetting("active_model", modelName);
  process.env.ACTIVE_MODEL = modelName;
  if (apiKey) {
    setSetting("openai_api_key", apiKey);
    process.env.OPENAI_API_KEY = apiKey;
  }

  // Forward update to Fiscal AI Engine
  try {
    const upstream = await fetch(`${FISCAL_ENGINE_URL}/api/v1/model`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_name: modelName, api_key: apiKey }),
      signal: AbortSignal.timeout(4000),
    });
    if (upstream.ok) {
      const data = (await upstream.json()) as any;
      return reply.status(200).send({
        status: "UPDATED",
        active_model: modelName,
        provider: data.provider,
        message: `Active model successfully changed to ${modelName}.`,
      });
    }
  } catch (err: any) {
    console.warn("[Gateway] Failed to synchronize model with Fiscal Engine:", err.message);
  }

  return reply.status(200).send({
    status: "UPDATED",
    active_model: modelName,
    message: `Active model updated in database to ${modelName}.`,
  });
});

// ============================================================================
// Multi-Scenario Database Management (Replaces JSON File Downloads)
// ============================================================================

/**
 * GET /api/scenarios/saves
 * Lists all scenarios saved in SQLite for the authenticated user.
 */
server.get("/api/scenarios/saves", async (request: FastifyRequest, reply: FastifyReply) => {
  await requireAuthHook(request, reply);
  const scenarios = listUserScenarios(request.user!.id, request.user!.tier);
  return reply.status(200).send({
    count: scenarios.length,
    scenarios,
  });
});

/**
 * POST /api/scenarios/saves
 * Saves the current active diagram state into the SQLite database under a user-defined title.
 */
server.post("/api/scenarios/saves", async (request: FastifyRequest, reply: FastifyReply) => {
  await requireAuthHook(request, reply);
  await ensureInitialBudgetLoaded();

  const body = (request.body as Record<string, unknown>) || {};
  const title = typeof body.title === "string" && body.title.trim() ? body.title.trim() : "Untitled Fiscal Scenario";
  const description = typeof body.description === "string" ? body.description.trim() : "";
  const tags = Array.isArray(body.tags) ? (body.tags as string[]) : [];

  const scenarioId = `scen_${crypto.randomUUID()}`;
  const saved = saveNewScenario({
    id: scenarioId,
    user_id: request.user!.id,
    title,
    description,
    tags,
    active_summary: activeBudgetState.active_summary || activeBudgetState.initial_2027_summary,
    active_bar_chart: activeBudgetState.active_bar_chart,
    active_sankey_graph: activeBudgetState.active_sankey_graph,
    applied_proposals: activeBudgetState.applied_proposals,
  });

  return reply.status(201).send({
    status: "SAVED",
    message: `Scenario '${saved.title}' saved successfully to your cloud database.`,
    scenario: saved,
  });
});

/**
 * GET /api/scenarios/saves/:id
 * Loads a saved scenario from SQLite directly into the active diagram workspace.
 */
server.get<{ Params: { id: string } }>(
  "/api/scenarios/saves/:id",
  async (request, reply) => {
    await requireAuthHook(request, reply);
    await ensureInitialBudgetLoaded();
    const user = request.user!;

    if (user.tier === "free") {
      const allScenarios = listUserScenarios(user.id, "free");
      const top5Ids = allScenarios.slice(0, 5).map((s) => s.id);
      if (!top5Ids.includes(request.params.id)) {
        throw new GatewayHttpError(
          403,
          "DRAFT_HISTORY_LIMIT",
          "Free tier accounts can revisit their last 5 drafts. Upgrade to Pro or Expert to access your complete draft history."
        );
      }
    }

    const scenario = getScenarioById(request.params.id, user.id);
    if (!scenario) {
      throw new GatewayHttpError(404, "SCENARIO_NOT_FOUND", "Saved scenario was not found in your account.");
    }

    // Restore into active diagram state
    activeBudgetState.active_summary = scenario.active_summary;
    activeBudgetState.active_bar_chart = scenario.active_bar_chart;
    activeBudgetState.active_sankey_graph = scenario.active_sankey_graph;
    activeBudgetState.applied_proposals = scenario.applied_proposals;

    // Register proposals in proposalRegistry for card lookups
    (scenario.applied_proposals as AppliedProposalRecord[]).forEach((p: AppliedProposalRecord) => {
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
      status: "LOADED",
      scenario_id: scenario.id,
      title: scenario.title,
      description: scenario.description,
      applied_reforms_count: scenario.applied_proposals.length,
      active_diagram: activeBudgetState,
    });
  }
);

/**
 * PUT /api/scenarios/saves/:id
 * Updates or overwrites an existing scenario with the current active budget state.
 */
server.put<{ Params: { id: string } }>(
  "/api/scenarios/saves/:id",
  async (request, reply) => {
    await requireAuthHook(request, reply);
    const body = (request.body as Record<string, unknown>) || {};
    const title = typeof body.title === "string" ? body.title : undefined;
    const description = typeof body.description === "string" ? body.description : undefined;
    const overwriteWithActive = body.overwrite_with_active === true;

    const updates: any = {};
    if (title !== undefined) updates.title = title;
    if (description !== undefined) updates.description = description;

    if (overwriteWithActive) {
      updates.active_summary = activeBudgetState.active_summary || activeBudgetState.initial_2027_summary;
      updates.active_bar_chart = activeBudgetState.active_bar_chart;
      updates.active_sankey_graph = activeBudgetState.active_sankey_graph;
      updates.applied_proposals = activeBudgetState.applied_proposals;
    }

    const success = updateScenario(request.params.id, request.user!.id, updates);
    if (!success) {
      throw new GatewayHttpError(404, "SCENARIO_NOT_FOUND", "Saved scenario was not found.");
    }

    return reply.status(200).send({
      status: "UPDATED",
      message: "Scenario updated successfully.",
    });
  }
);

/**
 * DELETE /api/scenarios/saves/:id
 * Deletes a scenario from the database.
 */
server.delete<{ Params: { id: string } }>(
  "/api/scenarios/saves/:id",
  async (request, reply) => {
    await requireAuthHook(request, reply);
    const success = deleteScenario(request.params.id, request.user!.id);
    if (!success) {
      throw new GatewayHttpError(404, "SCENARIO_NOT_FOUND", "Saved scenario was not found.");
    }
    return reply.status(200).send({
      status: "DELETED",
      id: request.params.id,
      message: "Scenario deleted from database.",
    });
  }
);

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

  // Synchronize stored model & API key from SQLite to Fiscal AI Engine
  setTimeout(async () => {
    try {
      const storedModel = getSetting("active_model", "gpt-6-luna");
      const storedKey = getSetting("openai_api_key", undefined);
      if (storedModel || storedKey) {
        await fetch(`${FISCAL_ENGINE_URL}/api/v1/model`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model_name: storedModel, api_key: storedKey }),
          signal: AbortSignal.timeout(5000),
        });
        console.log(`[gateway_service] Synchronized active model '${storedModel}' and API credentials to Fiscal Engine.`);
      }
    } catch (e: any) {
      console.warn("[gateway_service] Initial sync to Fiscal Engine deferred:", e.message);
    }
  }, 1000);

  return server;
}

if (require.main === module) {
  startServer().catch((err) => {
    console.error("Failed to start gateway server:", err);
    process.exit(1);
  });
}
