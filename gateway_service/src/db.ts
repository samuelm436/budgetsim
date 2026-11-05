/**
 * GovSim AI — SQLite Database Layer
 * Manages Users, Tiers & Rate Limits, Multi-Scenario Saves, and Persistent Sessions.
 */

import fs from "fs";
import path from "path";
import Database, { Database as SqliteDatabase } from "better-sqlite3";

export type UserTier = "guest" | "free" | "pro" | "business";

export interface UserRecord {
  id: string;
  email: string;
  password_hash: string | null;
  name: string;
  tier: UserTier;
  role: string;
  auth_provider: "local" | "keycloak" | "google" | "apple";
  auth_provider_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface SavedScenarioRecord {
  id: string;
  user_id: string;
  title: string;
  description: string;
  tags: string[];
  applied_reforms_count: number;
  active_summary: any;
  active_bar_chart: any;
  active_sankey_graph: any;
  applied_proposals: any[];
  created_at: string;
  updated_at: string;
}

export interface UsageCheckResult {
  allowed: boolean;
  tier: UserTier;
  current_count: number;
  max_limit: number;
  remaining: number;
  reset_in_hours: number;
  message?: string;
}

// Tier limits definition
export const TIER_LIMITS: Record<UserTier, { daily_queries: number; max_saves: number; label: string }> = {
  guest: { daily_queries: 1, max_saves: 0, label: "Guest (1 Test Query)" },
  free: { daily_queries: 10, max_saves: 10, label: "Free Tier (10 Queries/Day)" },
  pro: { daily_queries: 250, max_saves: 100, label: "Pro Tier (250 Queries/Day)" },
  business: { daily_queries: 1000, max_saves: 1000, label: "Business Tier (Unlimited)" },
};

function resolveDatabasePath(): string {
  if (process.env.DB_PATH) {
    const customDir = path.dirname(process.env.DB_PATH);
    if (!fs.existsSync(customDir)) {
      fs.mkdirSync(customDir, { recursive: true });
    }
    return process.env.DB_PATH;
  }

  // Look for ./data relative to current working directory or parent
  const primaryDir = path.resolve(process.cwd(), "data");
  const parentDir = path.resolve(process.cwd(), "../data");

  const targetDir = fs.existsSync(parentDir) ? parentDir : primaryDir;
  if (!fs.existsSync(targetDir)) {
    fs.mkdirSync(targetDir, { recursive: true });
  }

  return path.join(targetDir, "govsim.sqlite3");
}

const dbPath = resolveDatabasePath();
console.log(`[Database] Initializing GovSim SQLite database at: ${dbPath}`);

export const db: SqliteDatabase = new Database(dbPath);

export const ADMIN_EMAILS = ["samuelmartin.nh16@gmail.com"];

export function isAdminEmail(email: string): boolean {
  return ADMIN_EMAILS.includes(email.toLowerCase().trim());
}

export function getSetting(key: string, defaultValue: string = ""): string {
  try {
    const row = db.prepare("SELECT value FROM app_settings WHERE key = ?").get(key) as any;
    return row?.value || defaultValue;
  } catch {
    return defaultValue;
  }
}

export function setSetting(key: string, value: string): void {
  const now = new Date().toISOString();
  db.prepare(`
    INSERT INTO app_settings (key, value, updated_at)
    VALUES (?, ?, ?)
    ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
  `).run(key, value, now);
}

// Enable foreign keys and WAL mode for high concurrency
db.pragma("foreign_keys = ON");
db.pragma("journal_mode = WAL");

// Initialize Database Schemas
export function initDatabase(): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS users (
      id TEXT PRIMARY KEY,
      email TEXT UNIQUE NOT NULL,
      password_hash TEXT,
      name TEXT NOT NULL,
      tier TEXT DEFAULT 'free',
      role TEXT DEFAULT 'user',
      auth_provider TEXT DEFAULT 'local',
      auth_provider_id TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS app_settings (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS daily_usage (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      identifier TEXT NOT NULL,
      date_utc TEXT NOT NULL,
      query_count INTEGER DEFAULT 0,
      last_query_at TEXT NOT NULL,
      UNIQUE(identifier, date_utc)
    );

    CREATE INDEX IF NOT EXISTS idx_daily_usage_lookup ON daily_usage(identifier, date_utc);

    CREATE TABLE IF NOT EXISTS saved_scenarios (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      title TEXT NOT NULL,
      description TEXT DEFAULT '',
      tags TEXT DEFAULT '[]',
      active_summary TEXT NOT NULL,
      active_bar_chart TEXT NOT NULL,
      active_sankey_graph TEXT DEFAULT '{}',
      applied_proposals TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_saved_scenarios_user ON saved_scenarios(user_id);

    CREATE TABLE IF NOT EXISTS user_active_state (
      user_id TEXT PRIMARY KEY,
      active_summary TEXT,
      active_bar_chart TEXT,
      active_sankey_graph TEXT,
      applied_proposals TEXT NOT NULL DEFAULT '[]',
      updated_at TEXT NOT NULL,
      FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    );
  `);

  // Ensure default active_model is set to gpt-6-luna
  const currentModel = getSetting("active_model");
  if (!currentModel) {
    setSetting("active_model", "gpt-6-luna");
  }

  // Ensure samuelmartin.nh16@gmail.com is always role = 'admin' and tier = 'business'
  for (const adminEmail of ADMIN_EMAILS) {
    db.prepare(`
      UPDATE users SET role = 'admin', tier = 'business' WHERE LOWER(email) = ?
    `).run(adminEmail.toLowerCase());
  }

  console.log("[Database] GovSim database schemas verified and initialized successfully.");
}

// Run schema initialization immediately
initDatabase();

// ============================================================================
// User Operations
// ============================================================================

export function findUserByEmail(email: string): UserRecord | null {
  const row = db.prepare("SELECT * FROM users WHERE email = ?").get(email.toLowerCase().trim()) as any;
  if (!row) return null;
  const user = row as UserRecord;
  if (isAdminEmail(user.email)) {
    user.role = "admin";
    user.tier = "business";
  }
  return user;
}

export function findUserById(id: string): UserRecord | null {
  const row = db.prepare("SELECT * FROM users WHERE id = ?").get(id) as any;
  if (!row) return null;
  const user = row as UserRecord;
  if (isAdminEmail(user.email)) {
    user.role = "admin";
    user.tier = "business";
  }
  return user;
}

export function findUserByProvider(provider: string, providerId: string): UserRecord | null {
  const row = db
    .prepare("SELECT * FROM users WHERE auth_provider = ? AND auth_provider_id = ?")
    .get(provider, providerId) as any;
  if (!row) return null;
  const user = row as UserRecord;
  if (isAdminEmail(user.email)) {
    user.role = "admin";
    user.tier = "business";
  }
  return user;
}

export function createUser(user: {
  id: string;
  email: string;
  password_hash?: string | null;
  name: string;
  tier?: UserTier;
  role?: string;
  auth_provider?: "local" | "keycloak" | "google" | "apple";
  auth_provider_id?: string | null;
}): UserRecord {
  const now = new Date().toISOString();
  const emailNorm = user.email.toLowerCase().trim();
  const isAdmin = isAdminEmail(emailNorm);
  const newUser: UserRecord = {
    id: user.id,
    email: emailNorm,
    password_hash: user.password_hash || null,
    name: user.name,
    tier: isAdmin ? "business" : user.tier || "free",
    role: isAdmin ? "admin" : user.role || "user",
    auth_provider: user.auth_provider || "local",
    auth_provider_id: user.auth_provider_id || null,
    created_at: now,
    updated_at: now,
  };

  db.prepare(`
    INSERT INTO users (id, email, password_hash, name, tier, role, auth_provider, auth_provider_id, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).run(
    newUser.id,
    newUser.email,
    newUser.password_hash,
    newUser.name,
    newUser.tier,
    newUser.role,
    newUser.auth_provider,
    newUser.auth_provider_id,
    newUser.created_at,
    newUser.updated_at
  );

  return newUser;
}

export function updateUserTier(userId: string, tier: UserTier): void {
  const now = new Date().toISOString();
  db.prepare("UPDATE users SET tier = ?, updated_at = ? WHERE id = ?").run(tier, now, userId);
}

// ============================================================================
// Tier Limit & Rate Limiting Enforcement
// ============================================================================

export function checkAndIncrementUsage(identifier: string, tier: UserTier): UsageCheckResult {
  const now = new Date();
  const dateUtc = now.toISOString().slice(0, 10); // YYYY-MM-DD
  const limits = TIER_LIMITS[tier] || TIER_LIMITS.free;

  // Calculate hours until midnight UTC reset
  const midnightUtc = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() + 1));
  const resetInHours = Math.max(1, Math.ceil((midnightUtc.getTime() - now.getTime()) / (1000 * 60 * 60)));

  // For guests, we limit total queries across all time per IP/session
  const lookupDate = tier === "guest" ? "TOTAL_GUEST" : dateUtc;

  const existing = db
    .prepare("SELECT query_count FROM daily_usage WHERE identifier = ? AND date_utc = ?")
    .get(identifier, lookupDate) as { query_count: number } | undefined;

  const currentCount = existing ? existing.query_count : 0;

  if (currentCount >= limits.daily_queries) {
    let message = `Daily simulation limit reached (${currentCount}/${limits.daily_queries} queries used today).`;
    if (tier === "guest") {
      message = "You have used your 1 free guest simulation. Create a free account to get 10 simulations per day!";
    } else if (tier === "free") {
      message = "You have reached your free daily limit of 10 simulations. Upgrade to Pro for 250 queries/day or wait until UTC midnight reset.";
    }

    return {
      allowed: false,
      tier,
      current_count: currentCount,
      max_limit: limits.daily_queries,
      remaining: 0,
      reset_in_hours: resetInHours,
      message,
    };
  }

  // Increment usage count atomically
  const nowIso = now.toISOString();
  db.prepare(`
    INSERT INTO daily_usage (identifier, date_utc, query_count, last_query_at)
    VALUES (?, ?, 1, ?)
    ON CONFLICT(identifier, date_utc) DO UPDATE SET
      query_count = query_count + 1,
      last_query_at = excluded.last_query_at
  `).run(identifier, lookupDate, nowIso);

  const updatedCount = currentCount + 1;
  const remaining = Math.max(0, limits.daily_queries - updatedCount);

  return {
    allowed: true,
    tier,
    current_count: updatedCount,
    max_limit: limits.daily_queries,
    remaining,
    reset_in_hours: resetInHours,
  };
}

export function getTodayUsage(identifier: string, tier: UserTier): UsageCheckResult {
  const now = new Date();
  const dateUtc = now.toISOString().slice(0, 10);
  const limits = TIER_LIMITS[tier] || TIER_LIMITS.free;

  const midnightUtc = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() + 1));
  const resetInHours = Math.max(1, Math.ceil((midnightUtc.getTime() - now.getTime()) / (1000 * 60 * 60)));

  const lookupDate = tier === "guest" ? "TOTAL_GUEST" : dateUtc;

  const existing = db
    .prepare("SELECT query_count FROM daily_usage WHERE identifier = ? AND date_utc = ?")
    .get(identifier, lookupDate) as { query_count: number } | undefined;

  const currentCount = existing ? existing.query_count : 0;
  const remaining = Math.max(0, limits.daily_queries - currentCount);

  return {
    allowed: currentCount < limits.daily_queries,
    tier,
    current_count: currentCount,
    max_limit: limits.daily_queries,
    remaining,
    reset_in_hours: resetInHours,
  };
}

// ============================================================================
// Multi-Scenario Saves Operations (No JSON files required)
// ============================================================================

export function listUserScenarios(userId: string): Array<{
  id: string;
  title: string;
  description: string;
  tags: string[];
  applied_reforms_count: number;
  created_at: string;
  updated_at: string;
}> {
  const rows = db
    .prepare(
      `SELECT id, title, description, tags, applied_proposals, created_at, updated_at 
       FROM saved_scenarios 
       WHERE user_id = ? 
       ORDER BY updated_at DESC`
    )
    .all(userId) as any[];

  return rows.map((r) => {
    let reformCount = 0;
    try {
      const proposals = JSON.parse(r.applied_proposals);
      reformCount = Array.isArray(proposals) ? proposals.length : 0;
    } catch {
      reformCount = 0;
    }

    let parsedTags: string[] = [];
    try {
      parsedTags = JSON.parse(r.tags || "[]");
    } catch {
      parsedTags = [];
    }

    return {
      id: r.id,
      title: r.title,
      description: r.description || "",
      tags: parsedTags,
      applied_reforms_count: reformCount,
      created_at: r.created_at,
      updated_at: r.updated_at,
    };
  });
}

export function getScenarioById(scenarioId: string, userId: string): SavedScenarioRecord | null {
  const row = db
    .prepare("SELECT * FROM saved_scenarios WHERE id = ? AND user_id = ?")
    .get(scenarioId, userId) as any;

  if (!row) return null;

  return {
    id: row.id,
    user_id: row.user_id,
    title: row.title,
    description: row.description || "",
    tags: JSON.parse(row.tags || "[]"),
    applied_reforms_count: 0,
    active_summary: JSON.parse(row.active_summary || "{}"),
    active_bar_chart: JSON.parse(row.active_bar_chart || "{}"),
    active_sankey_graph: JSON.parse(row.active_sankey_graph || "{}"),
    applied_proposals: JSON.parse(row.applied_proposals || "[]"),
    created_at: row.created_at,
    updated_at: row.updated_at,
  };
}

export function countUserScenarios(userId: string): number {
  const row = db.prepare("SELECT COUNT(*) as count FROM saved_scenarios WHERE user_id = ?").get(userId) as {
    count: number;
  };
  return row ? row.count : 0;
}

export function saveNewScenario(scenario: {
  id: string;
  user_id: string;
  title: string;
  description?: string;
  tags?: string[];
  active_summary: any;
  active_bar_chart: any;
  active_sankey_graph: any;
  applied_proposals: any[];
}): SavedScenarioRecord {
  const now = new Date().toISOString();
  const tagsJson = JSON.stringify(scenario.tags || []);
  const summaryJson = JSON.stringify(scenario.active_summary || {});
  const chartJson = JSON.stringify(scenario.active_bar_chart || {});
  const sankeyJson = JSON.stringify(scenario.active_sankey_graph || {});
  const proposalsJson = JSON.stringify(scenario.applied_proposals || []);

  db.prepare(`
    INSERT INTO saved_scenarios (
      id, user_id, title, description, tags, active_summary, active_bar_chart, active_sankey_graph, applied_proposals, created_at, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).run(
    scenario.id,
    scenario.user_id,
    scenario.title,
    scenario.description || "",
    tagsJson,
    summaryJson,
    chartJson,
    sankeyJson,
    proposalsJson,
    now,
    now
  );

  return {
    ...scenario,
    description: scenario.description || "",
    tags: scenario.tags || [],
    applied_reforms_count: scenario.applied_proposals.length,
    created_at: now,
    updated_at: now,
  };
}

export function updateScenario(
  scenarioId: string,
  userId: string,
  updates: {
    title?: string;
    description?: string;
    active_summary?: any;
    active_bar_chart?: any;
    active_sankey_graph?: any;
    applied_proposals?: any[];
  }
): boolean {
  const existing = getScenarioById(scenarioId, userId);
  if (!existing) return false;

  const now = new Date().toISOString();
  const title = updates.title !== undefined ? updates.title : existing.title;
  const description = updates.description !== undefined ? updates.description : existing.description;
  const summaryJson =
    updates.active_summary !== undefined ? JSON.stringify(updates.active_summary) : JSON.stringify(existing.active_summary);
  const chartJson =
    updates.active_bar_chart !== undefined ? JSON.stringify(updates.active_bar_chart) : JSON.stringify(existing.active_bar_chart);
  const sankeyJson =
    updates.active_sankey_graph !== undefined
      ? JSON.stringify(updates.active_sankey_graph)
      : JSON.stringify(existing.active_sankey_graph);
  const proposalsJson =
    updates.applied_proposals !== undefined
      ? JSON.stringify(updates.applied_proposals)
      : JSON.stringify(existing.applied_proposals);

  db.prepare(`
    UPDATE saved_scenarios SET
      title = ?,
      description = ?,
      active_summary = ?,
      active_bar_chart = ?,
      active_sankey_graph = ?,
      applied_proposals = ?,
      updated_at = ?
    WHERE id = ? AND user_id = ?
  `).run(title, description, summaryJson, chartJson, sankeyJson, proposalsJson, now, scenarioId, userId);

  return true;
}

export function deleteScenario(scenarioId: string, userId: string): boolean {
  const result = db.prepare("DELETE FROM saved_scenarios WHERE id = ? AND user_id = ?").run(scenarioId, userId);
  return result.changes > 0;
}
