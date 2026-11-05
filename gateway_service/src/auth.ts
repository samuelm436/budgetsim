/**
 * GovSim AI — Authentication & Identity Module
 * Supports:
 * 1. Keycloak OpenID Connect / OAuth2 Bearer Tokens
 * 2. Direct Social Login (Google & Apple ID tokens / federated OAuth)
 * 3. Local Email/Password with Bcrypt & JWT
 * 4. Automatic User Provisioning & Tier Synchronization into SQLite
 */

import crypto from "crypto";
import jwt from "jsonwebtoken";
import bcrypt from "bcryptjs";
import { FastifyRequest, FastifyReply } from "fastify";
import {
  UserRecord,
  UserTier,
  findUserByEmail,
  findUserById,
  findUserByProvider,
  createUser,
  updateUserTier,
} from "./db";

const JWT_SECRET = process.env.JWT_SECRET || "govsim-ai-secret-key-2027-change-in-production";
const JWT_EXPIRES_IN = "7d";

export interface AuthTokenPayload {
  userId: string;
  email: string;
  name: string;
  tier: UserTier;
  role: string;
  provider: string;
  iat?: number;
  exp?: number;
}

export interface AuthenticatedUser {
  id: string;
  email: string;
  name: string;
  tier: UserTier;
  role: string;
  provider: string;
}

// Extend FastifyRequest type to include optional authenticated user
declare module "fastify" {
  interface FastifyRequest {
    user?: AuthenticatedUser | null;
    clientIdentifier?: string;
  }
}

/**
 * Signs a JWT for a local or provisioned user
 */
export function signUserToken(user: UserRecord): string {
  const payload: AuthTokenPayload = {
    userId: user.id,
    email: user.email,
    name: user.name,
    tier: user.tier,
    role: user.role,
    provider: user.auth_provider,
  };
  return jwt.sign(payload, JWT_SECRET, { expiresIn: JWT_EXPIRES_IN });
}

/**
 * Validates a password against its bcrypt hash
 */
export async function verifyPassword(password: string, hash: string): Promise<boolean> {
  return bcrypt.compare(password, hash);
}

/**
 * Hashes a plaintext password
 */
export async function hashPassword(password: string): Promise<string> {
  const salt = await bcrypt.genSalt(10);
  return bcrypt.hash(password, salt);
}

/**
 * Resolves or auto-provisions a user from Keycloak / Social payload
 */
export function syncExternalUser(external: {
  email: string;
  name: string;
  provider: "keycloak" | "google" | "apple";
  providerId: string;
  tier?: UserTier;
}): UserRecord {
  // Check by provider ID first
  let user = findUserByProvider(external.provider, external.providerId);
  if (user) {
    if (external.tier && user.tier !== external.tier) {
      updateUserTier(user.id, external.tier);
      user.tier = external.tier;
    }
    return user;
  }

  // Check by email
  user = findUserByEmail(external.email);
  if (user) {
    if (external.tier && user.tier !== external.tier) {
      updateUserTier(user.id, external.tier);
      user.tier = external.tier;
    }
    return user;
  }

  // Create new user in SQLite
  const newId = crypto.randomUUID();
  return createUser({
    id: newId,
    email: external.email,
    name: external.name || external.email.split("@")[0],
    tier: external.tier || "free",
    role: "user",
    auth_provider: external.provider,
    auth_provider_id: external.providerId,
  });
}

/**
 * Verifies incoming Bearer Token:
 * Handles either GovSim internal JWT or external Keycloak / OpenID Connect JWT.
 */
export async function parseAndVerifyToken(token: string): Promise<AuthenticatedUser | null> {
  try {
    // 1. Try internal JWT verification
    const decoded = jwt.verify(token, JWT_SECRET) as AuthTokenPayload;
    if (decoded && decoded.userId) {
      const user = findUserById(decoded.userId);
      if (user) {
        return {
          id: user.id,
          email: user.email,
          name: user.name,
          tier: user.tier,
          role: user.role,
          provider: user.auth_provider,
        };
      }
    }
  } catch {
    // Not a local secret token or expired, try inspecting if Keycloak token
  }

  try {
    // 2. Decode token unverified header to see if it's from Keycloak or Google/Apple
    const unverified = jwt.decode(token) as any;
    if (unverified && (unverified.iss || unverified.sub)) {
      const isKeycloak =
        (unverified.iss && unverified.iss.includes("/realms/")) ||
        (process.env.KEYCLOAK_URL && unverified.iss && unverified.iss.includes(process.env.KEYCLOAK_URL));

      if (isKeycloak) {
        // Extract Keycloak user info
        const email = unverified.email || `${unverified.preferred_username || unverified.sub}@keycloak.local`;
        const name = unverified.name || unverified.preferred_username || "Keycloak User";
        const providerId = unverified.sub;

        // Check realm roles for tier assignments (e.g., 'tier_pro', 'pro', 'tier_business')
        const roles: string[] = unverified.realm_access?.roles || [];
        let tier: UserTier = "free";
        if (roles.includes("pro") || roles.includes("tier_pro") || roles.includes("journalist")) {
          tier = "pro";
        } else if (roles.includes("business") || roles.includes("enterprise")) {
          tier = "business";
        }

        const user = syncExternalUser({
          email,
          name,
          provider: "keycloak",
          providerId,
          tier,
        });

        return {
          id: user.id,
          email: user.email,
          name: user.name,
          tier: user.tier,
          role: user.role,
          provider: "keycloak",
        };
      }
    }
  } catch (kcErr) {
    console.warn("[Auth] Failed to parse token as Keycloak token:", kcErr);
  }

  return null;
}

/**
 * Middleware: Attaches user if valid token is provided, otherwise leaves as Guest.
 */
export async function optionalAuthHook(request: FastifyRequest, _reply: FastifyReply): Promise<void> {
  const authHeader = request.headers.authorization;
  let token: string | null = null;

  if (authHeader && authHeader.startsWith("Bearer ")) {
    token = authHeader.substring(7).trim();
  } else if ((request.query as any)?.access_token) {
    token = (request.query as any).access_token;
  }

  if (token) {
    const authUser = await parseAndVerifyToken(token);
    if (authUser) {
      request.user = authUser;
      request.clientIdentifier = `user:${authUser.id}`;
      return;
    }
  }

  // Fallback to Guest identifier (IP address or session header)
  const clientIp =
    (request.headers["x-forwarded-for"] as string)?.split(",")[0].trim() ||
    request.socket.remoteAddress ||
    "127.0.0.1";

  request.user = null;
  request.clientIdentifier = `guest_ip:${clientIp}`;
}

/**
 * Middleware: Requires the user to be logged in.
 */
export async function requireAuthHook(request: FastifyRequest, reply: FastifyReply): Promise<void> {
  await optionalAuthHook(request, reply);
  if (!request.user) {
    reply.status(401).send({
      error: {
        code: "UNAUTHORIZED",
        message: "You must be signed in to perform this action. Register or log in to manage your saved scenarios.",
      },
    });
  }
}

/**
 * Middleware: Requires the user to have administrator role.
 */
export async function requireAdminHook(request: FastifyRequest, reply: FastifyReply): Promise<void> {
  await optionalAuthHook(request, reply);
  if (!request.user) {
    reply.status(401).send({
      error: {
        code: "UNAUTHORIZED",
        message: "Sign in with your administrator account to access this control.",
      },
    });
    return;
  }

  if (request.user.role !== "admin") {
    reply.status(403).send({
      error: {
        code: "FORBIDDEN",
        message: "Access denied. Only the platform administrator (samuelmartin.nh16@gmail.com) can manage models and testing controls.",
      },
    });
  }
}
