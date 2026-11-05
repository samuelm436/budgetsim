/**
 * GovSim AI — Unified Multi-Service Local Orchestrator
 * Spawns and manages all components together in one single process:
 * 1. Python FastAPI Fiscal AI Engine (Port 8000)
 * 2. Keycloak 26.0 Identity Provider (Port 8080)
 * 3. Node.js Fastify API Gateway & Web Interface (Port 3000)
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const rootDir = __dirname;

// Set JAVA_HOME for Keycloak if not already set
if (!process.env.JAVA_HOME) {
  const javaJdkPath = 'C:\\Program Files\\Java\\jdk-24';
  if (fs.existsSync(javaJdkPath)) {
    process.env.JAVA_HOME = javaJdkPath;
  }
}

// Colors for terminal log prefixes
const colors = {
  reset: '\x1b[0m',
  cyan: '\x1b[36m',
  yellow: '\x1b[33m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  dim: '\x1b[2m',
};

function prefixLog(name, color, data) {
  const lines = data.toString().split('\n');
  for (const line of lines) {
    if (line.trim().length > 0) {
      console.log(`${color}[${name}]${colors.reset} ${line}`);
    }
  }
}

const processes = [];

function startProcess(name, color, command, args, options = {}) {
  const isWindows = process.platform === 'win32';
  const child = spawn(command, args, {
    shell: isWindows,
    cwd: options.cwd || rootDir,
    env: { ...process.env, ...options.env },
  });

  child.stdout.on('data', (d) => prefixLog(name, color, d));
  child.stderr.on('data', (d) => prefixLog(name, color, d));

  child.on('exit', (code, sig) => {
    if (code !== 0 && code !== null) {
      console.log(`${colors.red}[${name}] Exited with code ${code}${colors.reset}`);
    }
  });

  processes.push(child);
  return child;
}

console.log(`${colors.cyan}================================================================${colors.reset}`);
console.log(`${colors.cyan}  GovSim AI — Starting Unified Stack (All-in-One Process)       ${colors.reset}`);
console.log(`${colors.cyan}================================================================${colors.reset}`);
console.log(`  • Web App & Gateway:    http://localhost:3000`);
console.log(`  • Keycloak IAM & Admin: http://localhost:8080 (admin / admin)`);
console.log(`  • Fiscal AI Engine:     http://localhost:8000`);
console.log(`  • Persistent Database:  data/govsim.sqlite3`);
console.log(`${colors.dim}----------------------------------------------------------------${colors.reset}`);

// 1. Start Python Fiscal AI Engine
const pythonExe = path.join(rootDir, '.venv', 'Scripts', 'python.exe');
startProcess('FiscalEngine', colors.green, pythonExe, [
  '-m',
  'uvicorn',
  'fiscal_ai_engine.main:app',
  '--host',
  '127.0.0.1',
  '--port',
  '8000',
  '--reload',
]);

// 2. Start Keycloak 26.0 IAM Server (if installed)
const kcBat = path.join(rootDir, 'keycloak', 'keycloak-26.0.0', 'bin', 'kc.bat');
if (fs.existsSync(kcBat)) {
  // Ensure imported realm is up to date
  const importDir = path.join(rootDir, 'keycloak', 'keycloak-26.0.0', 'data', 'import');
  fs.mkdirSync(importDir, { recursive: true });
  fs.copyFileSync(
    path.join(rootDir, 'keycloak', 'govsim-realm.json'),
    path.join(importDir, 'govsim-realm.json')
  );

  startProcess('Keycloak', colors.yellow, kcBat, ['start-dev', '--import-realm'], {
    env: {
      KEYCLOAK_ADMIN: process.env.KEYCLOAK_ADMIN || 'admin',
      KEYCLOAK_ADMIN_PASSWORD: process.env.KEYCLOAK_ADMIN_PASSWORD || 'admin',
      KC_HTTP_PORT: '8080',
      KC_HOSTNAME_STRICT: 'false',
      KC_HTTP_ENABLED: 'true',
      JAVA_OPTS_APPEND: '-Dnet.bytebuddy.experimental=true',
    },
  });
} else {
  console.log(`${colors.yellow}[Keycloak] Not downloaded locally. Run 'node keycloak/setup-keycloak.js' to install standalone.${colors.reset}`);
}

// 3. Start Node.js Fastify Gateway
const gatewayDir = path.join(rootDir, 'gateway_service');
startProcess('Gateway', colors.cyan, 'npm', ['run', 'dev'], {
  cwd: gatewayDir,
  env: {
    PORT: '3000',
    FISCAL_ENGINE_URL: 'http://127.0.0.1:8000',
    KEYCLOAK_URL: 'http://localhost:8080',
    KEYCLOAK_REALM: 'govsim',
    DB_PATH: path.join(rootDir, 'data', 'govsim.sqlite3'),
  },
});

// Clean shutdown handler
function shutdown() {
  console.log(`\n${colors.yellow}Shutting down all GovSim services...${colors.reset}`);
  for (const p of processes) {
    try {
      if (process.platform === 'win32') {
        spawn('taskkill', ['/pid', p.pid, '/T', '/F']);
      } else {
        p.kill('SIGTERM');
      }
    } catch {}
  }
  process.exit(0);
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
process.on('exit', shutdown);
