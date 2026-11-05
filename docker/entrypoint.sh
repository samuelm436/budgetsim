#!/bin/bash
set -e

# 1. Ensure /app/data and Keycloak H2 subdirectories exist
mkdir -p /app/data/keycloak_h2
mkdir -p /opt/keycloak/data

# 2. Symlink Keycloak H2 directory to /app/data/keycloak_h2 so Keycloak state persists in the volume
if [ ! -L /opt/keycloak/data/h2 ]; then
  rm -rf /opt/keycloak/data/h2
  ln -s /app/data/keycloak_h2 /opt/keycloak/data/h2
fi

# 3. Ensure realm is present in import location
mkdir -p /opt/keycloak/data/import
cp /app/keycloak/govsim-realm.json /opt/keycloak/data/import/govsim-realm.json

# 4. Inject Google / Apple credentials if provided in container environment
if [ -n "$KEYCLOAK_GOOGLE_CLIENT_ID" ]; then
  sed -i "s|YOUR_GOOGLE_CLIENT_ID.apps.googleusercontent.com|${KEYCLOAK_GOOGLE_CLIENT_ID}|g" /opt/keycloak/data/import/govsim-realm.json
fi
if [ -n "$KEYCLOAK_GOOGLE_CLIENT_SECRET" ]; then
  sed -i "s|YOUR_GOOGLE_CLIENT_SECRET|${KEYCLOAK_GOOGLE_CLIENT_SECRET}|g" /opt/keycloak/data/import/govsim-realm.json
fi
if [ -n "$KEYCLOAK_APPLE_CLIENT_ID" ]; then
  sed -i "s|com.govsim.auth|${KEYCLOAK_APPLE_CLIENT_ID}|g" /opt/keycloak/data/import/govsim-realm.json
fi
if [ -n "$KEYCLOAK_APPLE_TEAM_ID" ]; then
  sed -i "s|YOUR_APPLE_TEAM_ID|${KEYCLOAK_APPLE_TEAM_ID}|g" /opt/keycloak/data/import/govsim-realm.json
fi
if [ -n "$KEYCLOAK_APPLE_KEY_ID" ]; then
  sed -i "s|YOUR_APPLE_KEY_ID|${KEYCLOAK_APPLE_KEY_ID}|g" /opt/keycloak/data/import/govsim-realm.json
fi

# 5. Start supervisord to launch Fiscal Engine, Keycloak, and Gateway
echo "Starting GovSim AI All-in-One Container..."
echo "  - Web App & Gateway: http://localhost:3000"
echo "  - Keycloak IAM:      http://localhost:8080 (Admin: admin / admin)"
echo "  - Persistent Volume: /app/data (SQLite + Keycloak H2)"

exec /usr/bin/supervisord -c /etc/supervisor/conf.d/govsim.conf
