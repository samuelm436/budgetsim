import sqlite3

conn = sqlite3.connect('data/govsim.sqlite3')
cur = conn.cursor()

# Create app_settings table for dynamic model configuration
cur.execute('''
CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
''')

# Set default active model to gpt-6-luna
cur.execute('''
INSERT OR REPLACE INTO app_settings (key, value, updated_at)
VALUES ('active_model', 'gpt-6-luna', datetime('now'))
''')

# Promote samuelmartin.nh16@gmail.com to admin & business tier
cur.execute('''
UPDATE users 
SET role = 'admin', tier = 'business' 
WHERE LOWER(email) = 'samuelmartin.nh16@gmail.com'
''')

conn.commit()

cur.execute("SELECT id, email, name, tier, role FROM users WHERE LOWER(email) = 'samuelmartin.nh16@gmail.com'")
row = cur.fetchone()
print("samuelmartin user updated:", row)

cur.execute("SELECT key, value FROM app_settings")
print("Settings:", cur.fetchall())

conn.close()
