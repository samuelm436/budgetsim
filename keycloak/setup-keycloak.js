const fs = require('fs');
const path = require('path');
const https = require('https');
const { execSync } = require('child_process');

const targetDir = path.resolve(__dirname, 'keycloak-26.0.0');
const kcBat = path.join(targetDir, 'bin', 'kc.bat');

if (fs.existsSync(kcBat)) {
  console.log('Keycloak already installed at:', targetDir);
  process.exit(0);
}

const zipPath = path.resolve(__dirname, 'keycloak-26.0.0.zip');
console.log('Downloading Keycloak 26.0.0...');

function download(url, dest, cb) {
  https.get(url, (res) => {
    if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
      return download(res.headers.location, dest, cb);
    }
    const file = fs.createWriteStream(dest);
    res.pipe(file);
    file.on('finish', () => file.close(cb));
  }).on('error', (err) => {
    fs.unlink(dest, () => {});
    console.error('Download error:', err.message);
  });
}

download(
  'https://github.com/keycloak/keycloak/releases/download/26.0.0/keycloak-26.0.0.zip',
  zipPath,
  () => {
    console.log('Downloaded Keycloak zip. Extracting...');
    try {
      execSync(`tar -xf "${zipPath}" -C "${__dirname}"`, { stdio: 'inherit' });
      fs.unlinkSync(zipPath);
      console.log('Keycloak extracted successfully.');

      // Prepare import directory
      const importDir = path.join(targetDir, 'data', 'import');
      fs.mkdirSync(importDir, { recursive: true });
      fs.copyFileSync(path.join(__dirname, 'govsim-realm.json'), path.join(importDir, 'govsim-realm.json'));
      console.log('Realm govsim-realm.json copied to import directory.');
    } catch (e) {
      console.error('Extract error:', e.message);
    }
  }
);
