#!/bin/bash
# Stage 30: mkcert SSL Setup for Localhost Security
# Generates trusted certificates for wss:// and https:// connections

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERTS_DIR="${SCRIPT_DIR}/certs"
DOMAINS=("localhost" "127.0.0.1" "::1")

echo "[*] Setting up local SSL certificates with mkcert..."

# Check if mkcert is installed
if ! command -v mkcert &> /dev/null; then
    echo "[!] mkcert not found. Installing..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt &> /dev/null; then
            sudo apt install -y mkcert libnss3-tools
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y mkcert nss-tools
        elif command -v pacman &> /dev/null; then
            sudo pacman -S --noconfirm mkcert nss
        else
            echo "[ERROR] Unsupported Linux distribution. Please install mkcert manually."
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install mkcert nss
    else
        echo "[ERROR] Unknown OS. Please install mkcert manually from https://github.com/FiloSottile/mkcert"
        exit 1
    fi
fi

# Create certs directory
mkdir -p "${CERTS_DIR}"

# Install local CA if not already done
echo "[*] Installing local CA..."
mkcert -install

# Generate certificate for all domains
echo "[*] Generating certificate for: ${DOMAINS[*]}"
cd "${CERTS_DIR}"
mkcert "${DOMAINS[@]}"

# Rename to expected filenames
if [ -f "localhost+4.pem" ]; then
    mv "localhost+4.pem" "localhost.crt" 2>/dev/null || true
fi
if [ -f "localhost+4-key.pem" ]; then
    mv "localhost+4-key.pem" "localhost.key" 2>/dev/null || true
fi

# Fallback: if standard naming exists
if [ ! -f "localhost.crt" ] && [ -f "localhost.pem" ]; then
    cp "localhost.pem" "localhost.crt"
fi
if [ ! -f "localhost.key" ] && [ -f "localhost-key.pem" ]; then
    cp "localhost-key.pem" "localhost.key"
fi

# Verify files exist
if [ ! -f "${CERTS_DIR}/localhost.crt" ] || [ ! -f "${CERTS_DIR}/localhost.key" ]; then
    echo "[ERROR] Certificate generation failed. Files not found."
    ls -la "${CERTS_DIR}"
    exit 1
fi

# Set secure permissions
chmod 600 "${CERTS_DIR}/localhost.key"
chmod 644 "${CERTS_DIR}/localhost.crt"

echo "[+] SSL certificates generated successfully:"
echo "    Certificate: ${CERTS_DIR}/localhost.crt"
echo "    Private Key: ${CERTS_DIR}/localhost.key"
echo ""
echo "[*] Your browser should now trust wss://localhost and https://localhost"
echo "[*] Restart your browser if you encounter certificate warnings."
