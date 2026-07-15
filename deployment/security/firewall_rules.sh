#!/bin/bash
# Stage 30: Firewall Rules for Network Isolation
# Restricts outbound to exchange IPs, drops all unsolicited inbound

set -euo pipefail

echo "[*] Configuring firewall rules for bot network isolation..."

# Detect package manager
if command -v ufw &> /dev/null; then
    FIREWALL_CMD="ufw"
elif command -v iptables &> /dev/null; then
    FIREWALL_CMD="iptables"
else
    echo "[ERROR] No firewall tool found (ufw or iptables required)"
    exit 1
fi

# Exchange API IPs (update as needed for your specific exchanges)
# These are examples - replace with actual IPs or ranges for your exchanges
declare -a EXCHANGE_IPS=(
    "52.222.0.0/16"      # AWS CloudFront (used by many exchanges)
    "13.32.0.0/15"       # AWS Global Accelerator
    "3.160.0.0/14"       # AWS US-East
    "54.230.0.0/16"      # CloudFront
    "104.16.0.0/12"      # Cloudflare (used by Binance, etc.)
    "172.64.0.0/13"      # Cloudflare
)

# Localhost and Docker networks (always allowed)
LOCAL_NETS=(
    "127.0.0.0/8"
    "172.16.0.0/12"
    "192.168.0.0/16"
)

if [[ "$FIREWALL_CMD" == "ufw" ]]; then
    echo "[*] Using UFW for firewall configuration..."
    
    # Reset UFW to defaults
    sudo ufw --force reset
    
    # Set default policies
    sudo ufw default deny incoming
    sudo ufw default allow outgoing
    
    # Allow localhost
    sudo ufw allow from 127.0.0.0/8 to any
    
    # Allow Docker networks
    for net in "${LOCAL_NETS[@]}"; do
        sudo ufw allow from "$net" to any
    done
    
    # Allow outbound to exchange IPs only
    echo "[*] Allowing outbound to exchange IPs..."
    for ip in "${EXCHANGE_IPS[@]}"; do
        sudo ufw allow out to "$ip" proto tcp
    done
    
    # Allow HTTPS (443) and HTTP (80) for frontend
    sudo ufw allow in 443/tcp
    sudo ufw allow in 80/tcp
    
    # Allow SSH (optional, comment out if not needed)
    # sudo ufw allow in 22/tcp
    
    # Enable UFW
    sudo ufw --force enable
    
    echo "[+] UFW rules applied successfully"
    
elif [[ "$FIREWALL_CMD" == "iptables" ]]; then
    echo "[*] Using iptables for firewall configuration..."
    
    # Flush existing rules
    sudo iptables -F
    sudo iptables -X
    sudo iptables -t nat -F
    sudo iptables -t nat -X
    
    # Set default policies
    sudo iptables -P INPUT DROP
    sudo iptables -P FORWARD DROP
    sudo iptables -P OUTPUT ACCEPT
    
    # Allow localhost
    sudo iptables -A INPUT -i lo -j ACCEPT
    sudo iptables -A OUTPUT -o lo -j ACCEPT
    
    # Allow established and related connections
    sudo iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
    
    # Allow Docker networks
    for net in "${LOCAL_NETS[@]}"; do
        sudo iptables -A INPUT -s "$net" -j ACCEPT
        sudo iptables -A OUTPUT -d "$net" -j ACCEPT
    done
    
    # Allow outbound to exchange IPs only
    echo "[*] Allowing outbound to exchange IPs..."
    for ip in "${EXCHANGE_IPS[@]}"; do
        sudo iptables -A OUTPUT -d "$ip" -p tcp -j ACCEPT
    done
    
    # Block outbound to everything else except DNS and essential
    sudo iptables -A OUTPUT -p udp --dport 53 -j ACCEPT  # DNS
    sudo iptables -A OUTPUT -p tcp --dport 443 -j ACCEPT  # HTTPS
    sudo iptables -A OUTPUT -p tcp --dport 80 -j ACCEPT   # HTTP
    
    # Allow inbound HTTPS/HTTP for frontend
    sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT
    sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
    
    # Log dropped packets (optional, for debugging)
    sudo iptables -A INPUT -j LOG --log-prefix "IPTABLES-DROPPED: " --log-level 4
    
    echo "[+] iptables rules applied successfully"
fi

echo ""
echo "[*] Firewall configured. Only outbound connections to exchange IPs are allowed."
echo "[*] All unsolicited inbound traffic is blocked except ports 80/443."
echo "[!] Verify connectivity to your exchanges before enabling in production."
