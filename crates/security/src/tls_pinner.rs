//! Custom TLS certificate pinning module for WebSocket and REST connections.
//! Prevents Man-in-the-Middle (MITM) attacks by strictly validating the exchange's SSL certificate fingerprint.

use sha2::{Sha256, Digest};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Represents a pinned certificate fingerprint (SHA-256 hash of the DER-encoded cert).
#[derive(Clone, Debug)]
pub struct PinnedCertificate {
    pub exchange: String,
    pub fingerprint: [u8; 32],
    pub subject_cn: String,
}

/// TLS Certificate Pinner for secure exchange connections.
pub struct TLSPinner {
    pinned_certs: Arc<RwLock<HashMap<String, PinnedCertificate>>>,
}

impl TLSPinner {
    /// Creates a new TLS pinner with pre-configured pinned certificates.
    pub fn new() -> Self {
        TLSPinner {
            pinned_certs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Pins a certificate for a specific exchange.
    pub async fn pin_certificate(&self, cert: PinnedCertificate) {
        let mut certs = self.pinned_certs.write().await;
        certs.insert(cert.exchange.clone(), cert);
    }

    /// Validates a certificate against the pinned fingerprint.
    pub async fn validate_certificate(
        &self,
        exchange: &str,
        der_encoded_cert: &[u8],
    ) -> Result<bool, String> {
        let certs = self.pinned_certs.read().await;
        
        let pinned = certs.get(exchange)
            .ok_or_else(|| format!("No pinned certificate found for exchange: {}", exchange))?;

        // Calculate SHA-256 fingerprint of the provided certificate
        let mut hasher = Sha256::new();
        hasher.update(der_encoded_cert);
        let computed_fingerprint: [u8; 32] = hasher.finalize().into();

        // Constant-time comparison to prevent timing attacks
        if self.constant_time_compare(&computed_fingerprint, &pinned.fingerprint) {
            Ok(true)
        } else {
            Err(format!(
                "Certificate pinning failed for {}. Expected: {:02x?}, Got: {:02x?}",
                exchange, pinned.fingerprint, computed_fingerprint
            ))
        }
    }

    /// Constant-time byte array comparison to prevent timing attacks.
    fn constant_time_compare(&self, a: &[u8], b: &[u8]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        
        let mut result: u8 = 0;
        for (x, y) in a.iter().zip(b.iter()) {
            result |= x ^ y;
        }
        result == 0
    }

    /// Creates a TLS connector with certificate pinning for hyper/reqwest.
    /// This is a conceptual example - actual implementation depends on the HTTP client.
    pub async fn create_pinned_connector(
        &self,
        exchange: &str,
    ) -> Result<PinnedTLSConnector, String> {
        let certs = self.pinned_certs.read().await;
        
        let pinned = certs.get(exchange)
            .ok_or_else(|| format!("No pinned certificate found for exchange: {}", exchange))?;
        
        Ok(PinnedTLSConnector {
            exchange: exchange.to_string(),
            fingerprint: pinned.fingerprint,
        })
    }

    /// Loads default pinned certificates for major exchanges.
    pub async fn load_default_pins(&self) {
        // These are example fingerprints - replace with actual production values
        let default_pins = vec![
            ("binance", "BINANCE_CERT_FINGERPRINT_HERE"),
            ("coinbase", "COINBASE_CERT_FINGERPRINT_HERE"),
            ("kraken", "KRAKEN_CERT_FINGERPRINT_HERE"),
            ("bybit", "BYBIT_CERT_FINGERPRINT_HERE"),
            ("deribit", "DERIBIT_CERT_FINGERPRINT_HERE"),
        ];

        for (exchange, fingerprint_hex) in default_pins {
            let fingerprint = hex_to_bytes(fingerprint_hex).unwrap_or([0u8; 32]);
            let cert = PinnedCertificate {
                exchange: exchange.to_string(),
                fingerprint,
                subject_cn: format!("*.{}.com", exchange),
            };
            self.pin_certificate(cert).await;
        }
    }
}

impl Default for TLSPinner {
    fn default() -> Self {
        Self::new()
    }
}

/// A TLS connector that enforces certificate pinning.
pub struct PinnedTLSConnector {
    exchange: String,
    fingerprint: [u8; 32],
}

impl PinnedTLSConnector {
    /// Returns the exchange name this connector is configured for.
    pub fn exchange(&self) -> &str {
        &self.exchange
    }

    /// Returns the expected certificate fingerprint.
    pub fn expected_fingerprint(&self) -> &[u8; 32] {
        &self.fingerprint
    }
}

/// Helper function to convert hex string to byte array.
fn hex_to_bytes(hex: &str) -> Option<[u8; 32]> {
    let hex = hex.replace(':', "").replace(' ', "");
    if hex.len() != 64 {
        return None;
    }
    
    let mut bytes = [0u8; 32];
    for i in 0..32 {
        bytes[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_certificate_pinning() {
        let pinner = TLSPinner::new();
        
        // Create a mock certificate
        let mock_cert_data = b"mock_certificate_data";
        let mut hasher = Sha256::new();
        hasher.update(mock_cert_data);
        let fingerprint: [u8; 32] = hasher.finalize().into();
        
        let cert = PinnedCertificate {
            exchange: "test_exchange".to_string(),
            fingerprint,
            subject_cn: "*.test.com".to_string(),
        };
        
        pinner.pin_certificate(cert).await;
        
        // Validate the same certificate
        let result = pinner.validate_certificate("test_exchange", mock_cert_data).await;
        assert!(result.is_ok());
        assert!(result.unwrap());
        
        // Validate a different certificate (should fail)
        let wrong_cert = b"wrong_certificate_data";
        let result = pinner.validate_certificate("test_exchange", wrong_cert).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_hex_to_bytes() {
        let hex = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let bytes = hex_to_bytes(hex).unwrap();
        assert_eq!(bytes.len(), 32);
        assert_eq!(bytes[0], 0x01);
        assert_eq!(bytes[31], 0xef);
    }
}
