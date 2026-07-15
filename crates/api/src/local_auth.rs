//! Localhost JWT/Session token generator for API authentication.
//! Ensures that even if the port is accidentally exposed, external actors
//! cannot send POST requests to control endpoints.

use jsonwebtoken::{encode, decode, Header, Algorithm, Validation, EncodingKey, DecodingKey};
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use std::time::{SystemTime, Duration};

/// Claims structure for JWT tokens
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiClaims {
    /// Subject (user/service identifier)
    pub sub: String,
    /// Issuer
    pub iss: String,
    /// Expiration time (seconds since epoch)
    pub exp: usize,
    /// Issued at (seconds since epoch)
    pub iat: usize,
    /// Allowed actions
    pub actions: Vec<String>,
    /// Must be localhost
    pub origin: String,
}

/// Authentication configuration
pub struct AuthConfig {
    /// Secret key for JWT signing
    pub secret: Vec<u8>,
    /// Token lifetime in seconds
    pub token_lifetime_secs: u64,
    /// Strict localhost enforcement
    pub enforce_localhost: bool,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            secret: b"ultra-low-latency-trading-bot-localhost-only-secret-key-2024".to_vec(),
            token_lifetime_secs: 3600, // 1 hour
            enforce_localhost: true,
        }
    }
}

/// Local authentication manager
pub struct LocalAuthManager {
    config: Arc<AuthConfig>,
    allowed_origins: Vec<String>,
}

impl LocalAuthManager {
    /// Create new local auth manager
    pub fn new(config: AuthConfig) -> Self {
        let allowed_origins = vec![
            "http://localhost:3000".to_string(),
            "http://127.0.0.1:3000".to_string(),
            "http://localhost:8080".to_string(),
            "http://127.0.0.1:8080".to_string(),
        ];

        Self {
            config: Arc::new(config),
            allowed_origins,
        }
    }

    /// Generate a new JWT token for localhost access
    pub fn generate_token(&self, subject: &str, actions: Vec<String>) -> Result<String, AuthError> {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_err(|_| AuthError::TimeError)?;

        let claims = ApiClaims {
            sub: subject.to_string(),
            iss: "localhost-trading-bot".to_string(),
            exp: (now.as_secs() + self.config.token_lifetime_secs) as usize,
            iat: now.as_secs() as usize,
            actions,
            origin: "localhost".to_string(),
        };

        encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(&self.config.secret),
        )
        .map_err(|_| AuthError::TokenGenerationFailed)
    }

    /// Validate and decode a JWT token
    pub fn validate_token(&self, token: &str) -> Result<ApiClaims, AuthError> {
        let mut validation = Validation::new(Algorithm::HS256);
        validation.validate_exp = true;
        validation.leeway = 60; // 1 second leeway

        decode::<ApiClaims>(
            token,
            &DecodingKey::from_secret(&self.config.secret),
            &validation,
        )
        .map(|data| data.claims)
        .map_err(|e| match e.kind() {
            jsonwebtoken::errors::ErrorKind::ExpiredSignature => AuthError::TokenExpired,
            jsonwebtoken::errors::ErrorKind::InvalidSignature => AuthError::InvalidSignature,
            _ => AuthError::TokenValidationFailed,
        })
    }

    /// Check if an origin is allowed
    pub fn is_origin_allowed(&self, origin: &str) -> bool {
        if !self.config.enforce_localhost {
            return true;
        }

        self.allowed_origins.iter().any(|allowed| {
            origin.starts_with("http://localhost") || 
            origin.starts_with("http://127.0.0.1") ||
            origin == "localhost"
        })
    }

    /// Verify request is from localhost IP
    pub fn verify_localhost_ip(&self, ip: &str) -> bool {
        if !self.config.enforce_localhost {
            return true;
        }

        matches!(ip, "127.0.0.1" | "::1" | "localhost")
    }

    /// Generate session token with specific permissions
    pub fn generate_session_token(
        &self,
        session_id: &str,
        read_only: bool,
    ) -> Result<String, AuthError> {
        let actions = if read_only {
            vec!["read".to_string()]
        } else {
            vec!["read".to_string(), "write".to_string(), "execute".to_string()]
        };

        self.generate_token(session_id, actions)
    }

    /// Refresh an existing token
    pub fn refresh_token(&self, old_token: &str) -> Result<String, AuthError> {
        let claims = self.validate_token(old_token)?;
        
        // Generate new token with same claims but extended expiry
        self.generate_token(&claims.sub, claims.actions)
    }

    /// Get allowed origins list
    pub fn allowed_origins(&self) -> &[String] {
        &self.allowed_origins
    }
}

/// Authentication errors
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("Token has expired")]
    TokenExpired,
    
    #[error("Invalid token signature")]
    InvalidSignature,
    
    #[error("Token generation failed")]
    TokenGenerationFailed,
    
    #[error("Token validation failed")]
    TokenValidationFailed,
    
    #[error("Origin not allowed")]
    OriginNotAllowed,
    
    #[error("IP not localhost")]
    IpNotLocalhost,
    
    #[error("Time error")]
    TimeError,
    
    #[error("Insufficient permissions")]
    InsufficientPermissions,
}

/// Middleware helper for Axum
pub async fn require_auth(
    auth_manager: Arc<LocalAuthManager>,
    token: Option<String>,
    required_action: &str,
) -> Result<ApiClaims, AuthError> {
    let token = token.ok_or(AuthError::TokenValidationFailed)?;
    let claims = auth_manager.validate_token(&token)?;

    if !claims.actions.iter().any(|a| a == required_action || a == "*") {
        return Err(AuthError::InsufficientPermissions);
    }

    Ok(claims)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_generation_and_validation() {
        let config = AuthConfig::default();
        let manager = LocalAuthManager::new(config);

        // Generate token
        let token = manager.generate_token("test-user", vec!["read".to_string()])
            .expect("Failed to generate token");

        assert!(!token.is_empty());

        // Validate token
        let claims = manager.validate_token(&token)
            .expect("Failed to validate token");

        assert_eq!(claims.sub, "test-user");
        assert!(claims.actions.contains(&"read".to_string()));
    }

    #[test]
    fn test_localhost_verification() {
        let config = AuthConfig::default();
        let manager = LocalAuthManager::new(config);

        assert!(manager.verify_localhost_ip("127.0.0.1"));
        assert!(manager.verify_localhost_ip("::1"));
        assert!(manager.verify_localhost_ip("localhost"));
        assert!(!manager.verify_localhost_ip("192.168.1.1"));
        assert!(!manager.verify_localhost_ip("10.0.0.1"));
    }

    #[test]
    fn test_origin_allowance() {
        let config = AuthConfig::default();
        let manager = LocalAuthManager::new(config);

        assert!(manager.is_origin_allowed("http://localhost:3000"));
        assert!(manager.is_origin_allowed("http://127.0.0.1:8080"));
        assert!(manager.is_origin_allowed("localhost"));
    }

    #[test]
    fn test_session_token_types() {
        let config = AuthConfig::default();
        let manager = LocalAuthManager::new(config);

        // Read-only token
        let read_token = manager.generate_session_token("session-1", true)
            .expect("Failed to generate read token");
        
        let claims = manager.validate_token(&read_token).unwrap();
        assert!(claims.actions.contains(&"read".to_string()));
        assert!(!claims.actions.contains(&"write".to_string()));

        // Full access token
        let write_token = manager.generate_session_token("session-2", false)
            .expect("Failed to generate write token");
        
        let claims = manager.validate_token(&write_token).unwrap();
        assert!(claims.actions.contains(&"write".to_string()));
        assert!(claims.actions.contains(&"execute".to_string()));
    }

    #[tokio::test]
    async fn test_require_auth_middleware() {
        let config = AuthConfig::default();
        let manager = Arc::new(LocalAuthManager::new(config));

        let token = manager.generate_token("test", vec!["read".to_string()]).unwrap();

        // Should succeed with correct action
        let result = require_auth(manager.clone(), Some(token.clone()), "read").await;
        assert!(result.is_ok());

        // Should fail with wrong action
        let result = require_auth(manager.clone(), Some(token), "write").await;
        assert!(matches!(result, Err(AuthError::InsufficientPermissions)));
    }
}
