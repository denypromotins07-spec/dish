#!/usr/bin/env python3
"""
Strict Configuration Validator.
Parses TOML/YAML files, checks for logical contradictions, and fails fast with descriptive errors.
"""

import os
import sys
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    import tomli  # Python 3.11+ has tomllib in stdlib
except ImportError:
    try:
        import tomllib as tomli
    except ImportError:
        tomli = None

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class ValidationError:
    """Represents a configuration validation error."""
    field: str
    message: str
    severity: str  # "error" or "warning"
    current_value: Any = None
    suggested_value: Any = None
    
    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.field}: {self.message}"


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0
    
    def raise_if_invalid(self) -> None:
        """Raise exception if validation failed."""
        if self.has_errors:
            error_msgs = "\n".join(str(e) for e in self.errors)
            raise ConfigurationError(f"Configuration validation failed:\n{error_msgs}")


class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass


class ConfigValidator:
    """Strict configuration validator for trading bot."""
    
    # Valid exchange names
    VALID_EXCHANGES = {"binance", "coinbase", "okx", "kraken", "bybit", "ftx"}
    
    # Exchange-specific leverage limits
    MAX_LEVERAGE_BY_EXCHANGE = {
        "binance": 10.0,
        "coinbase": 5.0,
        "okx": 20.0,
        "kraken": 5.0,
        "bybit": 50.0,
    }
    
    # Valid risk limit range
    MIN_RISK_LIMIT_PCT = 0.5
    MAX_RISK_LIMIT_PCT = 5.0
    
    # Valid symbols pattern
    SYMBOL_PATTERN = re.compile(r'^[A-Z]{3,6}(-USD|-USDT|-BTC|-ETH)?$')
    
    def __init__(self):
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
        
    def validate_file(self, filepath: str) -> ValidationResult:
        """Validate a configuration file (TOML or YAML)."""
        path = Path(filepath)
        
        if not path.exists():
            self.errors.append(ValidationError(
                field="file",
                message=f"Configuration file not found: {filepath}",
                severity="error"
            ))
            return ValidationResult(False, self.errors, self.warnings)
        
        # Parse file based on extension
        config = self._parse_file(path)
        if config is None:
            return ValidationResult(False, self.errors, self.warnings)
        
        # Run all validations
        self._validate_exchange(config)
        self._validate_api_credentials(config)
        self._validate_symbols(config)
        self._validate_leverage(config)
        self._validate_risk_limits(config)
        self._validate_strategy_config(config)
        self._validate_execution_config(config)
        self._validate_memory_limits(config)
        
        return ValidationResult(
            is_valid=not self.has_errors,
            errors=self.errors,
            warnings=self.warnings
        )
    
    def _parse_file(self, path: Path) -> Optional[Dict[str, Any]]:
        """Parse configuration file based on extension."""
        suffix = path.suffix.lower()
        
        try:
            with open(path, 'rb') as f:
                content = f.read()
        except IOError as e:
            self.errors.append(ValidationError(
                field="file",
                message=f"Cannot read file: {e}",
                severity="error"
            ))
            return None
        
        if suffix in ('.toml',):
            if tomli is None:
                self.errors.append(ValidationError(
                    field="file",
                    message="TOML parsing requires 'tomli' package. Install with: pip install tomli",
                    severity="error"
                ))
                return None
            
            try:
                return tomli.loads(content.decode('utf-8'))
            except Exception as e:
                self.errors.append(ValidationError(
                    field="file",
                    message=f"TOML parse error: {e}",
                    severity="error"
                ))
                return None
                
        elif suffix in ('.yaml', '.yml'):
            if yaml is None:
                self.errors.append(ValidationError(
                    field="file",
                    message="YAML parsing requires 'pyyaml' package. Install with: pip install pyyaml",
                    severity="error"
                ))
                return None
            
            try:
                return yaml.safe_load(content.decode('utf-8'))
            except Exception as e:
                self.errors.append(ValidationError(
                    field="file",
                    message=f"YAML parse error: {e}",
                    severity="error"
                ))
                return None
        else:
            self.errors.append(ValidationError(
                field="file",
                message=f"Unsupported file extension: {suffix}. Use .toml or .yaml",
                severity="error"
            ))
            return None
    
    def _validate_exchange(self, config: Dict[str, Any]) -> None:
        """Validate exchange configuration."""
        exchange = config.get('exchange', '').lower()
        
        if not exchange:
            self.errors.append(ValidationError(
                field="exchange",
                message="Exchange is required",
                severity="error"
            ))
        elif exchange not in self.VALID_EXCHANGES:
            self.errors.append(ValidationError(
                field="exchange",
                message=f"Invalid exchange '{exchange}'. Must be one of: {', '.join(self.VALID_EXCHANGES)}",
                severity="error",
                current_value=exchange,
                suggested_value=list(self.VALID_EXCHANGES)[0]
            ))
    
    def _validate_api_credentials(self, config: Dict[str, Any]) -> None:
        """Validate API credentials."""
        api_key = config.get('api_key', '')
        api_secret = config.get('api_secret', '')
        
        # Check if credentials are set (preferably via environment)
        if not api_key and not os.environ.get('EXCHANGE_API_KEY'):
            self.warnings.append(ValidationError(
                field="api_key",
                message="API key not set. Will run in simulation mode.",
                severity="warning"
            ))
        
        if not api_secret and not os.environ.get('EXCHANGE_API_SECRET'):
            self.warnings.append(ValidationError(
                field="api_secret",
                message="API secret not set. Will run in simulation mode.",
                severity="warning"
            ))
        
        # Validate key format if provided
        if api_key and len(api_key) < 10:
            self.errors.append(ValidationError(
                field="api_key",
                message="API key appears too short",
                severity="error",
                current_value=api_key[:4] + "..." if api_key else None
            ))
    
    def _validate_symbols(self, config: Dict[str, Any]) -> None:
        """Validate trading symbols."""
        symbols = config.get('symbols', [])
        
        if not symbols:
            self.errors.append(ValidationError(
                field="symbols",
                message="At least one trading symbol is required",
                severity="error"
            ))
            return
        
        invalid_symbols = []
        for symbol in symbols:
            if not self.SYMBOL_PATTERN.match(symbol):
                invalid_symbols.append(symbol)
        
        if invalid_symbols:
            self.errors.append(ValidationError(
                field="symbols",
                message=f"Invalid symbol format: {', '.join(invalid_symbols)}. Expected format like BTC-USD, ETH-USDT",
                severity="error",
                current_value=invalid_symbols
            ))
    
    def _validate_leverage(self, config: Dict[str, Any]) -> None:
        """Validate leverage settings."""
        leverage = config.get('max_leverage', 1.0)
        exchange = config.get('exchange', '').lower()
        
        if leverage < 1.0:
            self.errors.append(ValidationError(
                field="max_leverage",
                message="Leverage must be at least 1.0",
                severity="error",
                current_value=leverage,
                suggested_value=1.0
            ))
        
        # Check against exchange limits
        if exchange in self.MAX_LEVERAGE_BY_EXCHANGE:
            max_allowed = self.MAX_LEVERAGE_BY_EXCHANGE[exchange]
            if leverage > max_allowed:
                self.errors.append(ValidationError(
                    field="max_leverage",
                    message=f"Leverage {leverage} exceeds {exchange}'s limit of {max_allowed}",
                    severity="error",
                    current_value=leverage,
                    suggested_value=max_allowed
                ))
    
    def _validate_risk_limits(self, config: Dict[str, Any]) -> None:
        """Validate risk limit configuration."""
        risk_limit = config.get('risk_limit_pct', 1.0)
        
        if risk_limit < self.MIN_RISK_LIMIT_PCT:
            self.errors.append(ValidationError(
                field="risk_limit_pct",
                message=f"Risk limit too low. Minimum is {self.MIN_RISK_LIMIT_PCT}%",
                severity="error",
                current_value=risk_limit,
                suggested_value=self.MIN_RISK_LIMIT_PCT
            ))
        
        if risk_limit > self.MAX_RISK_LIMIT_PCT:
            self.errors.append(ValidationError(
                field="risk_limit_pct",
                message=f"Risk limit too high. Maximum is {self.MAX_RISK_LIMIT_PCT}%",
                severity="error",
                current_value=risk_limit,
                suggested_value=self.MAX_RISK_LIMIT_PCT
            ))
    
    def _validate_strategy_config(self, config: Dict[str, Any]) -> None:
        """Validate strategy configuration."""
        strategies = config.get('strategies', [])
        
        for i, strategy in enumerate(strategies):
            name = strategy.get('name', f'strategy_{i}')
            
            if 'enabled' not in strategy:
                self.warnings.append(ValidationError(
                    field=f"strategies[{i}].enabled",
                    message=f"Strategy '{name}' does not specify enabled status",
                    severity="warning"
                ))
            
            # Check for contradictory settings
            if strategy.get('mode') == 'live' and not strategy.get('paper_trading', True):
                self.warnings.append(ValidationError(
                    field=f"strategies[{i}].mode",
                    message=f"Strategy '{name}' is in live mode without paper trading fallback",
                    severity="warning"
                ))
    
    def _validate_execution_config(self, config: Dict[str, Any]) -> None:
        """Validate execution configuration."""
        execution = config.get('execution', {})
        
        timeout_ms = execution.get('order_timeout_ms', 5000)
        if timeout_ms < 100:
            self.errors.append(ValidationError(
                field="execution.order_timeout_ms",
                message="Order timeout too low (< 100ms may cause issues)",
                severity="error",
                current_value=timeout_ms,
                suggested_value=5000
            ))
        
        retries = execution.get('max_retries', 3)
        if retries < 0:
            self.errors.append(ValidationError(
                field="execution.max_retries",
                message="Max retries cannot be negative",
                severity="error",
                current_value=retries,
                suggested_value=3
            ))
    
    def _validate_memory_limits(self, config: Dict[str, Any]) -> None:
        """Validate memory limit configuration (strict 14GB constraint)."""
        memory = config.get('memory', {})
        
        max_ram_gb = memory.get('max_ram_gb', 14.0)
        if max_ram_gb > 14.0:
            self.errors.append(ValidationError(
                field="memory.max_ram_gb",
                message=f"Memory limit exceeds strict 14GB constraint",
                severity="error",
                current_value=max_ram_gb,
                suggested_value=14.0
            ))
        
        # Check subsystem allocations sum to <= 14
        subsystems = memory.get('subsystems', {})
        total_allocated = sum(subsystems.values()) if isinstance(subsystems, dict) else 0
        
        if total_allocated > 14.0:
            self.errors.append(ValidationError(
                field="memory.subsystems",
                message=f"Total subsystem allocation ({total_allocated}GB) exceeds 14GB limit",
                severity="error",
                current_value=total_allocated,
                suggested_value=14.0
            ))


def main():
    """Main entry point for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration Validator")
    parser.add_argument("config_file", help="Path to configuration file (TOML or YAML)")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    
    args = parser.parse_args()
    
    validator = ConfigValidator()
    result = validator.validate_file(args.config_file)
    
    if args.json:
        import json
        output = {
            "is_valid": result.is_valid,
            "errors": [
                {"field": e.field, "message": e.message, "severity": e.severity}
                for e in result.errors
            ],
            "warnings": [
                {"field": w.field, "message": w.message, "severity": w.severity}
                for w in result.warnings
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        # Human-readable output
        if result.has_errors:
            print("\n❌ CONFIGURATION ERRORS:")
            for error in result.errors:
                print(f"  {error}")
        
        if result.has_warnings:
            print("\n⚠️  WARNINGS:")
            for warning in result.warnings:
                print(f"  {warning}")
        
        if result.is_valid and not result.has_warnings:
            print("\n✅ Configuration is valid!")
        elif result.is_valid:
            print("\n✅ Configuration is valid (with warnings)")
    
    # Exit code
    if result.has_errors:
        sys.exit(1)
    elif args.strict and result.has_warnings:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
