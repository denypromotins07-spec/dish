"""
Bayesian optimization configuration using Optuna integrated with Ray Tune.
Defines hyperparameter spaces for strategies (SMC thresholds, TWAP slice sizes, Avellaneda-Stoikov risk aversion).
"""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterSpace:
    """Definition of a single hyperparameter space."""
    name: str
    param_type: str  # "float", "int", "categorical"
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[float] = None
    log: bool = False
    choices: Optional[List[Any]] = None
    default: Optional[Any] = None


@dataclass
class StrategyConfig:
    """Complete hyperparameter configuration for a strategy."""
    strategy_name: str
    spaces: List[HyperparameterSpace]
    objective_fn: Optional[Callable] = None


def create_smc_space() -> StrategyConfig:
    """
    Create hyperparameter space for Smart Money Concepts (SMC) strategy.
    
    Returns
    -------
    StrategyConfig
        Configuration with SMC-specific parameters.
    """
    return StrategyConfig(
        strategy_name="SMC",
        spaces=[
            HyperparameterSpace(
                name="order_block_threshold",
                param_type="float",
                low=0.001,
                high=0.05,
                log=True,
                default=0.01,
            ),
            HyperparameterSpace(
                name="fair_value_gap_min_size",
                param_type="float",
                low=0.0005,
                high=0.02,
                log=True,
                default=0.005,
            ),
            HyperparameterSpace(
                name="liquidity_sweep_threshold",
                param_type="float",
                low=0.001,
                high=0.03,
                default=0.01,
            ),
            HyperparameterSpace(
                name="break_of_structure_confirmation",
                param_type="categorical",
                choices=["aggressive", "moderate", "conservative"],
                default="moderate",
            ),
            HyperparameterSpace(
                name="mitigation_zone_depth",
                param_type="float",
                low=0.002,
                high=0.1,
                log=True,
                default=0.02,
            ),
        ],
    )


def create_twap_vwap_space() -> StrategyConfig:
    """
    Create hyperparameter space for TWAP/VWAP execution algorithms.
    
    Returns
    -------
    StrategyConfig
        Configuration with execution parameters.
    """
    return StrategyConfig(
        strategy_name="TWAP_VWAP",
        spaces=[
            HyperparameterSpace(
                name="slice_count",
                param_type="int",
                low=5,
                high=100,
                default=20,
            ),
            HyperparameterSpace(
                name="time_interval_seconds",
                param_type="int",
                low=10,
                high=600,
                default=60,
            ),
            HyperparameterSpace(
                name="volume_participation_rate",
                param_type="float",
                low=0.01,
                high=0.3,
                default=0.1,
            ),
            HyperparameterSpace(
                name="price_tolerance_bps",
                param_type="float",
                low=1.0,
                high=50.0,
                default=10.0,
            ),
            HyperparameterSpace(
                name="urgency_factor",
                param_type="float",
                low=0.1,
                high=2.0,
                default=1.0,
            ),
            HyperparameterSpace(
                name="randomization_factor",
                param_type="float",
                low=0.0,
                high=0.5,
                default=0.1,
            ),
        ],
    )


def create_avellaneda_stoikov_space() -> StrategyConfig:
    """
    Create hyperparameter space for Avellaneda-Stoikov market making.
    
    Returns
    -------
    StrategyConfig
        Configuration with MM parameters.
    """
    return StrategyConfig(
        strategy_name="Avellaneda_Stoikov",
        spaces=[
            HyperparameterSpace(
                name="risk_aversion_gamma",
                param_type="float",
                low=0.0001,
                high=0.1,
                log=True,
                default=0.01,
            ),
            HyperparameterSpace(
                name="inventory_skew_factor",
                param_type="float",
                low=0.0,
                high=0.5,
                default=0.1,
            ),
            HyperparameterSpace(
                name="max_inventory_units",
                param_type="int",
                low=10,
                high=1000,
                default=100,
            ),
            HyperparameterSpace(
                name="spread_multiplier",
                param_type="float",
                low=0.5,
                high=3.0,
                default=1.0,
            ),
            HyperparameterSpace(
                name="volatility_lookback_minutes",
                param_type="int",
                low=1,
                high=120,
                default=15,
            ),
            HyperparameterSpace(
                name="quote_refresh_ms",
                param_type="int",
                low=10,
                high=1000,
                default=100,
            ),
        ],
    )


def create_stat_arb_space() -> StrategyConfig:
    """
    Create hyperparameter space for Statistical Arbitrage / Pairs Trading.
    
    Returns
    -------
    StrategyConfig
        Configuration with stat arb parameters.
    """
    return StrategyConfig(
        strategy_name="StatArb",
        spaces=[
            HyperparameterSpace(
                name="z_score_entry",
                param_type="float",
                low=1.0,
                high=4.0,
                default=2.0,
            ),
            HyperparameterSpace(
                name="z_score_exit",
                param_type="float",
                low=0.5,
                high=2.0,
                default=1.0,
            ),
            HyperparameterSpace(
                name="lookback_period_days",
                param_type="int",
                low=10,
                high=252,
                default=60,
            ),
            HyperparameterSpace(
                name="hedge_ratio_update_frequency",
                param_type="categorical",
                choices=["daily", "hourly", "minute"],
                default="hourly",
            ),
            HyperparameterSpace(
                name="cointegration_pvalue_threshold",
                param_type="float",
                low=0.001,
                high=0.1,
                default=0.05,
            ),
            HyperparameterSpace(
                name="max_position_size_usd",
                param_type="float",
                low=1000,
                high=1000000,
                log=True,
                default=50000,
            ),
        ],
    )


def create_microstructure_space() -> StrategyConfig:
    """
    Create hyperparameter space for Market Microstructure strategies.
    
    Returns
    -------
    StrategyConfig
        Configuration with microstructure parameters.
    """
    return StrategyConfig(
        strategy_name="Microstructure",
        spaces=[
            HyperparameterSpace(
                name="vpin_threshold",
                param_type="float",
                low=0.3,
                high=0.9,
                default=0.7,
            ),
            HyperparameterSpace(
                name="bucket_size_ticks",
                param_type="int",
                low=10,
                high=1000,
                default=100,
            ),
            HyperparameterSpace(
                name="queue_position_alpha",
                param_type="float",
                low=0.1,
                high=0.9,
                default=0.5,
            ),
            HyperparameterSpace(
                name="toxic_flow_detection_window",
                param_type="int",
                low=5,
                high=100,
                default=20,
            ),
            HyperparameterSpace(
                name="spread_widen_factor",
                param_type="float",
                low=1.0,
                high=5.0,
                default=2.0,
            ),
        ],
    )


class BayesianOptimizer:
    """
    Bayesian optimization wrapper using Optuna with Ray integration.
    
    Features:
    - TPE sampler for efficient hyperparameter search
    - Median pruner for early stopping of poor trials
    - Ray Tune integration for distributed optimization
    - Custom objective functions with penalty terms
    """
    
    def __init__(
        self,
        study_name: str = "optimization_study",
        direction: str = "maximize",
        n_trials: int = 100,
        timeout_seconds: Optional[int] = None,
    ):
        self.study_name = study_name
        self.direction = direction
        self.n_trials = n_trials
        self.timeout_seconds = timeout_seconds
        
        # Initialize sampler and pruner
        self.sampler = TPESampler(seed=42)
        self.pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        
        self.study: Optional[optuna.Study] = None
        self.strategies: Dict[str, StrategyConfig] = {}
    
    def register_strategy(self, config: StrategyConfig):
        """Register a strategy configuration for optimization."""
        self.strategies[config.strategy_name] = config
        logger.info(f"Registered strategy: {config.strategy_name}")
    
    def create_study(self, storage: Optional[str] = None):
        """Create the Optuna study."""
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=optuna.study.StudyDirection.MAXIMIZE if self.direction == "maximize" else optuna.study.StudyDirection.MINIMIZE,
            sampler=self.sampler,
            pruner=self.pruner,
            storage=storage,
            load_if_exists=True,
        )
        logger.info(f"Created study: {self.study_name}")
    
    def suggest_parameters(self, trial: optuna.Trial, strategy_config: StrategyConfig) -> Dict[str, Any]:
        """
        Suggest parameters for a trial based on the strategy's hyperparameter space.
        
        Parameters
        ----------
        trial : optuna.Trial
            Current trial object.
        strategy_config : StrategyConfig
            Strategy configuration with parameter spaces.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of suggested parameters.
        """
        params = {}
        
        for space in strategy_config.spaces:
            if space.param_type == "float":
                params[space.name] = trial.suggest_float(
                    space.name,
                    space.low,
                    space.high,
                    step=space.step,
                    log=space.log,
                )
            elif space.param_type == "int":
                params[space.name] = trial.suggest_int(
                    space.name,
                    int(space.low),
                    int(space.high),
                )
            elif space.param_type == "categorical":
                params[space.name] = trial.suggest_categorical(
                    space.name,
                    space.choices,
                )
        
        return params
    
    def optimize(
        self,
        objective: Callable[[optuna.Trial], float],
        show_progress: bool = True,
    ) -> optuna.Study:
        """
        Run the optimization.
        
        Parameters
        ----------
        objective : Callable[[optuna.Trial], float]
            Objective function to maximize/minimize.
        show_progress : bool, default True
            Whether to show progress bar.
            
        Returns
        -------
        optuna.Study
            Completed study object.
        """
        if self.study is None:
            self.create_study()
        
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout_seconds,
            show_progress_bar=show_progress,
        )
        
        return self.study
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get the best parameters found."""
        if self.study is None or self.study.best_trial is None:
            return {}
        return self.study.best_params
    
    def get_best_value(self) -> float:
        """Get the best objective value."""
        if self.study is None or self.study.best_trial is None:
            return float('-inf') if self.direction == "maximize" else float('inf')
        return self.study.best_value
    
    def get_optimization_history(self) -> Dict[str, List]:
        """Get optimization history for analysis."""
        if self.study is None:
            return {'values': [], 'params': []}
        
        return {
            'values': [t.value for t in self.study.trials if t.value is not None],
            'params': [t.params for t in self.study.trials if t.params],
        }


def create_combined_strategy_space() -> List[StrategyConfig]:
    """
    Create hyperparameter spaces for all strategies.
    
    Returns
    -------
    List[StrategyConfig]
        List of all strategy configurations.
    """
    return [
        create_smc_space(),
        create_twap_vwap_space(),
        create_avellaneda_stoikov_space(),
        create_stat_arb_space(),
        create_microstructure_space(),
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create optimizer
    optimizer = BayesianOptimizer(
        study_name="crypto_bot_optimization",
        n_trials=50,
    )
    
    # Register all strategies
    for config in create_combined_strategy_space():
        optimizer.register_strategy(config)
    
    print(f"Registered {len(optimizer.strategies)} strategies:")
    for name, config in optimizer.strategies.items():
        print(f"  - {name}: {len(config.spaces)} parameters")
    
    # Example: Create study and show parameter suggestion
    optimizer.create_study()
    
    def dummy_objective(trial: optuna.Trial) -> float:
        smc_config = optimizer.strategies.get("SMC")
        if smc_config:
            params = optimizer.suggest_parameters(trial, smc_config)
            # Dummy objective
            return sum(params.values()) if params else 0.0
        return 0.0
    
    # Don't actually run optimization in this example
    print("\nOptimization setup complete. Call optimizer.optimize() to run.")
