from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DashboardConfig:
    default_regime: str = "baseline"
    default_strategy_a: str = "aggressive"
    default_strategy_b: str = "patient"
    default_seed: int = 7
    default_event_count: int = 500
    enabled_regimes: tuple[str, ...] = ("baseline", "trend", "whipsaw")
    enabled_strategies: tuple[str, ...] = ("aggressive", "patient")
    output_dir: str = "results/mvp"


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass(frozen=True)
class AppConfig:
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


def default_app_config() -> AppConfig:
    return AppConfig()
