from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

PositiveFloat = Annotated[float, Field(ge=0.0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class SimulationMode(StrEnum):
    DETERMINISTIC = "Deterministic"
    MONTE_CARLO = "Monte Carlo"


class ScenarioInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=80)
    initial_investment: PositiveFloat
    daily_addition: PositiveFloat = 0.0
    daily_growth_pct: float = Field(ge=-99.0, le=1000.0)
    days: NonNegativeInt = Field(le=365_000)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("initial_investment", "daily_addition", "daily_growth_pct")
    @classmethod
    def ensure_finite_numeric_inputs(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Numeric inputs must be finite values")
        return value


class MonteCarloConfig(BaseModel):
    paths: Annotated[int, Field(ge=100, le=50_000)] = 2_000
    expected_daily_return_pct: float = Field(ge=-50.0, le=50.0)
    daily_volatility_pct: float = Field(ge=0.0, le=50.0)
    random_seed: Annotated[int, Field(ge=0, le=2_147_483_647)] = 42
    low_quantile: float = Field(default=0.1, gt=0.0, lt=0.5)
    high_quantile: float = Field(default=0.9, gt=0.5, lt=1.0)

    @field_validator(
        "expected_daily_return_pct",
        "daily_volatility_pct",
        "low_quantile",
        "high_quantile",
    )
    @classmethod
    def ensure_finite_monte_carlo_inputs(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Monte Carlo inputs must be finite values")
        return value


class AppConfig(BaseModel):
    mode: SimulationMode = SimulationMode.DETERMINISTIC
    trading_days_per_year: Annotated[int, Field(ge=1, le=366)] = 218
    monte_carlo: MonteCarloConfig = MonteCarloConfig(
        expected_daily_return_pct=0.2,
        daily_volatility_pct=1.4,
    )


class ScenarioSummary(BaseModel):
    scenario: str
    invested_capital: float
    final_value: float
    absolute_gain: float
    gain_pct: float
    annualized_return_pct: float
