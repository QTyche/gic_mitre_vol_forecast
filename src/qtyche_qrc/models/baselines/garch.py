"""Deterministic stationary Gaussian GARCH(1,1) estimation and forecasting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize  # type: ignore[import-untyped]

LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class GARCHParameters:
    """Stationary Gaussian GARCH(1,1) parameters in daily-return units."""

    omega: float
    alpha: float
    beta: float
    mu: float

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def unconditional_variance(self) -> float:
        return self.omega / (1.0 - self.persistence)

    def validate(self) -> None:
        values = (self.omega, self.alpha, self.beta, self.mu)
        if not np.isfinite(values).all():
            raise ValueError("GARCH parameters must be finite")
        if self.omega <= 0:
            raise ValueError("GARCH omega must be positive")
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("GARCH alpha and beta must be non-negative")
        if self.persistence >= 1.0:
            raise ValueError("GARCH alpha + beta must be strictly below one")
        if not np.isfinite(self.unconditional_variance) or self.unconditional_variance <= 0:
            raise ValueError("GARCH unconditional variance must be finite and positive")


@dataclass(frozen=True)
class GARCHOptimizationAttempt:
    """One fixed-start optimiser outcome."""

    start_index: int
    initial_parameters: dict[str, float]
    success: bool
    status: int
    message: str
    negative_log_likelihood: float | None
    number_of_iterations: int
    function_evaluations: int
    fitted_parameters: dict[str, float] | None


@dataclass(frozen=True)
class GARCHFitResult:
    """Selected converged fit plus complete deterministic optimisation evidence."""

    parameters: GARCHParameters
    training_log_likelihood: float
    selected_start_index: int
    optimiser_status: int
    optimiser_message: str
    number_of_iterations: int
    function_evaluations: int
    training_return_count: int
    training_return_mean: float
    training_return_variance: float
    initial_conditional_variance: float
    final_conditional_variance: float
    next_conditional_variance: float
    convergence_warnings: tuple[str, ...]
    attempts: tuple[GARCHOptimizationAttempt, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "parameters": {
                **asdict(self.parameters),
                "persistence": self.parameters.persistence,
                "unconditional_variance": self.parameters.unconditional_variance,
            },
        }


@dataclass(frozen=True)
class GARCHForecastPath:
    """Causal filtered state and forecasts for each supplied post-fit return."""

    filtered_variance_at_origin: NDArray[np.float64]
    one_day_variance: NDArray[np.float64]
    five_day_cumulative_variance: NDArray[np.float64]
    target_unit_forecast: NDArray[np.float64]


class GaussianGARCH11:
    """Gaussian QML GARCH(1,1) with positivity and stationarity by construction."""

    def __init__(
        self,
        *,
        horizon: int = 5,
        annualization: float = 252.0,
        stationarity_margin: float = 1e-6,
        variance_floor: float = 1e-12,
        maximum_iterations: int = 2000,
        tolerance: float = 1e-10,
    ) -> None:
        if horizon != 5:
            raise ValueError("the frozen realized-variance target requires horizon=5")
        if annualization <= 0:
            raise ValueError("annualization must be positive")
        if not 0 < stationarity_margin < 0.1:
            raise ValueError("stationarity_margin must lie in (0, 0.1)")
        if variance_floor <= 0:
            raise ValueError("variance_floor must be positive")
        if maximum_iterations <= 0 or tolerance <= 0:
            raise ValueError("optimiser controls must be positive")
        self.horizon = horizon
        self.annualization = annualization
        self.stationarity_margin = stationarity_margin
        self.variance_floor = variance_floor
        self.maximum_iterations = maximum_iterations
        self.tolerance = tolerance
        self.fit_result: GARCHFitResult | None = None

    @property
    def parameters(self) -> GARCHParameters:
        if self.fit_result is None:
            raise RuntimeError("GARCH model is not fitted")
        return self.fit_result.parameters

    def _transform_parameters(self, values: NDArray[np.float64]) -> GARCHParameters:
        theta = np.asarray(values, dtype=float).reshape(-1)
        if theta.shape != (4,) or not np.isfinite(theta).all():
            raise ValueError("unconstrained GARCH parameter vector must contain four values")
        omega = self.variance_floor + float(np.logaddexp(0.0, theta[0]))
        logits = np.asarray([theta[1], theta[2], 0.0], dtype=float)
        logits -= float(logits.max())
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        scale = 1.0 - self.stationarity_margin
        parameters = GARCHParameters(
            omega=omega,
            alpha=float(scale * probabilities[0]),
            beta=float(scale * probabilities[1]),
            mu=float(theta[3]),
        )
        parameters.validate()
        return parameters

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        if value <= 0:
            raise ValueError("softplus inverse requires a positive value")
        if value > 20:
            return value
        return float(np.log(np.expm1(value)))

    def _unconstrained_start(self, parameters: GARCHParameters) -> NDArray[np.float64]:
        parameters.validate()
        scale = 1.0 - self.stationarity_margin
        alpha_probability = parameters.alpha / scale
        beta_probability = parameters.beta / scale
        remainder = 1.0 - alpha_probability - beta_probability
        if min(alpha_probability, beta_probability, remainder) <= 0:
            raise ValueError("fixed GARCH start is incompatible with transformed constraints")
        omega_value = max(parameters.omega - self.variance_floor, self.variance_floor)
        return np.asarray(
            [
                self._inverse_softplus(omega_value),
                np.log(alpha_probability / remainder),
                np.log(beta_probability / remainder),
                parameters.mu,
            ],
            dtype=float,
        )

    @staticmethod
    def _validate_returns(returns: NDArray[np.float64]) -> NDArray[np.float64]:
        values = np.asarray(returns, dtype=float).reshape(-1)
        if len(values) < 50:
            raise ValueError("GARCH estimation requires at least 50 training returns")
        if not np.isfinite(values).all():
            raise ValueError("GARCH returns must be finite")
        if float(np.var(values, ddof=0)) <= 0:
            raise ValueError("GARCH training returns must have positive variance")
        return values

    @staticmethod
    def _filter_with_parameters(
        returns: NDArray[np.float64],
        parameters: GARCHParameters,
    ) -> tuple[NDArray[np.float64], float]:
        parameters.validate()
        values = np.asarray(returns, dtype=float).reshape(-1)
        variances = np.empty(len(values), dtype=float)
        variances[0] = parameters.unconditional_variance
        residuals = values - parameters.mu
        for index in range(1, len(values)):
            variances[index] = (
                parameters.omega
                + parameters.alpha * residuals[index - 1] ** 2
                + parameters.beta * variances[index - 1]
            )
        next_variance = (
            parameters.omega
            + parameters.alpha * residuals[-1] ** 2
            + parameters.beta * variances[-1]
        )
        return variances, float(next_variance)

    def _negative_log_likelihood(
        self,
        unconstrained: NDArray[np.float64],
        returns: NDArray[np.float64],
    ) -> float:
        try:
            parameters = self._transform_parameters(unconstrained)
            variances, _ = self._filter_with_parameters(returns, parameters)
        except (FloatingPointError, OverflowError, ValueError):
            return 1e100
        if (
            not np.isfinite(variances).all()
            or np.any(variances <= self.variance_floor)
            or float(variances.max()) > 1e6
        ):
            return 1e100
        residuals = returns - parameters.mu
        terms = LOG_2PI + np.log(variances) + residuals**2 / variances
        value = 0.5 * float(np.sum(terms))
        return value if np.isfinite(value) else 1e100

    def _fixed_starts(
        self,
        returns: NDArray[np.float64],
    ) -> tuple[GARCHParameters, ...]:
        sample_mean = float(np.mean(returns))
        sample_variance = float(np.var(returns, ddof=0))
        shapes = (
            (0.05, 0.90, sample_mean),
            (0.10, 0.80, sample_mean),
            (0.15, 0.70, sample_mean),
            (0.03, 0.95, sample_mean),
            (0.20, 0.60, sample_mean),
            (0.05, 0.90, 0.0),
            (0.10, 0.80, 0.0),
            (0.20, 0.60, 0.0),
        )
        return tuple(
            GARCHParameters(
                omega=max(sample_variance * (1.0 - alpha - beta), self.variance_floor * 10),
                alpha=alpha,
                beta=beta,
                mu=mu,
            )
            for alpha, beta, mu in shapes
        )

    def fit(
        self,
        training_returns: NDArray[np.float64],
        *,
        maximum_starts: int | None = None,
    ) -> GARCHFitResult:
        """Fit fixed parameters once using only the supplied training returns."""

        returns = self._validate_returns(training_returns)
        starts = self._fixed_starts(returns)
        if maximum_starts is not None:
            if maximum_starts <= 0:
                raise ValueError("maximum_starts must be positive")
            starts = starts[:maximum_starts]
        attempts: list[GARCHOptimizationAttempt] = []
        successful: list[tuple[float, int, Any, GARCHParameters]] = []
        bounds = ((-30.0, -2.0), (-20.0, 20.0), (-20.0, 20.0), (-0.1, 0.1))
        for index, initial in enumerate(starts):
            unconstrained = self._unconstrained_start(initial)
            result = minimize(
                self._negative_log_likelihood,
                unconstrained,
                args=(returns,),
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": self.maximum_iterations,
                    "ftol": self.tolerance,
                    "gtol": self.tolerance,
                    "maxls": 50,
                },
            )
            fitted: GARCHParameters | None = None
            objective: float | None = None
            valid = False
            try:
                fitted = self._transform_parameters(np.asarray(result.x, dtype=float))
                objective = float(result.fun)
                valid = bool(result.success and np.isfinite(objective) and objective < 1e90)
            except (FloatingPointError, OverflowError, ValueError):
                valid = False
            attempts.append(
                GARCHOptimizationAttempt(
                    start_index=index,
                    initial_parameters=asdict(initial),
                    success=valid,
                    status=int(result.status),
                    message=str(result.message),
                    negative_log_likelihood=(
                        objective if objective is not None and np.isfinite(objective) else None
                    ),
                    number_of_iterations=int(result.nit),
                    function_evaluations=int(result.nfev),
                    fitted_parameters=asdict(fitted) if fitted is not None else None,
                )
            )
            if valid and fitted is not None and objective is not None:
                successful.append((objective, index, result, fitted))
        if not successful:
            messages = "; ".join(
                f"start {attempt.start_index}: {attempt.message}" for attempt in attempts
            )
            raise RuntimeError(f"all deterministic GARCH fits failed to converge: {messages}")
        objective, selected_index, selected_result, parameters = min(
            successful, key=lambda item: (item[0], item[1])
        )
        parameters.validate()
        variances, next_variance = self._filter_with_parameters(returns, parameters)
        warnings = tuple(
            f"start {attempt.start_index} did not converge: {attempt.message}"
            for attempt in attempts
            if not attempt.success
        )
        fit_result = GARCHFitResult(
            parameters=parameters,
            training_log_likelihood=-objective,
            selected_start_index=selected_index,
            optimiser_status=int(selected_result.status),
            optimiser_message=str(selected_result.message),
            number_of_iterations=int(selected_result.nit),
            function_evaluations=int(selected_result.nfev),
            training_return_count=len(returns),
            training_return_mean=float(np.mean(returns)),
            training_return_variance=float(np.var(returns, ddof=0)),
            initial_conditional_variance=float(variances[0]),
            final_conditional_variance=float(variances[-1]),
            next_conditional_variance=next_variance,
            convergence_warnings=warnings,
            attempts=tuple(attempts),
        )
        self.fit_result = fit_result
        return fit_result

    def filter_training_returns(
        self,
        training_returns: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float]:
        """Filter a training sequence with already fitted, fixed parameters."""

        return self._filter_with_parameters(
            self._validate_returns(training_returns),
            self.parameters,
        )

    def one_step_variance(
        self,
        current_variance: float,
        observed_return: float,
    ) -> float:
        """Return h_(t+1) using only h_t and the return observed at origin t."""

        if not np.isfinite(current_variance) or current_variance <= 0:
            raise ValueError("current conditional variance must be finite and positive")
        if not np.isfinite(observed_return):
            raise ValueError("observed return must be finite")
        parameters = self.parameters
        residual = observed_return - parameters.mu
        value = (
            parameters.omega + parameters.alpha * residual**2 + parameters.beta * current_variance
        )
        if not np.isfinite(value) or value <= 0:
            raise FloatingPointError("GARCH one-step variance became invalid")
        return float(value)

    def cumulative_variance_forecast(self, one_day_variance: float) -> float:
        """Return expected sum h_(t+1)+...+h_(t+5) under fixed parameters."""

        if not np.isfinite(one_day_variance) or one_day_variance <= 0:
            raise ValueError("one-day variance forecast must be finite and positive")
        parameters = self.parameters
        variance = float(one_day_variance)
        cumulative = variance
        for _ in range(2, self.horizon + 1):
            variance = parameters.omega + parameters.persistence * variance
            cumulative += variance
        return float(cumulative)

    def forecast_sequence(
        self,
        observed_returns: NDArray[np.float64],
        *,
        initial_variance: float | None = None,
    ) -> GARCHForecastPath:
        """Causally filter returns and forecast at each origin without lookahead."""

        returns = np.asarray(observed_returns, dtype=float).reshape(-1)
        if not len(returns) or not np.isfinite(returns).all():
            raise ValueError("forecast returns must be a non-empty finite vector")
        if initial_variance is None:
            if self.fit_result is None:
                raise RuntimeError("GARCH model is not fitted")
            current_variance = self.fit_result.next_conditional_variance
        else:
            current_variance = float(initial_variance)
        if not np.isfinite(current_variance) or current_variance <= 0:
            raise ValueError("forecast initial variance must be finite and positive")
        filtered = np.empty(len(returns), dtype=float)
        one_day = np.empty(len(returns), dtype=float)
        cumulative = np.empty(len(returns), dtype=float)
        for index, observed_return in enumerate(returns):
            filtered[index] = current_variance
            next_variance = self.one_step_variance(current_variance, float(observed_return))
            one_day[index] = next_variance
            cumulative[index] = self.cumulative_variance_forecast(next_variance)
            current_variance = next_variance
        target_unit = (self.annualization / self.horizon) * cumulative
        return GARCHForecastPath(filtered, one_day, cumulative, target_unit)
