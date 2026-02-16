"""
Coplanar LEO-to-GEO orbital transfer environment with continuous thrust.

This module implements a Gymnasium environment for training RL agents
to perform orbital maneuvers using RK4-integrated two-body dynamics.
All units are km, km/s, kg, and seconds.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numba import njit
from pydantic import BaseModel, Field


# ── Numba-accelerated dynamics for rollout speed ──────────────────────


@njit(cache=True)
def _dynamics(state, mu, Tx, Ty, Isp, g0):
    """Two-body equations of motion with continuous thrust and mass depletion.

    Parameters
    ----------
    state : ndarray
        [x, y, vx, vy, m] in km, km/s, kg.
    mu : float
        Gravitational parameter (km^3/s^2).
    Tx, Ty : float
        Thrust components in km*kg/s^2.
    Isp : float
        Specific impulse in seconds.
    g0 : float
        Standard gravity in km/s^2.

    Returns
    -------
    ndarray
        Time derivatives of the state vector.
    """
    x, y, vx, vy, m = state[0], state[1], state[2], state[3], state[4]
    r = np.sqrt(x * x + y * y)
    r3 = r * r * r
    T_mag = np.sqrt(Tx * Tx + Ty * Ty)
    return np.array([
        vx,
        vy,
        -mu * x / r3 + Tx / m,
        -mu * y / r3 + Ty / m,
        -T_mag / (Isp * g0) if T_mag > 1e-12 else 0.0,
    ])


@njit(cache=True)
def _rk4_step(state, dt, mu, Tx, Ty, Isp, g0):
    """Single RK4 integration step.

    Parameters
    ----------
    state : ndarray
        Current state vector [x, y, vx, vy, m].
    dt : float
        Time step in seconds.
    mu, Tx, Ty, Isp, g0 : float
        Physical parameters (see ``_dynamics``).

    Returns
    -------
    ndarray
        State vector after one RK4 step.
    """
    k1 = _dynamics(state, mu, Tx, Ty, Isp, g0)
    k2 = _dynamics(state + 0.5 * dt * k1, mu, Tx, Ty, Isp, g0)
    k3 = _dynamics(state + 0.5 * dt * k2, mu, Tx, Ty, Isp, g0)
    k4 = _dynamics(state + dt * k3, mu, Tx, Ty, Isp, g0)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@njit(cache=True)
def _state_to_elements(x, y, vx, vy, mu):
    """Convert 2-D Cartesian state to Keplerian elements.

    Parameters
    ----------
    x, y : float
        Position components (km).
    vx, vy : float
        Velocity components (km/s).
    mu : float
        Gravitational parameter (km^3/s^2).

    Returns
    -------
    a : float
        Semi-major axis (km).
    e : float
        Eccentricity.
    r : float
        Current radius (km).
    v : float
        Current speed (km/s).
    """
    r = np.sqrt(x * x + y * y)
    v = np.sqrt(vx * vx + vy * vy)
    eps = v * v / 2.0 - mu / r
    a = -mu / (2.0 * eps) if abs(eps) > 1e-10 else 1e10
    h = x * vy - y * vx
    e_sq = 1.0 + 2.0 * eps * h * h / (mu * mu)
    e = np.sqrt(max(0.0, e_sq))
    return a, e, r, v


class EnvConfig(BaseModel):
    """Validated configuration for :class:`OrbitalTransferEnv`.

    All fields have sensible defaults; pass overrides as keyword
    arguments or as a dict.  Pydantic handles type coercion and
    validation automatically.
    """

    # Mission
    alt_leo: float = Field(400.0, description="LEO altitude above surface (km)")
    # Spacecraft
    mass_kg: float = Field(2000.0, gt=0, description="Wet mass (kg)")
    thrust_N: float = Field(0.5, ge=0, description="Max thrust (Newtons)")
    Isp_s: float = Field(3000.0, gt=0, description="Specific impulse (s)")
    # Simulation timing
    dt_inner: float = Field(60.0, gt=0, description="RK4 step size (s)")
    n_substeps: int = Field(5, ge=1, description="RK4 steps per RL action")
    max_steps: int = Field(10000, ge=1, description="Max episode steps")
    # Reward weights
    w_a: float = Field(10.0, description="Semi-major axis reward weight")
    w_e: float = Field(5.0, description="Eccentricity reward weight")
    w_fuel: float = Field(0.1, description="Fuel penalty weight")
    w_time: float = Field(0.001, description="Time penalty weight")

    model_config = {"extra": "forbid"}


class OrbitalTransferEnv(gym.Env):
    """Coplanar LEO-to-GEO transfer with continuous low-thrust propulsion.

    State
        ``[x, y, vx, vy, m]``  (km, km/s, kg)
    Observation
        ``[x/r_geo, y/r_geo, vx/v_leo, vy/v_leo, m/m0, a/a_tgt, e, t/t_max]``
    Action
        ``[thrust_frac in [0,1], angle_norm in [-1,1]]``

    Parameters
    ----------
    config : dict or EnvConfig, optional
        Override default mission and spacecraft parameters.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config=None):
        super().__init__()
        if isinstance(config, EnvConfig):
            cfg = config
        elif isinstance(config, dict):
            cfg = EnvConfig(**config)
        else:
            cfg = EnvConfig()

        # ── Physical constants ─────────────────────────────────────────
        self.mu = 3.986004418e5       # km^3/s^2
        self.R_earth = 6378.137       # km
        self.g0 = 9.80665e-3          # km/s^2

        # ── Mission parameters ─────────────────────────────────────────
        self.r_leo = self.R_earth + cfg.alt_leo
        self.r_geo = 42164.0
        self.a_target = self.r_geo    # circular GEO
        self.e_target = 0.0

        # ── Spacecraft ─────────────────────────────────────────────────
        self.m0 = cfg.mass_kg
        self.T_max = cfg.thrust_N
        self.Isp = cfg.Isp_s

        # ── Simulation timing ──────────────────────────────────────────
        self.dt_inner = cfg.dt_inner
        self.n_substeps = cfg.n_substeps
        self.dt_rl = self.dt_inner * self.n_substeps
        self.max_steps = cfg.max_steps

        # ── Safety bounds ──────────────────────────────────────────────
        self.r_min = self.R_earth + 200    # atmospheric reentry
        self.r_max = 50000.0               # escape prevention
        self.m_min = self.m0 * 0.10        # 10% mass floor
        self.v_esc_factor = 1.1            # 110% local escape velocity

        # ── Hohmann baseline (precomputed) ─────────────────────────────
        a_t = (self.r_leo + self.r_geo) / 2.0
        v_leo = np.sqrt(self.mu / self.r_leo)
        v_geo = np.sqrt(self.mu / self.r_geo)
        v_tp = np.sqrt(self.mu * (2.0 / self.r_leo - 1.0 / a_t))
        v_ta = np.sqrt(self.mu * (2.0 / self.r_geo - 1.0 / a_t))
        self.dv_hohmann = abs(v_tp - v_leo) + abs(v_geo - v_ta)
        self.v_leo_circ = v_leo

        # ── Gymnasium spaces ───────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        # ── Reward weights ─────────────────────────────────────────────
        self.w_a = cfg.w_a
        self.w_e = cfg.w_e
        self.w_fuel = cfg.w_fuel
        self.w_time = cfg.w_time

    # ── helpers ────────────────────────────────────────────────────────

    def _get_obs(self):
        a, e, r, v = _state_to_elements(
            self.state[0], self.state[1],
            self.state[2], self.state[3], self.mu,
        )
        return np.array([
            self.state[0] / self.r_geo,       # x normalised
            self.state[1] / self.r_geo,       # y normalised
            self.state[2] / self.v_leo_circ,  # vx normalised
            self.state[3] / self.v_leo_circ,  # vy normalised
            self.state[4] / self.m0,          # mass ratio
            a / self.a_target,                # a normalised
            e,                                # eccentricity
            self.step_count / self.max_steps, # time fraction
        ], dtype=np.float32)

    # ── Gymnasium API ──────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        v0 = np.sqrt(self.mu / self.r_leo)
        theta = self.np_random.uniform(0, 2 * np.pi)
        self.state = np.array([
            self.r_leo * np.cos(theta),
            self.r_leo * np.sin(theta),
            -v0 * np.sin(theta),
            v0 * np.cos(theta),
            self.m0,
        ])
        self.step_count = 0
        self.total_dv = 0.0
        a0, e0, _, _ = _state_to_elements(
            self.state[0], self.state[1],
            self.state[2], self.state[3], self.mu,
        )
        self.prev_da = abs(a0 - self.a_target) / self.a_target
        self.prev_de = abs(e0 - self.e_target)
        self.best_da = self.prev_da  # anti-gaming: high-water mark
        return self._get_obs(), {}

    def step(self, action):
        thrust_frac = float(np.clip(action[0], 0.0, 1.0))
        angle = float(action[1]) * np.pi  # [-1,1] -> [-pi,pi]

        # Convert Newtons to km*kg/s^2 (1 N = 1e-3 km*kg/s^2)
        T = thrust_frac * self.T_max * 1e-3
        Tx = T * np.cos(angle)
        Ty = T * np.sin(angle)

        # ── Integrate n_substeps of RK4 ───────────────────────────────
        for _ in range(self.n_substeps):
            self.state = _rk4_step(
                self.state, self.dt_inner,
                self.mu, Tx, Ty, self.Isp, self.g0,
            )

        self.step_count += 1

        # ── Track cumulative delta-V ──────────────────────────────────
        accel = thrust_frac * self.T_max * 1e-3 / self.state[4]
        self.total_dv += accel * self.dt_rl

        # ── Compute orbital elements ──────────────────────────────────
        a, e, r, v = _state_to_elements(
            self.state[0], self.state[1],
            self.state[2], self.state[3], self.mu,
        )
        da = abs(a - self.a_target) / self.a_target
        de = abs(e - self.e_target)

        # ── Dense reward (potential-based shaping) ────────────────────
        reward = self.w_a * (self.prev_da - da)
        reward += self.w_e * (self.prev_de - de)
        reward -= self.w_fuel * thrust_frac
        reward -= self.w_time

        # ── Anti-gaming: only reward genuine new progress ─────────────
        if da < self.best_da:
            reward += 0.5 * (self.best_da - da)
            self.best_da = da

        self.prev_da, self.prev_de = da, de

        # ── Terminal conditions ────────────────────────────────────────
        terminated, truncated = False, False
        info = {
            "a": a,
            "e": e,
            "r": r,
            "dv_total": self.total_dv,
            "mass_ratio": self.state[4] / self.m0,
            "dv_hohmann": self.dv_hohmann,
        }

        # Success: within 1% of target a and e < 0.01
        if da < 0.01 and de < 0.01:
            efficiency = max(0, 1.0 - self.total_dv / self.dv_hohmann)
            reward += 100.0 + 50.0 * efficiency
            terminated = True
            info["success"] = True

        # Crash: below atmosphere or above escape
        elif r < self.r_min or r > self.r_max:
            reward -= 100.0
            terminated = True
            info["success"] = False

        # Fuel exhausted
        elif self.state[4] < self.m_min:
            reward -= 50.0
            truncated = True
            info["success"] = False

        # Excessive fuel: >2x Hohmann budget
        elif self.total_dv > 2.0 * self.dv_hohmann:
            reward -= 10.0

        # Time limit
        if self.step_count >= self.max_steps:
            truncated = True

        reward = float(np.clip(reward, -110.0, 160.0))
        return self._get_obs(), reward, terminated, truncated, info
