"""Tests for the orbital-mechanics RL pipeline.

Covers:
- OrbitalTransferEnv (reset, step, energy conservation, terminal conditions)
- ActionParser (all four fallback strategies)
- compute_score (format reward)
- hohmann_baseline (analytical delta-V)
"""

import json
import numpy as np
import pytest


# =====================================================================
# OrbitalTransferEnv
# =====================================================================

class TestOrbitalTransferEnv:
    """Tests for the Gymnasium orbital-transfer environment."""

    def _make_env(self, **kwargs):
        from loka.envs.orbital_transfer import OrbitalTransferEnv
        return OrbitalTransferEnv(config=kwargs)

    def test_reset_obs_shape(self):
        env = self._make_env()
        obs, info = env.reset(seed=0)
        assert obs.shape == (8,)
        assert obs.dtype == np.float32

    def test_reset_circular_orbit_elements(self):
        """After reset, a ≈ r_leo and e ≈ 0 (circular LEO)."""
        from loka.envs.orbital_transfer import _state_to_elements
        env = self._make_env()
        obs, _ = env.reset(seed=42)
        a, e, r, v = _state_to_elements(
            env.state[0], env.state[1],
            env.state[2], env.state[3], env.mu,
        )
        assert abs(a - env.r_leo) / env.r_leo < 1e-6, f"a={a}, r_leo={env.r_leo}"
        assert e < 1e-6, f"e={e}"

    def test_step_returns_correct_shapes(self):
        env = self._make_env()
        env.reset(seed=0)
        action = np.array([0.5, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (8,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_coast_energy_conservation(self):
        """With zero thrust, specific energy should stay constant (RK4)."""
        from loka.envs.orbital_transfer import _state_to_elements
        env = self._make_env()
        env.reset(seed=7)
        mu = env.mu

        # Compute initial energy
        a0, e0, r0, v0 = _state_to_elements(
            env.state[0], env.state[1],
            env.state[2], env.state[3], mu,
        )
        eps0 = v0**2 / 2.0 - mu / r0

        # Coast 200 steps with zero thrust
        zero_action = np.array([0.0, 0.0], dtype=np.float32)
        for _ in range(200):
            env.step(zero_action)

        a1, e1, r1, v1 = _state_to_elements(
            env.state[0], env.state[1],
            env.state[2], env.state[3], mu,
        )
        eps1 = v1**2 / 2.0 - mu / r1

        # Energy drift should be small (RK4 on ~12 000 s of coast)
        assert abs(eps1 - eps0) / abs(eps0) < 1e-5, (
            f"Energy drift: {abs(eps1 - eps0)/abs(eps0):.2e}"
        )

    def test_crash_terminates(self):
        """Pushing the spacecraft inward should eventually crash."""
        env = self._make_env(max_steps=50000)
        env.reset(seed=0)
        # Full retrograde thrust
        action = np.array([1.0, -1.0], dtype=np.float32)
        terminated = False
        for _ in range(50000):
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        # Should crash (r < r_min) or at least terminate
        assert terminated or truncated

    def test_hohmann_dv_value(self):
        """The precomputed Hohmann delta-V should be close to 3.85 km/s."""
        env = self._make_env()
        assert abs(env.dv_hohmann - 3.854) < 0.02, f"dv_hohmann={env.dv_hohmann}"


# =====================================================================
# ActionParser
# =====================================================================

class TestActionParser:
    """Tests for the cascading fallback action parser."""

    def _parser(self):
        from loka.rl.bridge import ActionParser
        return ActionParser()

    def test_xml_json_parse(self):
        parser = self._parser()
        text = (
            '<think>Orbit is low.</think>\n'
            '<action>\n{"thrust": 0.8, "angle": 45.0}\n</action>'
        )
        action, reward, method = parser.parse(text)
        assert method == "xml_json"
        assert reward == 1.0
        assert abs(action[0] - 0.8) < 1e-5
        assert abs(action[1] - 45.0 / 180.0) < 1e-5

    def test_bare_json_parse(self):
        parser = self._parser()
        text = 'Some reasoning... {"thrust": 0.3, "angle": -90.0} done'
        action, reward, method = parser.parse(text)
        assert method == "bare_json"
        assert reward == 0.5
        assert abs(action[0] - 0.3) < 1e-5
        assert abs(action[1] - (-90.0 / 180.0)) < 1e-5

    def test_regex_parse(self):
        parser = self._parser()
        text = 'thrust is "thrust": 0.5 and "angle": 10.0 roughly'
        action, reward, method = parser.parse(text)
        assert method == "regex"
        assert reward == 0.25
        assert abs(action[0] - 0.5) < 1e-5

    def test_fallback(self):
        parser = self._parser()
        text = "I have no idea what to do."
        action, reward, method = parser.parse(text)
        assert method == "fallback"
        assert reward == -0.5
        np.testing.assert_array_equal(action, [0.0, 0.0])

    def test_clamps_values(self):
        parser = self._parser()
        text = '<action>{"thrust": 5.0, "angle": 999.0}</action>'
        action, reward, method = parser.parse(text)
        assert action[0] == 1.0    # clamped
        assert action[1] == 1.0    # 180/180 clamped

    def test_non_dict_json_falls_through(self):
        """Regression test: LLM may emit <action>false</action> or other
        non-dict JSON, which should gracefully fall to a lower strategy."""
        parser = self._parser()
        # bool
        text = "<action>false</action>"
        action, reward, method = parser.parse(text)
        assert method in ("regex", "fallback")
        # null
        text2 = "<action>null</action>"
        _, _, method2 = parser.parse(text2)
        assert method2 in ("regex", "fallback")
        # list
        text3 = "<action>[0.5, 10.0]</action>"
        _, _, method3 = parser.parse(text3)
        assert method3 in ("regex", "fallback")
        # bare number
        text4 = "<action>42</action>"
        _, _, method4 = parser.parse(text4)
        assert method4 in ("regex", "fallback")
        # string
        text5 = '<action>"coast"</action>'
        _, _, method5 = parser.parse(text5)
        assert method5 in ("regex", "fallback")


# =====================================================================
# compute_score
# =====================================================================

class TestComputeScore:
    """Tests for the Verl reward function."""

    def test_format_only(self):
        from loka.rl.reward import compute_score
        gt = json.dumps({"a_target": 42164, "e_target": 0, "dv_hohmann": 3.94,
                         "max_steps": 10000, "initial_state": [0]*8})
        good = '<think>Hello</think>\n<action>{"thrust":0.5,"angle":0}</action>'
        bad = "random garbage"

        score_good = compute_score("test", good, gt)
        score_bad = compute_score("test", bad, gt)
        assert score_good > score_bad

    def test_with_mission_reward(self):
        from loka.rl.reward import compute_score
        gt = json.dumps({"a_target": 42164, "e_target": 0, "dv_hohmann": 3.94,
                         "max_steps": 10000, "initial_state": [0]*8})
        text = '<think>Go</think>\n<action>{"thrust":0.5,"angle":0}</action>'

        score_hi = compute_score("test", text, gt,
                                 extra_info={"mission_reward": 150.0})
        score_lo = compute_score("test", text, gt,
                                 extra_info={"mission_reward": -100.0})
        assert score_hi > score_lo

    def test_score_bounds(self):
        from loka.rl.reward import compute_score
        gt = json.dumps({"a_target": 42164, "e_target": 0, "dv_hohmann": 3.94,
                         "max_steps": 10000, "initial_state": [0]*8})
        text = '<think>X</think>\n<action>{"thrust":1,"angle":0}</action>'
        # Extreme mission reward
        score = compute_score("test", text, gt,
                              extra_info={"mission_reward": 9999.0})
        assert -1.0 <= score <= 1.0


# =====================================================================
# hohmann_baseline
# =====================================================================

class TestHohmannBaseline:
    """Tests for the analytical Hohmann baseline."""

    def test_dv_total(self):
        from loka.astro.hohmann import hohmann_baseline
        result = hohmann_baseline()
        # LEO 400 km -> GEO: ~3.854 km/s
        assert abs(result.dv_total_kms - 3.854) < 0.01

    def test_transfer_time_hours(self):
        from loka.astro.hohmann import hohmann_baseline
        result = hohmann_baseline()
        # ~5.2–5.3 hours for LEO→GEO Hohmann
        assert 5.0 < result.transfer_time_hours < 5.5

    def test_fuel_fractions_keys(self):
        from loka.astro.hohmann import hohmann_baseline
        result = hohmann_baseline()
        assert "Isp_300s" in result.fuel_fractions
        assert "Isp_3000s" in result.fuel_fractions
        # Electric propulsion should need much less fuel
        assert result.fuel_fractions["Isp_3000s"] < result.fuel_fractions["Isp_300s"]


# =====================================================================
# CurriculumScheduler
# =====================================================================

class TestCurriculumScheduler:
    """Tests for the 3-stage curriculum."""

    def test_stage_1_dominated_by_circularize(self):
        from loka.rl.curriculum import CurriculumScheduler
        sched = CurriculumScheduler()
        mix = sched.get_mix(global_step=10, total_steps=1000)
        assert mix.circularize > mix.hohmann
        assert mix.circularize > mix.plane_change

    def test_stage_2_dominated_by_hohmann(self):
        from loka.rl.curriculum import CurriculumScheduler
        sched = CurriculumScheduler()
        mix = sched.get_mix(global_step=500, total_steps=1000)
        assert mix.hohmann > mix.circularize
        assert mix.hohmann > mix.plane_change

    def test_stage_3_dominated_by_plane_change(self):
        from loka.rl.curriculum import CurriculumScheduler
        sched = CurriculumScheduler()
        mix = sched.get_mix(global_step=900, total_steps=1000)
        assert mix.plane_change > mix.circularize
        assert mix.plane_change > mix.hohmann

    def test_mix_sums_to_one(self):
        from loka.rl.curriculum import CurriculumScheduler
        sched = CurriculumScheduler()
        for step in [0, 100, 500, 700, 999]:
            mix = sched.get_mix(step, 1000)
            total = mix.circularize + mix.hohmann + mix.plane_change
            assert abs(total - 1.0) < 1e-9


# =====================================================================
# MetricsTracker
# =====================================================================

class TestMetricsTracker:
    """Tests for the domain-specific metrics tracking."""

    def test_sample_metrics_defaults(self):
        from loka.rl.metrics import SampleMetrics
        m = SampleMetrics()
        assert m.total_reward == 0.0
        assert m.parse_method == "fallback"
        assert m.success is None

    def test_sample_metrics_with_values(self):
        from loka.rl.metrics import SampleMetrics
        m = SampleMetrics(
            total_reward=0.75,
            format_reward=0.15,
            physics_reward=0.60,
            parse_method="xml_json",
            success=True,
            final_a_km=42164.0,
            final_e=0.001,
            dv_total_kms=3.9,
            dv_hohmann_kms=3.854,
            mass_ratio=0.6,
            steps_used=500,
            response_length=120,
            has_think_tag=True,
            has_action_tag=True,
        )
        assert m.total_reward == 0.75
        assert m.parse_method == "xml_json"
        assert m.success is True
        assert m.final_a_km == 42164.0

    def test_tracker_record_and_summary(self):
        from loka.rl.metrics import MetricsTracker, SampleMetrics

        tracker = MetricsTracker(flush_every=1000, enabled=True)
        for i in range(10):
            tracker.record(SampleMetrics(
                total_reward=float(i) / 10,
                format_reward=0.1,
                physics_reward=float(i) / 10 - 0.1,
                parse_method="xml_json" if i % 2 == 0 else "bare_json",
                response_length=100 + i,
                has_think_tag=True,
                has_action_tag=True,
            ))

        summary = tracker.get_summary()
        assert "loka/reward/total_mean" in summary
        assert "loka/reward/format_mean" in summary
        assert "loka/reward/physics_mean" in summary
        assert "loka/parse/xml_json_rate" in summary
        assert "loka/parse/bare_json_rate" in summary
        assert "loka/format/think_rate" in summary
        assert "loka/format/action_rate" in summary
        assert "loka/response/length_mean" in summary

        # Check computed values
        assert abs(summary["loka/reward/format_mean"] - 0.1) < 1e-6
        assert abs(summary["loka/parse/xml_json_rate"] - 0.5) < 1e-6
        assert abs(summary["loka/parse/bare_json_rate"] - 0.5) < 1e-6
        assert summary["loka/format/think_rate"] == 1.0
        assert summary["loka/format/action_rate"] == 1.0

    def test_tracker_orbital_metrics(self):
        from loka.rl.metrics import MetricsTracker, SampleMetrics

        tracker = MetricsTracker(flush_every=1000, enabled=True)
        tracker.record(SampleMetrics(
            total_reward=0.8,
            format_reward=0.15,
            physics_reward=0.65,
            parse_method="xml_json",
            success=True,
            final_a_km=42164.0,
            final_e=0.001,
            dv_total_kms=4.0,
            dv_hohmann_kms=3.854,
            mass_ratio=0.55,
            steps_used=800,
            response_length=150,
            has_think_tag=True,
            has_action_tag=True,
        ))
        tracker.record(SampleMetrics(
            total_reward=0.1,
            format_reward=0.1,
            physics_reward=0.0,
            parse_method="fallback",
            success=False,
            final_a_km=7000.0,
            final_e=0.3,
            steps_used=200,
            response_length=30,
            has_think_tag=False,
            has_action_tag=False,
        ))

        summary = tracker.get_summary()
        assert summary["loka/orbital/success_rate"] == 0.5
        # Only 1 successful episode with both dv values
        assert abs(summary["loka/orbital/dv_efficiency_mean"] - 3.854 / 4.0) < 1e-4
        assert abs(summary["loka/orbital/final_a_mean_km"] - (42164.0 + 7000.0) / 2) < 1
        assert abs(summary["loka/orbital/steps_mean"] - 500.0) < 1e-6
        assert summary["loka/orbital/steps_max"] == 800

    def test_tracker_empty_summary(self):
        from loka.rl.metrics import MetricsTracker
        tracker = MetricsTracker(flush_every=1000, enabled=True)
        assert tracker.get_summary() == {}

    def test_tracker_disabled_noop(self):
        from loka.rl.metrics import MetricsTracker, SampleMetrics
        tracker = MetricsTracker(enabled=False)
        tracker.record(SampleMetrics(total_reward=0.5))
        assert tracker.get_summary() == {}

    def test_compute_score_records_metrics(self):
        """compute_score should record to the global tracker."""
        from loka.rl.reward import compute_score
        from loka.rl import metrics as metrics_mod

        # Replace the global tracker with a test one
        old_tracker = metrics_mod._tracker
        test_tracker = metrics_mod.MetricsTracker(flush_every=10000, enabled=True)
        metrics_mod._tracker = test_tracker

        try:
            gt = json.dumps({"a_target": 42164, "e_target": 0, "dv_hohmann": 3.94,
                             "max_steps": 10000, "initial_state": [0]*8})
            text = '<think>Go</think>\n<action>{"thrust":0.5,"angle":0}</action>'
            compute_score("test", text, gt, extra_info={
                "mission_reward": 100.0,
                "success": True,
                "a": 42100.0,
                "e": 0.01,
                "dv_total": 4.1,
                "mass_ratio": 0.58,
                "steps_used": 750,
            })

            summary = test_tracker.get_summary()
            assert summary["loka/reward/total_mean"] > 0
            assert summary["loka/reward/format_mean"] > 0
            assert summary["loka/reward/physics_mean"] > 0
            assert summary["loka/parse/xml_json_rate"] == 1.0
            assert summary["loka/orbital/success_rate"] == 1.0
            assert abs(summary["loka/orbital/final_a_mean_km"] - 42100.0) < 1
        finally:
            metrics_mod._tracker = old_tracker
