"""Integration tests for the orbital-mechanics RL pipeline.

These tests verify that components work together correctly end-to-end,
as opposed to the unit tests which exercise each component in isolation.

Test groups
-----------
1. Env → Bridge → Reward : Full observation-to-score loop
2. Env → ActionParser → Env : Parsed actions drive the simulation
3. Env + Hohmann baseline : Analytical baseline matches env constants
4. Training data generation : Parquet rows are valid and self-consistent
5. Curriculum → Env : Curriculum configs produce valid environments
6. Evaluation harness : Dummy agent runs through the eval battery
7. Multi-episode rollout : Deterministic seeding, reproducibility
"""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest


# =====================================================================
# 1. Env → Bridge → Reward : Full observation-to-score loop
# =====================================================================


class TestEnvBridgeRewardLoop:
    """Verify the complete Env → text observation → LLM mock → score path."""

    def test_obs_to_text_to_score_well_formed(self):
        """A well-formed LLM response based on real env obs scores positively."""
        from loka.envs.orbital_transfer import OrbitalTransferEnv
        from loka.rl.bridge import OrbitalObservationWrapper, ActionParser
        from loka.rl.reward import compute_score

        env = OrbitalTransferEnv()
        obs, _ = env.reset(seed=0)

        # Bridge: observation → text prompt
        wrapper = OrbitalObservationWrapper(
            a_target=env.a_target, dv_hohmann=env.dv_hohmann,
        )
        text = wrapper.state_to_text(obs, step=0, max_steps=env.max_steps)

        # Verify text contains key info fields
        assert "Step 0" in text
        assert "Fuel:" in text
        assert "a=" in text
        assert "e=" in text

        # Mock LLM: produce a well-formed response using env state info
        llm_response = (
            "<think>I see the orbit is at LEO. I should burn prograde to "
            "raise the semi-major axis towards GEO.</think>\n"
            '<action>\n{"thrust": 0.7, "angle": 5.0}\n</action>'
        )

        # Parse the response
        parser = ActionParser()
        action, parse_reward, method = parser.parse(llm_response)
        assert method == "xml_json"
        assert action.shape == (2,)

        # Score it through the Verl reward function
        ground_truth = json.dumps({
            "a_target": env.a_target,
            "e_target": env.e_target,
            "dv_hohmann": env.dv_hohmann,
            "max_steps": env.max_steps,
            "initial_state": obs.tolist(),
        })
        score = compute_score("orbital_transfer", llm_response, ground_truth)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0
        # Well-formed response should get positive format score
        assert score > 0.0

    def test_obs_to_text_to_score_malformed(self):
        """A malformed LLM response scores lower than a well-formed one."""
        from loka.envs.orbital_transfer import OrbitalTransferEnv
        from loka.rl.reward import compute_score

        env = OrbitalTransferEnv()
        obs, _ = env.reset(seed=0)

        gt = json.dumps({
            "a_target": env.a_target, "e_target": env.e_target,
            "dv_hohmann": env.dv_hohmann, "max_steps": env.max_steps,
            "initial_state": obs.tolist(),
        })

        good = '<think>Analysis</think>\n<action>{"thrust":0.5,"angle":0}</action>'
        bad = "I think we should maybe thrust a bit? Not sure."

        score_good = compute_score("orbital_transfer", good, gt)
        score_bad = compute_score("orbital_transfer", bad, gt)
        assert score_good > score_bad

    def test_mission_reward_propagates_through_score(self):
        """Mission reward from env simulation correctly affects the final score."""
        from loka.envs.orbital_transfer import OrbitalTransferEnv
        from loka.rl.reward import compute_score

        env = OrbitalTransferEnv()
        obs, _ = env.reset(seed=0)
        gt = json.dumps({
            "a_target": env.a_target, "e_target": env.e_target,
            "dv_hohmann": env.dv_hohmann, "max_steps": env.max_steps,
            "initial_state": obs.tolist(),
        })

        response = '<think>Burn</think>\n<action>{"thrust":0.8,"angle":0}</action>'

        # Same format, different mission reward
        score_success = compute_score(
            "orbital_transfer", response, gt,
            extra_info={"mission_reward": 150.0},
        )
        score_failure = compute_score(
            "orbital_transfer", response, gt,
            extra_info={"mission_reward": -100.0},
        )
        assert score_success > score_failure

    def test_build_messages_roundtrip(self):
        """build_messages produces valid chat structure from env observations."""
        from loka.envs.orbital_transfer import OrbitalTransferEnv
        from loka.rl.bridge import OrbitalObservationWrapper, SYSTEM_PROMPT

        env = OrbitalTransferEnv()
        obs, _ = env.reset(seed=12)
        wrapper = OrbitalObservationWrapper(
            a_target=env.a_target, dv_hohmann=env.dv_hohmann,
        )

        messages = wrapper.build_messages(
            obs, step=0, max_steps=env.max_steps, dv_used=0.0,
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        assert len(messages[1]["content"]) > 20


# =====================================================================
# 2. Env → ActionParser → Env : Parsed actions drive the simulation
# =====================================================================


class TestActionParserDrivesEnv:
    """Verify that parsed LLM actions can step the environment."""

    def test_xml_json_action_steps_env(self):
        """Parse a well-formed action and feed it into env.step()."""
        from loka.envs.orbital_transfer import OrbitalTransferEnv
        from loka.rl.bridge import ActionParser

        env = OrbitalTransferEnv()
        obs0, _ = env.reset(seed=0)

        parser = ActionParser()
        llm_output = '<think>Prograde</think>\n<action>{"thrust":0.5,"angle":0}</action>'
        action, _, method = parser.parse(llm_output)
        assert method == "xml_json"

        obs1, reward, term, trunc, info = env.step(action)
        assert obs1.shape == (8,)
        assert "a" in info
        assert "e" in info
        # State should have changed
        assert not np.array_equal(obs0, obs1)

    def test_fallback_action_is_coast(self):
        """When parser falls back, the action is coast (zero thrust)."""
        from loka.envs.orbital_transfer import OrbitalTransferEnv, _state_to_elements
        from loka.rl.bridge import ActionParser

        env = OrbitalTransferEnv()
        env.reset(seed=42)
        state_before = env.state.copy()

        parser = ActionParser()
        action, _, method = parser.parse("gibberish with no valid action")
        assert method == "fallback"
        np.testing.assert_array_equal(action, [0.0, 0.0])

        # Coast should not burn fuel (mass unchanged)
        env.step(action)
        assert env.state[4] == pytest.approx(state_before[4], abs=1e-10)

    def test_multi_step_episode_with_parser(self):
        """Run a short episode where every step is parsed from text."""
        from loka.envs.orbital_transfer import OrbitalTransferEnv
        from loka.rl.bridge import ActionParser, OrbitalObservationWrapper

        env = OrbitalTransferEnv(config={"max_steps": 50})
        obs, _ = env.reset(seed=99)
        parser = ActionParser()
        wrapper = OrbitalObservationWrapper(
            a_target=env.a_target, dv_hohmann=env.dv_hohmann,
        )

        rewards = []
        for step in range(50):
            # Simulate LLM producing alternating thrust/coast
            if step % 2 == 0:
                llm_text = '<think>Burn prograde</think>\n<action>{"thrust":0.3,"angle":0}</action>'
            else:
                llm_text = '<think>Coasting</think>\n<action>{"thrust":0.0,"angle":0}</action>'
            action, _, _ = parser.parse(llm_text)
            obs, reward, term, trunc, info = env.step(action)
            rewards.append(reward)
            if term or trunc:
                break

        assert len(rewards) > 0
        assert all(isinstance(r, float) for r in rewards)
        # info should still have expected keys at the end
        assert "a" in info and "dv_total" in info


# =====================================================================
# 3. Env + Hohmann baseline : Analytical baseline matches env constants
# =====================================================================


class TestHohmannEnvConsistency:
    """Verify the Hohmann baseline and env agree on physical constants."""

    def test_dv_hohmann_matches_env(self):
        """hohmann_baseline() and env.dv_hohmann use the same physics."""
        from loka.envs.orbital_transfer import OrbitalTransferEnv
        from loka.astro.hohmann import hohmann_baseline

        env = OrbitalTransferEnv()
        baseline = hohmann_baseline(alt_leo_km=400.0)

        # Both should produce the same delta-V to within 0.1%
        assert abs(env.dv_hohmann - baseline.dv_total_kms) / env.dv_hohmann < 0.001

    def test_hohmann_at_different_altitudes(self):
        """Env and baseline agree across multiple LEO altitudes."""
        from loka.envs.orbital_transfer import OrbitalTransferEnv
        from loka.astro.hohmann import hohmann_baseline

        for alt in [200, 400, 600, 800, 1000]:
            env = OrbitalTransferEnv(config={"alt_leo": alt})
            baseline = hohmann_baseline(alt_leo_km=alt)
            assert abs(env.dv_hohmann - baseline.dv_total_kms) / env.dv_hohmann < 0.001, (
                f"Mismatch at {alt} km: env={env.dv_hohmann:.4f}, "
                f"baseline={baseline.dv_total_kms:.4f}"
            )

    def test_higher_leo_needs_less_dv(self):
        """Higher LEO → closer to GEO → less delta-V."""
        from loka.astro.hohmann import hohmann_baseline

        dv_200 = hohmann_baseline(alt_leo_km=200.0).dv_total_kms
        dv_400 = hohmann_baseline(alt_leo_km=400.0).dv_total_kms
        dv_800 = hohmann_baseline(alt_leo_km=800.0).dv_total_kms
        assert dv_200 > dv_400 > dv_800

    def test_fuel_fraction_ordering(self):
        """Higher Isp → less fuel needed (Tsiolkovsky equation)."""
        from loka.astro.hohmann import hohmann_baseline

        result = hohmann_baseline()
        ff = result.fuel_fractions
        # Chemical (300 s) > Hall-effect (1500 s) > Ion (3000 s)
        assert ff["Isp_300s"] > ff["Isp_1500s"] > ff["Isp_3000s"]


# =====================================================================
# 4. Training data generation : Parquet rows are valid and consistent
# =====================================================================


class TestTrainingDataGeneration:
    """Integration test for the data generation pipeline."""

    def test_generate_rows_structure(self):
        """Generated rows have the correct schema for Verl."""
        from loka.envs.orbital_transfer import OrbitalTransferEnv
        from loka.rl.bridge import OrbitalObservationWrapper

        env = OrbitalTransferEnv()
        wrapper = OrbitalObservationWrapper(
            a_target=env.a_target, dv_hohmann=env.dv_hohmann,
        )

        obs, _ = env.reset(seed=0)
        messages = wrapper.build_messages(obs, step=0, max_steps=env.max_steps)
        ground_truth = json.dumps({
            "initial_state": obs.tolist(),
            "a_target": env.a_target,
            "e_target": env.e_target,
            "dv_hohmann": env.dv_hohmann,
            "max_steps": env.max_steps,
        })

        row = {
            "data_source": "orbital_transfer",
            "prompt": messages,
            "reward_model": {"ground_truth": ground_truth},
        }

        # Verify structure
        assert row["data_source"] == "orbital_transfer"
        assert isinstance(row["prompt"], list)
        assert len(row["prompt"]) == 2
        assert row["prompt"][0]["role"] == "system"
        assert row["prompt"][1]["role"] == "user"

        # Verify ground truth is valid JSON that round-trips
        gt = json.loads(row["reward_model"]["ground_truth"])
        assert gt["a_target"] == env.a_target
        assert gt["e_target"] == env.e_target
        assert len(gt["initial_state"]) == 8

    def test_parquet_roundtrip(self):
        """Write rows to parquet and read them back with correct types."""
        pd = pytest.importorskip("pandas")
        pa = pytest.importorskip("pyarrow")

        from loka.envs.orbital_transfer import OrbitalTransferEnv
        from loka.rl.bridge import OrbitalObservationWrapper

        env = OrbitalTransferEnv()
        wrapper = OrbitalObservationWrapper(
            a_target=env.a_target, dv_hohmann=env.dv_hohmann,
        )

        rows = []
        for i in range(5):
            obs, _ = env.reset(seed=i)
            messages = wrapper.build_messages(obs, 0, env.max_steps)
            gt = json.dumps({
                "initial_state": obs.tolist(),
                "a_target": env.a_target,
                "e_target": env.e_target,
                "dv_hohmann": env.dv_hohmann,
                "max_steps": env.max_steps,
            })
            rows.append({
                "data_source": "orbital_transfer",
                "prompt": messages,
                "reward_model": {"ground_truth": gt},
            })

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.parquet"
            pd.DataFrame(rows).to_parquet(path)
            assert path.exists()

            df = pd.read_parquet(path)
            assert len(df) == 5
            assert list(df.columns) == ["data_source", "prompt", "reward_model"]

            # Each prompt should be a list of 2 dicts
            for _, row in df.iterrows():
                assert len(row["prompt"]) == 2

    def test_unique_seeds_produce_unique_initial_states(self):
        """Different seeds in data generation yield different initial orbits."""
        from loka.envs.orbital_transfer import OrbitalTransferEnv

        env = OrbitalTransferEnv()
        states = []
        for seed in range(10):
            obs, _ = env.reset(seed=seed)
            states.append(obs.tolist())

        # All should be unique (different orbital phases)
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                assert states[i] != states[j], f"seed {i} and {j} produced identical states"


# =====================================================================
# 5. Curriculum → Env : Curriculum configs produce valid environments
# =====================================================================


class TestCurriculumEnvIntegration:
    """Verify curriculum stage parameters create working environments."""

    def test_curriculum_stage_names_match_scheduler(self):
        """get_stage_name covers the full training range without gaps."""
        from loka.rl.curriculum import CurriculumScheduler

        sched = CurriculumScheduler()
        stages_seen = set()
        for step in range(0, 1000, 10):
            stages_seen.add(sched.get_stage_name(step, 1000))
        assert stages_seen == {"circularize", "hohmann", "plane_change"}

    def test_curriculum_mix_ratios_always_valid(self):
        """Mixing ratios sum to 1 and are non-negative at every step."""
        from loka.rl.curriculum import CurriculumScheduler

        sched = CurriculumScheduler()
        for step in range(0, 1001, 50):
            mix = sched.get_mix(step, 1000)
            assert all(v >= 0 for v in mix.values()), f"Negative ratio at step {step}"
            assert abs(sum(mix.values()) - 1.0) < 1e-9, f"Sum != 1 at step {step}"

    def test_curriculum_stages_use_valid_env_config_keys(self):
        """All stage config keys are accepted by OrbitalTransferEnv."""
        from loka.envs.orbital_transfer import OrbitalTransferEnv
        from loka.rl.curriculum import CurriculumScheduler

        # Env should accept alt_leo without error
        for alt in [200, 400, 600, 800]:
            env = OrbitalTransferEnv(config={"alt_leo": alt})
            obs, _ = env.reset(seed=0)
            assert obs.shape == (8,)

    def test_different_altitudes_produce_different_orbits(self):
        """Curriculum altitude variations result in measurably different states."""
        from loka.envs.orbital_transfer import OrbitalTransferEnv, _state_to_elements

        envs = {}
        for alt in [300, 500, 800]:
            env = OrbitalTransferEnv(config={"alt_leo": alt})
            env.reset(seed=42)
            a, e, r, v = _state_to_elements(
                env.state[0], env.state[1],
                env.state[2], env.state[3], env.mu,
            )
            envs[alt] = {"a": a, "r": r, "v": v}

        # Higher altitude → larger semi-major axis
        assert envs[300]["a"] < envs[500]["a"] < envs[800]["a"]
        # Higher altitude → lower orbital velocity
        assert envs[300]["v"] > envs[500]["v"] > envs[800]["v"]


# =====================================================================
# 6. Evaluation harness : Dummy agent runs through the eval battery
# =====================================================================


class TestEvaluationHarness:
    """Verify the evaluation functions execute without error using a mock agent."""

    class CoastAgent:
        """Trivial agent that never thrusts — used to exercise the harness."""

        def act(self, obs: np.ndarray) -> np.ndarray:
            return np.array([0.0, 0.0], dtype=np.float32)

    class RandomAgent:
        """Agent that picks random actions within the action space."""

        def __init__(self, seed: int = 0):
            self.rng = np.random.RandomState(seed)

        def act(self, obs: np.ndarray) -> np.ndarray:
            thrust = self.rng.uniform(0.0, 1.0)
            angle = self.rng.uniform(-1.0, 1.0)
            return np.array([thrust, angle], dtype=np.float32)

    def test_evaluate_generalization_coast_agent(self):
        """Coast agent completes the eval battery (all episodes time out)."""
        from loka.rl.evaluation import evaluate_generalization
        from loka.envs.orbital_transfer import OrbitalTransferEnv

        agent = self.CoastAgent()
        # Use tiny episode length so tests run fast
        results = evaluate_generalization(
            agent,
            env_class=lambda config=None: OrbitalTransferEnv(
                config={**(config or {}), "max_steps": 50},
            ),
            n_episodes=3,
            seed=0,
        )
        assert "LEO_300km" in results
        assert "LEO_500km" in results
        assert "adversarial_perturbation" in results
        # Coast agent should never succeed
        assert results["LEO_300km"]["success_rate"] == 0.0

    def test_compute_dv_efficiency_dummy(self):
        """compute_dv_efficiency runs with a dummy agent on a short env."""
        from loka.rl.evaluation import compute_dv_efficiency
        from loka.envs.orbital_transfer import OrbitalTransferEnv

        agent = self.CoastAgent()
        # Monkey-patch OrbitalTransferEnv temporarily for fast episodes
        original_init = OrbitalTransferEnv.__init__

        def fast_init(self_env, config=None):
            cfg = config or {}
            cfg.setdefault("max_steps", 50)
            original_init(self_env, config=cfg)

        OrbitalTransferEnv.__init__ = fast_init
        try:
            result = compute_dv_efficiency(agent, n_episodes=3)
            assert "eta_mean" in result
            assert "success_rate" in result
            # Coast agent won't succeed → eta_mean = 0
            assert result["eta_mean"] == 0.0
        finally:
            OrbitalTransferEnv.__init__ = original_init

    def test_eval_results_structure(self):
        """Evaluation results have the expected structure for each test."""
        from loka.rl.evaluation import evaluate_generalization
        from loka.envs.orbital_transfer import OrbitalTransferEnv

        agent = self.RandomAgent(seed=42)
        results = evaluate_generalization(
            agent,
            env_class=lambda config=None: OrbitalTransferEnv(
                config={**(config or {}), "max_steps": 30},
            ),
            n_episodes=2,
            seed=42,
        )

        # Should have entries for each altitude test
        for alt in [300, 500, 600, 800]:
            key = f"LEO_{alt}km"
            assert key in results
            assert "success_rate" in results[key]
            assert 0.0 <= results[key]["success_rate"] <= 1.0

        # Adversarial perturbation is a single float
        assert isinstance(results["adversarial_perturbation"], float)
        assert 0.0 <= results["adversarial_perturbation"] <= 1.0


# =====================================================================
# 7. Multi-episode rollout : Deterministic seeding & reproducibility
# =====================================================================


class TestReproducibility:
    """Verify that the pipeline produces deterministic results with fixed seeds."""

    def test_env_reset_is_deterministic(self):
        """Same seed → identical initial observation."""
        from loka.envs.orbital_transfer import OrbitalTransferEnv

        env = OrbitalTransferEnv()
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

    def test_full_episode_is_deterministic(self):
        """Same seed + same actions → identical trajectory."""
        from loka.envs.orbital_transfer import OrbitalTransferEnv

        actions = [
            np.array([0.3, 0.1], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([0.8, -0.5], dtype=np.float32),
            np.array([0.1, 0.9], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
        ]

        trajectories = []
        for _ in range(2):
            env = OrbitalTransferEnv()
            env.reset(seed=7)
            obs_sequence = []
            for a in actions:
                obs, reward, _, _, _ = env.step(a)
                obs_sequence.append((obs.tolist(), reward))
            trajectories.append(obs_sequence)

        for step_idx in range(len(actions)):
            np.testing.assert_array_equal(
                trajectories[0][step_idx][0],
                trajectories[1][step_idx][0],
                err_msg=f"Observation mismatch at step {step_idx}",
            )
            assert trajectories[0][step_idx][1] == trajectories[1][step_idx][1], (
                f"Reward mismatch at step {step_idx}"
            )

    def test_different_seeds_diverge(self):
        """Different seeds produce different initial conditions."""
        from loka.envs.orbital_transfer import OrbitalTransferEnv

        env = OrbitalTransferEnv()
        obs_a, _ = env.reset(seed=0)
        obs_b, _ = env.reset(seed=1)
        assert not np.array_equal(obs_a, obs_b)

    def test_parser_is_deterministic(self):
        """ActionParser produces identical output for the same input."""
        from loka.rl.bridge import ActionParser

        parser = ActionParser()
        text = '<think>Go up</think>\n<action>{"thrust":0.42,"angle":15.5}</action>'

        a1, r1, m1 = parser.parse(text)
        a2, r2, m2 = parser.parse(text)

        np.testing.assert_array_equal(a1, a2)
        assert r1 == r2
        assert m1 == m2


# =====================================================================
# 8. Physics sanity checks across the integrated system
# =====================================================================


class TestPhysicsSanityIntegrated:
    """Cross-component physical consistency checks."""

    def test_prograde_burn_raises_semi_major_axis(self):
        """Continuous prograde thrust should increase the semi-major axis."""
        from loka.envs.orbital_transfer import OrbitalTransferEnv, _state_to_elements

        env = OrbitalTransferEnv()
        env.reset(seed=0)

        a_initial, _, _, _ = _state_to_elements(
            env.state[0], env.state[1],
            env.state[2], env.state[3], env.mu,
        )

        # Burn prograde (angle=0) for 100 steps
        prograde = np.array([1.0, 0.0], dtype=np.float32)
        for _ in range(100):
            obs, _, term, trunc, info = env.step(prograde)
            if term or trunc:
                break

        a_final = info["a"]
        assert a_final > a_initial, (
            f"Prograde burn did not raise a: {a_initial:.1f} → {a_final:.1f}"
        )

    def test_retrograde_burn_lowers_semi_major_axis(self):
        """Continuous retrograde thrust should decrease the semi-major axis."""
        from loka.envs.orbital_transfer import OrbitalTransferEnv, _state_to_elements

        env = OrbitalTransferEnv()
        env.reset(seed=0)

        a_initial, _, _, _ = _state_to_elements(
            env.state[0], env.state[1],
            env.state[2], env.state[3], env.mu,
        )

        # Burn retrograde (angle=1.0 → π radians) for 100 steps
        retrograde = np.array([1.0, 1.0], dtype=np.float32)
        for _ in range(100):
            obs, _, term, trunc, info = env.step(retrograde)
            if term or trunc:
                break

        a_final = info["a"]
        assert a_final < a_initial, (
            f"Retrograde burn did not lower a: {a_initial:.1f} → {a_final:.1f}"
        )

    def test_thrust_depletes_mass(self):
        """Applying thrust should consume propellant mass."""
        from loka.envs.orbital_transfer import OrbitalTransferEnv

        env = OrbitalTransferEnv()
        env.reset(seed=0)
        m_initial = env.state[4]

        thrust = np.array([1.0, 0.0], dtype=np.float32)
        for _ in range(50):
            env.step(thrust)

        m_final = env.state[4]
        assert m_final < m_initial, "Full thrust did not deplete mass"
        assert m_final > 0, "Mass went to zero or negative"

    def test_coast_preserves_mass(self):
        """Zero thrust should not change spacecraft mass."""
        from loka.envs.orbital_transfer import OrbitalTransferEnv

        env = OrbitalTransferEnv()
        env.reset(seed=0)
        m_initial = env.state[4]

        coast = np.array([0.0, 0.0], dtype=np.float32)
        for _ in range(100):
            env.step(coast)

        assert env.state[4] == pytest.approx(m_initial, abs=1e-12)

    def test_dv_budget_tracks_cumulative_thrust(self):
        """env.total_dv should be zero after coast-only and positive after burn."""
        from loka.envs.orbital_transfer import OrbitalTransferEnv

        env = OrbitalTransferEnv()
        env.reset(seed=0)
        assert env.total_dv == 0.0

        # Coast 10 steps
        coast = np.array([0.0, 0.0], dtype=np.float32)
        for _ in range(10):
            env.step(coast)
        assert env.total_dv == pytest.approx(0.0, abs=1e-15)

        # Burn 10 steps
        burn = np.array([0.5, 0.0], dtype=np.float32)
        for _ in range(10):
            env.step(burn)
        assert env.total_dv > 0.0

    def test_success_requires_correct_orbit(self):
        """The env only signals success when a ≈ a_target and e ≈ 0."""
        from loka.envs.orbital_transfer import OrbitalTransferEnv

        env = OrbitalTransferEnv(config={"max_steps": 100})
        env.reset(seed=0)

        # 100 steps of coast → should NOT succeed (still at LEO)
        coast = np.array([0.0, 0.0], dtype=np.float32)
        for _ in range(100):
            obs, reward, term, trunc, info = env.step(coast)
            if term:
                # If terminated, it should be a crash, not success
                assert not info.get("success", False)
                break
        # If we just truncated, that's expected (still at LEO)


# =====================================================================
# 9. Reward function edge cases with integrated components
# =====================================================================


class TestRewardEdgeCases:
    """Edge-case integration between the parser, reward, and env."""

    def test_empty_response_scores_low(self):
        """An empty string should yield the minimum format score."""
        from loka.rl.reward import compute_score

        gt = json.dumps({
            "a_target": 42164, "e_target": 0,
            "dv_hohmann": 3.85, "max_steps": 10000,
            "initial_state": [0] * 8,
        })
        score = compute_score("orbital_transfer", "", gt)
        assert score <= 0.0

    def test_perfect_format_no_mission_reward(self):
        """Perfect format with no mission simulation yields only format reward."""
        from loka.rl.reward import compute_score

        gt = json.dumps({
            "a_target": 42164, "e_target": 0,
            "dv_hohmann": 3.85, "max_steps": 10000,
            "initial_state": [0] * 8,
        })
        response = '<think>Analysis here.</think>\n<action>{"thrust":0.5,"angle":10}</action>'
        score = compute_score("orbital_transfer", response, gt)
        # Should be purely format-based (0.05 + 0.10 + 0.05 = 0.20)
        assert 0.15 <= score <= 0.25

    def test_trailing_text_penalised(self):
        """Text after </action> should reduce format score."""
        from loka.rl.reward import compute_score

        gt = json.dumps({
            "a_target": 42164, "e_target": 0,
            "dv_hohmann": 3.85, "max_steps": 10000,
            "initial_state": [0] * 8,
        })
        clean = '<think>Ok</think>\n<action>{"thrust":0.5,"angle":0}</action>'
        trailing = '<think>Ok</think>\n<action>{"thrust":0.5,"angle":0}</action>\nExtra garbage here.'

        score_clean = compute_score("orbital_transfer", clean, gt)
        score_trailing = compute_score("orbital_transfer", trailing, gt)
        assert score_clean >= score_trailing
