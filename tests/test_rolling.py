import unittest
from unittest.mock import patch

import numpy as np
import torch

import config
from dynamics.rolling import (
    _expert_plan_to_waypoint_numpy,
    _expert_plans_batch_numpy,
    expert_candidate_cost_numpy,
    expert_candidate_costs_batch_numpy,
    energy_aware_speed_limit_numpy,
    mission_time_limit_hours,
    make_condition,
    plan_expert_controls_numpy,
    required_energy_reserve_numpy,
    rollout_controls_numpy,
    rollout_controls_torch,
    target_from_condition_numpy,
)
from models import create_denoiser
from models.diffusion import TrajectoryDiffusion
from models.wind_encoder import SpatioTemporalWindEncoder
from rolling_planner import run_rolling_planner
from dynamics.wind_field import _limit_vector_magnitude


class RollingConfigurationTest(unittest.TestCase):
    def test_time_scales(self):
        self.assertEqual(config.PLAN_HOURS, 1)
        self.assertEqual(config.TRAJ_LEN, 8)
        self.assertEqual(config.REPLAN_MINUTES, 7.5)
        self.assertEqual(config.EXECUTE_STEPS, 1)
        self.assertEqual(config.WIND_NUM_FRAMES, 16)
        self.assertEqual(config.EXPERT_EVAL_STEPS, 16)
        self.assertGreater(config.WIND_NUM_FRAMES, config.TRAJ_LEN)

    def test_relative_goal_survives_condition_build(self):
        state = np.array([0.2, 0.3, 0.1, -0.1, 0.5, 0.8], dtype=np.float32)
        target = [150, 75]
        condition = make_condition(state, target, 0.75)
        recovered = target_from_condition_numpy(condition)
        expected = np.array([target[0] / config.WIDTH, target[1] / config.HEIGHT])
        np.testing.assert_allclose(recovered, expected, atol=1e-6)

    def test_route_mode_is_one_hot_and_does_not_change_physical_target(self):
        state = np.array([0.2, 0.3, 0.1, -0.1, 0.5, 0.8], dtype=np.float32)
        direct = make_condition(state, [150, 75], 0.75, route_mode=0)
        right = make_condition(state, [150, 75], 0.75, route_mode=2)
        self.assertEqual(direct.shape, (config.COND_DIM,))
        self.assertEqual(float(direct[config.COND_ROUTE_MODE_START:].sum()), 1.0)
        self.assertEqual(float(right[config.COND_ROUTE_MODE_START:].sum()), 1.0)
        self.assertEqual(float(right[config.COND_ROUTE_MODE_START + 2]), 1.0)
        np.testing.assert_allclose(
            target_from_condition_numpy(direct),
            target_from_condition_numpy(right),
        )


class RollingPhysicsTest(unittest.TestCase):
    def test_wind_vector_limit_applies_to_resultant_magnitude(self):
        u = np.array([[8.0]], dtype=np.float32)
        v = np.array([[8.0]], dtype=np.float32)
        limited_u, limited_v = _limit_vector_magnitude(u, v, 8.0)
        self.assertAlmostEqual(
            float(np.hypot(limited_u, limited_v)[0, 0]), 8.0, places=5
        )

    def test_absolute_setpoint_uses_bounded_low_level_increment(self):
        state = np.array([0.3, 0.4, 0.0, 0.0, 0.5, 0.8], dtype=np.float32)
        condition = make_condition(state, [160, 80], 1.0)
        controls = np.zeros((2, config.TRAJ_LEN), dtype=np.float32)
        controls[0] = 1.0
        wind = np.zeros((config.WIND_NUM_FRAMES, 12, 21), dtype=np.float32)
        states = rollout_controls_numpy(controls, condition, wind, wind)
        expected = np.minimum(
            1.0,
            np.arange(1, 6, dtype=np.float32) * config.ACTION_RANGE / config.MAX_SPEED,
        )
        np.testing.assert_allclose(states[2, :5], expected, atol=1e-6)
        self.assertTrue(np.all(states[2, 5:] <= 1.0))

    def test_energy_aware_speed_limit_preserves_reserve_at_night(self):
        night_limit = energy_aware_speed_limit_numpy(22.0, 0.3)
        day_limit = energy_aware_speed_limit_numpy(13.0, 0.8)
        self.assertLess(night_limit, day_limit)
        self.assertLessEqual(day_limit, config.MAX_SPEED * config.EXPERT_AIRSPEED_RESERVE)

    def test_numpy_and_torch_rollouts_match(self):
        rng = np.random.default_rng(3)
        state = np.array([0.3, 0.4, 0.05, -0.03, 0.25, 0.8], dtype=np.float32)
        condition = make_condition(state, [160, 80], 1.0)
        controls = rng.uniform(-0.2, 0.2, size=(2, config.TRAJ_LEN)).astype(np.float32)
        wind_u = rng.normal(2.0, 0.2, size=(config.TRAJ_LEN, 12, 21)).astype(np.float32)
        wind_v = rng.normal(-1.0, 0.2, size=(config.TRAJ_LEN, 12, 21)).astype(np.float32)
        numpy_states = rollout_controls_numpy(controls, condition, wind_u, wind_v)
        torch_states = rollout_controls_torch(
            torch.from_numpy(controls).unsqueeze(0),
            torch.from_numpy(condition).unsqueeze(0),
            torch.from_numpy(wind_u).unsqueeze(0),
            torch.from_numpy(wind_v).unsqueeze(0),
        )[0].numpy()
        np.testing.assert_allclose(torch_states, numpy_states, atol=2e-5, rtol=2e-5)

    def test_diagonal_velocity_increment_uses_vector_norm_limit(self):
        state = np.array([0.3, 0.4, 0.0, 0.0, 0.5, 0.8], dtype=np.float32)
        condition = make_condition(state, [160, 80], 1.0)
        controls = np.ones((2, 1), dtype=np.float32)
        wind = np.zeros((1, 12, 21), dtype=np.float32)
        next_state = rollout_controls_numpy(controls, condition, wind, wind)[:, 0]
        achieved_speed = np.linalg.norm(next_state[2:4]) * config.MAX_SPEED
        self.assertAlmostEqual(achieved_speed, config.ACTION_RANGE, places=5)

    def test_dynamic_reserve_is_higher_before_a_long_night(self):
        night = required_energy_reserve_numpy(19.0, 200.0)
        day = required_energy_reserve_numpy(10.0, 50.0)
        self.assertGreater(night, day)

    def test_dynamic_deadline_responds_to_distance_wind_and_energy(self):
        easy = mission_time_limit_hours(100.0, 0.9, 0.0)
        long_route = mission_time_limit_hours(300.0, 0.9, 0.0)
        adverse = mission_time_limit_hours(100.0, 0.55, 8.0)
        self.assertGreater(long_route, easy)
        self.assertGreater(adverse, easy)
        self.assertGreaterEqual(easy, config.MIN_MISSION_HOURS)
        self.assertLessEqual(adverse, config.MAX_MISSION_HOURS)

    def test_expert_generates_distinct_direct_and_lateral_candidates(self):
        state = np.array([0.2, 0.5, 0.0, 0.0, 0.4, 0.8], dtype=np.float32)
        condition = make_condition(state, [170, 60], 1.0)
        wind = np.zeros((config.WIND_NUM_FRAMES, 12, 21), dtype=np.float32)
        controls, states, details = plan_expert_controls_numpy(
            condition, wind, wind, stochastic=False
        )
        self.assertEqual(controls.shape, (config.ACTION_DIM, config.TRAJ_LEN))
        self.assertEqual(states.shape, (config.STATE_DIM, config.TRAJ_LEN))
        self.assertEqual(details["states"].shape[0], (
            len(config.EXPERT_ROUTE_OFFSETS_KM)
            * (len(config.EXPERT_SPEED_RATIOS) - 1)
            + 1
        ))
        self.assertEqual(details["states"].shape[-1], config.EXPERT_EVAL_STEPS)
        endpoints = details["states"][:, :2, -1].copy()
        endpoints[:, 0] *= config.WIDTH_KM
        endpoints[:, 1] *= config.HEIGHT_KM
        self.assertGreater(float(np.ptp(endpoints[:, 1])), 1.0)

    def test_vectorized_expert_matches_scalar_candidate(self):
        rng = np.random.default_rng(11)
        state = np.array([0.3, 0.45, 0.03, -0.02, 0.75, 0.7], dtype=np.float32)
        condition = make_condition(state, [160, 75], 0.8)
        wind_u = rng.normal(
            2.0, 0.5, size=(config.WIND_NUM_FRAMES, 12, 21)
        ).astype(np.float32)
        wind_v = rng.normal(
            -1.0, 0.5, size=(config.WIND_NUM_FRAMES, 12, 21)
        ).astype(np.float32)
        waypoint = np.array([0.55, 0.60], dtype=np.float32)
        scalar_controls, scalar_states = _expert_plan_to_waypoint_numpy(
            condition, wind_u, wind_v, waypoint, 0.82
        )
        batch_controls, batch_states = _expert_plans_batch_numpy(
            condition, wind_u, wind_v, waypoint[None],
            np.array([0.82], dtype=np.float32),
        )
        np.testing.assert_allclose(batch_controls[0], scalar_controls, atol=2e-6)
        np.testing.assert_allclose(batch_states[0], scalar_states, atol=2e-6)

        scalar_score = expert_candidate_cost_numpy(
            scalar_states, scalar_controls, condition, wind_u, wind_v
        )["total"]
        batch_score = expert_candidate_costs_batch_numpy(
            batch_states, batch_controls, condition, wind_u, wind_v
        )[0]
        self.assertAlmostEqual(float(batch_score), float(scalar_score), places=5)


class RollingModelShapeTest(unittest.TestCase):
    def test_training_loss_with_closed_loop_terms_is_finite(self):
        torch.manual_seed(13)
        batch = 2
        height = config.HEIGHT // config.RANGE_SCALE
        width = config.WIDTH // config.RANGE_SCALE
        diffusion = TrajectoryDiffusion(
            create_denoiser(), SpatioTemporalWindEncoder()
        )
        controls = torch.empty(
            batch, config.ACTION_DIM, config.TRAJ_LEN
        ).uniform_(-0.4, 0.4)
        cond = torch.zeros(batch, config.COND_DIM)
        cond[:, config.COND_X] = 0.3
        cond[:, config.COND_Y] = 0.4
        cond[:, config.COND_GOAL_DX] = 0.25
        cond[:, config.COND_GOAL_DY] = 0.1
        cond[:, config.COND_ENERGY] = 0.8
        cond[:, config.COND_TIME_COS] = 1.0
        cond[:, config.COND_REMAINING] = 1.0
        cond[:, config.COND_ROUTE_MODE_START] = 1.0
        wind_u = torch.zeros(batch, config.WIND_NUM_FRAMES, height, width)
        wind_v = torch.zeros_like(wind_u)
        target_states = rollout_controls_torch(controls, cond, wind_u, wind_v)
        valid = torch.ones(batch, config.TRAJ_LEN, dtype=torch.bool)
        previous = torch.zeros_like(controls)
        previous_mask = torch.zeros_like(valid)
        losses = diffusion.loss(
            controls, cond, wind_u, wind_v, target_states, valid,
            previous, previous_mask,
        )
        for name in ("total", "first_step", "expert_progress", "closing", "stagnation"):
            self.assertTrue(torch.isfinite(losses[name]), name)
        losses["total"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in diffusion.parameters()))

    def test_spatiotemporal_wind_and_dit_shapes(self):
        batch = 2
        height = config.HEIGHT // config.RANGE_SCALE
        width = config.WIDTH // config.RANGE_SCALE
        wind_u = torch.randn(batch, config.WIND_NUM_FRAMES, height, width)
        wind_v = torch.randn_like(wind_u)
        encoder = SpatioTemporalWindEncoder()
        wind_tokens = encoder(wind_u, wind_v)
        self.assertEqual(
            tuple(wind_tokens.shape),
            (batch, config.WIND_NUM_FRAMES, config.WIND_PATCHES_Y * config.WIND_PATCHES_X, config.WIND_ENC_OUT),
        )
        denoiser = create_denoiser()
        output = denoiser(
            torch.randn(batch, config.ACTION_DIM, config.TRAJ_LEN),
            torch.zeros(batch, dtype=torch.long),
            torch.zeros(batch, config.COND_DIM),
            wind_tokens,
            torch.zeros(batch, config.ACTION_DIM, config.TRAJ_LEN),
        )
        self.assertEqual(tuple(output.shape), (batch, config.ACTION_DIM, config.TRAJ_LEN))

    def test_missing_and_zero_previous_plan_are_equivalent(self):
        torch.manual_seed(7)
        batch = 2
        height = config.HEIGHT // config.RANGE_SCALE
        width = config.WIDTH // config.RANGE_SCALE
        encoder = SpatioTemporalWindEncoder().eval()
        denoiser = create_denoiser().eval()
        wind_u = torch.randn(batch, config.WIND_NUM_FRAMES, height, width)
        wind_tokens = encoder(wind_u, torch.randn_like(wind_u))
        x_t = torch.randn(batch, config.ACTION_DIM, config.TRAJ_LEN)
        diffusion_t = torch.full((batch,), 10, dtype=torch.long)
        cond = torch.randn(batch, config.COND_DIM)
        with torch.no_grad():
            missing = denoiser(x_t, diffusion_t, cond, wind_tokens, None)
            zero = denoiser(
                x_t, diffusion_t, cond, wind_tokens,
                torch.zeros(batch, config.ACTION_DIM, config.TRAJ_LEN),
            )
        torch.testing.assert_close(missing, zero)

    def test_profiled_multi_candidate_ddim_sampling(self):
        torch.manual_seed(17)
        batch = 1
        height = config.HEIGHT // config.RANGE_SCALE
        width = config.WIDTH // config.RANGE_SCALE
        diffusion = TrajectoryDiffusion(
            create_denoiser().eval(), SpatioTemporalWindEncoder().eval()
        ).eval()
        condition = torch.zeros(batch, config.COND_DIM)
        condition[:, config.COND_X] = 0.3
        condition[:, config.COND_Y] = 0.4
        condition[:, config.COND_GOAL_DX] = 0.3
        condition[:, config.COND_GOAL_DY] = 0.1
        condition[:, config.COND_ENERGY] = 0.8
        wind = torch.zeros(batch, config.WIND_NUM_FRAMES, height, width)
        result = diffusion.sample_multi(
            condition, wind, wind,
            torch.zeros(batch, config.ACTION_DIM, config.TRAJ_LEN),
            guidance_scale=2.0, num_samples=config.ROUTE_MODE_COUNT,
            sampler="ddim", ddim_steps=2, eta=0.2,
        )
        self.assertEqual(
            tuple(result["controls"].shape),
            (batch, config.ROUTE_MODE_COUNT, config.ACTION_DIM, config.TRAJ_LEN),
        )
        self.assertTrue(torch.isfinite(result["costs"]).all())
        self.assertEqual(
            set(result["route_mode"][0].tolist()),
            set(range(config.ROUTE_MODE_COUNT)),
        )


class PureDitPlannerTest(unittest.TestCase):
    def test_dit_inference_never_calls_rule_expert(self):
        class FakeDiffusion:
            def sample_multi(self, cond, wind_u, wind_v, previous_plan, **kwargs):
                batch = cond.shape[0]
                samples = kwargs["num_samples"]
                controls = torch.zeros(
                    batch, samples, config.ACTION_DIM, config.TRAJ_LEN,
                    device=cond.device,
                )
                controls[:, :, 0] = 0.5
                flat_controls = controls.reshape(
                    batch * samples, config.ACTION_DIM, config.TRAJ_LEN
                )
                flat_cond = cond.repeat_interleave(samples, dim=0)
                flat_wind_u = wind_u.repeat_interleave(samples, dim=0)
                flat_wind_v = wind_v.repeat_interleave(samples, dim=0)
                states = rollout_controls_torch(
                    flat_controls, flat_cond, flat_wind_u, flat_wind_v
                ).reshape(batch, samples, config.STATE_DIM, config.TRAJ_LEN)
                costs = torch.zeros(batch, samples, device=cond.device)
                return {
                    "controls": controls,
                    "states": states,
                    "costs": costs,
                    "best_controls": controls[:, 0],
                    "best_states": states[:, 0],
                }

        frame = torch.zeros(1, config.HEIGHT // config.RANGE_SCALE,
                            config.WIDTH // config.RANGE_SCALE)
        winds = [frame.clone() for _ in range(config.WIND_NUM_FRAMES + 2)]
        with patch("config.MAX_MISSION_STEPS", 1), patch(
            "rolling_planner.plan_expert_controls_numpy",
            side_effect=AssertionError("online expert must not be called"),
        ):
            result = run_rolling_planner(
                FakeDiffusion(), winds, winds, [80, 60], [130, 60],
                candidates=2, sampler="ddim", ddim_steps=2,
                device=torch.device("cpu"),
            )
        self.assertNotIn("fallback_count", result.metrics)
        self.assertEqual(result.metrics["plan_reset_count"], 0.0)


if __name__ == "__main__":
    unittest.main()
