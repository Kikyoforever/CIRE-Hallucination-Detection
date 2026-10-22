"""No network, model files, datasets or cloud inference required."""

import math
import unittest

import numpy as np

from src.cire import (
    assert_matching_response_tokens,
    cire_scores,
    geometric_mean_token_probability,
    hallucination_risk,
    relative_frobenius_shift,
    response_hidden_block,
)


class TestCIREMetrics(unittest.TestCase):
    def setUp(self):
        self.a = np.array([[1., 0.], [0., 1.]])
        self.c = np.array([[2., 0.], [0., 1.]])
        self.log_probs = np.log(np.array([0.5, 0.8]))

    def test_relative_frobenius_eq5(self):
        self.assertAlmostEqual(relative_frobenius_shift(self.a, self.c), 1 / math.sqrt(2))

    def test_probability_geometric_mean_eq6(self):
        self.assertAlmostEqual(geometric_mean_token_probability(self.log_probs), math.sqrt(0.4))

    def test_full_score_eq7_and_direction(self):
        s = cire_scores(self.a, self.c, token_log_probs=self.log_probs)
        self.assertAlmostEqual(s.groundedness_score, 1 / math.sqrt(5))
        self.assertAlmostEqual(s.hallucination_risk_score, -1 / math.sqrt(5))
        self.assertGreater(s.normalized_output_probability, 0)

    def test_zero_shift_is_zero(self):
        s = cire_scores(self.a, self.a, token_log_probs=np.zeros(2))
        self.assertEqual(s.ifl, 0.0)
        self.assertEqual(s.groundedness_score, 0.0)

    def test_probability_of_one(self):
        self.assertEqual(geometric_mean_token_probability([0., 0.]), 1.)

    def test_long_log_probabilities_stay_stable(self):
        self.assertAlmostEqual(geometric_mean_token_probability([-600., -600.]), math.exp(-600.))

    def test_zero_denominator_fails(self):
        with self.assertRaises(ValueError):
            relative_frobenius_shift(np.zeros((2, 2)), self.c)

    def test_shape_mismatch_fails(self):
        with self.assertRaises(ValueError):
            relative_frobenius_shift(self.a, np.ones((3, 2)))
        with self.assertRaises(ValueError):
            cire_scores(self.a, self.c, token_log_probs=[-0.1])

    def test_invalid_probability_fails(self):
        for x in [[], [-0.1, 0.1], [float('nan')], [[-0.1, -0.2]]]:
            with self.subTest(x=x), self.assertRaises(ValueError):
                geometric_mean_token_probability(x)

    def test_nan_hidden_states_fails(self):
        with self.assertRaises(ValueError):
            relative_frobenius_shift([[1, float('nan')]], [[1, 0.]])

    def test_high_grounding_means_lower_risk(self):
        self.assertLess(hallucination_risk(2.0), hallucination_risk(0.5))

    def test_negative_grounding_rejected(self):
        with self.assertRaises(ValueError):
            hallucination_risk(-0.5)


class TestAlignment(unittest.TestCase):
    def test_equal_token_ids(self):
        assert_matching_response_tokens([42, 7, 0], [42, 7, 0])

    def test_mismatched_ids_fail(self):
        with self.assertRaises(ValueError):
            assert_matching_response_tokens([12, 24], [12, 25])

    def test_empty_ids_fail(self):
        with self.assertRaises(ValueError):
            assert_matching_response_tokens([], [])

    def test_slice_exact_answer_block(self):
        full = np.arange(30, dtype=float).reshape(10, 3)
        np.testing.assert_array_equal(
            response_hidden_block(full, answer_start=5, answer_length=3), full[5:8]
        )

    def test_invalid_span_fails(self):
        with self.assertRaises(ValueError):
            response_hidden_block(np.ones((4, 2)), answer_start=3, answer_length=2)


if __name__ == '__main__':
    unittest.main()
