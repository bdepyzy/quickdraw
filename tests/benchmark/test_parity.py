"""Check capture alignment and parity gates without loading a model or GPU."""

import unittest

import torch

from benchmark.offline.parity import logit_metrics, validate_capture


class ParityTests(unittest.TestCase):
    def test_extra_scheduler_row_is_excluded(self):
        logits = torch.tensor([[0., 8., 0.], [7., 0., 0.], [0., 0., 9.]])
        actual = validate_capture(logits, [1, 0], 3)
        torch.testing.assert_close(actual, logits[:2])

    def test_missing_or_shifted_rows_fail(self):
        logits = torch.tensor([[0., 8., 0.], [7., 0., 0.]])
        for ids in ([1, 0, 2], [0, 1]):
            with self.assertRaises(AssertionError):
                validate_capture(logits, ids, 3)

    def test_greedy_agreement_does_not_hide_distribution_error(self):
        metrics = logit_metrics(torch.tensor([0., .1]), torch.tensor([0., 8.]), 1,
                                max_rmse=.25, max_tv=.05)
        self.assertTrue(metrics["top1_match"])
        self.assertFalse(metrics["passed"])
        self.assertGreater(metrics["total_variation"], .4)

    def test_identical_logits_pass_with_zero_error(self):
        logits = torch.tensor([-2., 3., 1.])
        metrics = logit_metrics(logits, logits, 1, max_rmse=0, max_tv=0)
        self.assertTrue(metrics["passed"])
        self.assertEqual(metrics["rms_error"], 0)
        self.assertEqual(metrics["total_variation"], 0)

    def test_nonfinite_or_misaligned_reference_fails(self):
        logits = torch.tensor([-2., 3., 1.])
        for actual, token in ((torch.tensor([0., float("nan"), 1.]), 1), (logits, 0)):
            with self.assertRaises(AssertionError):
                logit_metrics(actual, logits, token, max_rmse=.25, max_tv=.05)


if __name__ == "__main__":
    unittest.main()
