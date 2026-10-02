# worker/test_eval_metrics.py
import unittest

from evaluation.metrics import macro_f1, match_events


class MatchEventsTests(unittest.TestCase):
    def test_perfect_match(self):
        match = match_events([100, 500], [100, 500], 50)
        self.assertEqual((match.true_positives, match.false_positives, match.false_negatives), (2, 0, 0))
        self.assertEqual(match.f1, 1.0)
        self.assertEqual(match.median_abs_error_ms, 0.0)

    def test_tolerance_is_inclusive_and_misses_count(self):
        match = match_events([130, 900], [100, 500], 30)
        self.assertEqual((match.true_positives, match.false_positives, match.false_negatives), (1, 1, 1))
        self.assertEqual(match.pairs, ((0, 0),))

    def test_one_prediction_cannot_match_two_truths_and_nearest_wins(self):
        match = match_events([110, 118], [100], 50)
        self.assertEqual((match.true_positives, match.false_positives, match.false_negatives), (1, 1, 0))
        self.assertEqual(match.pairs, ((0, 0),))
        self.assertEqual(match.median_abs_error_ms, 10.0)

    def test_median_error_uses_matched_pairs_only(self):
        match = match_events([100, 520, 900], [110, 500], 50)
        self.assertEqual(match.median_abs_error_ms, 15.0)

    def test_nothing_to_score_is_zero_not_an_error(self):
        match = match_events([], [], 50)
        self.assertEqual((match.precision, match.recall, match.f1), (0.0, 0.0, 0.0))
        self.assertIsNone(match.median_abs_error_ms)


class MacroF1Tests(unittest.TestCase):
    def test_perfect_prediction(self):
        result = macro_f1(["clear", "drop"], ["clear", "drop"])
        self.assertEqual(result["macroF1"], 1.0)

    def test_labels_never_seen_or_predicted_are_not_scored(self):
        result = macro_f1(["clear", "clear"], ["clear", "clear"], ("clear", "drop", "smash"))
        self.assertEqual(result["macroF1"], 1.0)
        self.assertEqual(result["perLabel"]["drop"]["support"], 0)

    def test_wrong_and_missing_labels_lower_the_score(self):
        y_true = ["clear", "smash", "drop"]
        y_pred = ["clear", "drive", "missed"]
        result = macro_f1(y_true, y_pred, ("clear", "smash", "drop", "drive"))
        # clear 1.0, smash 0.0, drop 0.0, drive 0.0 (predicted once, never true)
        self.assertAlmostEqual(result["macroF1"], 0.25)
        self.assertEqual(result["perLabel"]["drive"]["predicted"], 1)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            macro_f1(["clear"], [])


if __name__ == "__main__":
    unittest.main()
