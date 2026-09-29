import math
import unittest

from model import LABELS, HateSpeechService


class FakeSklearnPipeline:
    classes_ = ["normal", "hate", "offensive"]

    def predict_proba(self, _texts):
        return [[0.7, 0.2, 0.1]]


class HateSpeechServiceTests(unittest.TestCase):
    def test_placeholder_returns_normalized_probabilities(self) -> None:
        service = HateSpeechService()

        result = service.predict("Thanks for helping the community today.")

        self.assertEqual(result.label, "normal")
        self.assertEqual(set(result.probabilities), set(LABELS))
        self.assertTrue(math.isclose(sum(result.probabilities.values()), 1.0))

    def test_sklearn_class_order_is_mapped_by_label(self) -> None:
        service = HateSpeechService()
        service.backend = "sklearn"
        service._pipeline = FakeSklearnPipeline()

        result = service.predict("Example")

        self.assertEqual(result.label, "normal")
        self.assertAlmostEqual(result.probabilities["normal"], 0.7)
        self.assertAlmostEqual(result.probabilities["hate"], 0.2)
        self.assertAlmostEqual(result.probabilities["offensive"], 0.1)

    def test_unknown_model_labels_fail_loudly(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Model labels"):
            HateSpeechService._resolve_output_labels(["negative", "neutral", "positive"])

    def test_invalid_probability_vector_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "invalid class probabilities"):
            HateSpeechService._validated_probabilities(
                [0.2, float("nan"), 0.8],
                LABELS,
            )


if __name__ == "__main__":
    unittest.main()
