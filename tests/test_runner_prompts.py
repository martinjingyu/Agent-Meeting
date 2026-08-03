import unittest

from agent_meeting.runner import _build_planning_round_message, _question_with_planning_constraints


class PlanningRoundPromptTests(unittest.TestCase):
    def test_planning_question_adds_no_human_intervention_constraint(self) -> None:
        prompt = _question_with_planning_constraints("Select gallery images.")

        self.assertIn("fully automated", prompt)
        self.assertIn("manual labeling", prompt)
        self.assertIn("human-in-the-loop parameter tuning", prompt)

    def test_round_message_requires_evidence_backed_claims(self) -> None:
        prompt = _build_planning_round_message("Select gallery images.", 1, [])

        self.assertIn("small, cheap, reproducible tests", prompt)
        self.assertIn("State what you tested", prompt)
        self.assertIn("untested hypothesis", prompt)

    def test_round_message_requires_sample_size_policy(self) -> None:
        prompt = _build_planning_round_message("Select gallery images.", 1, [])

        self.assertIn("Test sample-size policy", prompt)
        self.assertIn("Cross-dataset generalization claim", prompt)
        self.assertIn("Threshold calibration claim", prompt)
        self.assertIn("validated -> hypothesis", prompt)


if __name__ == "__main__":
    unittest.main()
