import unittest

from agent_meeting.roles import RoleDefinition, extract_output_contract_verdict


def _reviewer_role() -> RoleDefinition:
    return RoleDefinition(
        name="skeptic-reviewer",
        frontmatter={
            "output_contract": "End every review with a one-line verdict: ACCEPT | REVISE | REJECT."
        },
    )


class ExtractOutputContractVerdictTests(unittest.TestCase):
    def test_explicit_verdict_beats_later_enum_mentions(self) -> None:
        output = "**VERDICT: REVISE** -- these unresolved gaps prevent ACCEPT."

        self.assertEqual(extract_output_contract_verdict(_reviewer_role(), output), "REVISE")

    def test_explicit_accept_verdict(self) -> None:
        output = "All blocking gaps have been closed.\nVERDICT: ACCEPT"

        self.assertEqual(extract_output_contract_verdict(_reviewer_role(), output), "ACCEPT")

    def test_ambiguous_final_line_without_explicit_verdict_returns_none(self) -> None:
        output = "This should not ACCEPT yet; it still needs REVISE."

        self.assertIsNone(extract_output_contract_verdict(_reviewer_role(), output))

    def test_unambiguous_final_line_fallback(self) -> None:
        output = "The proposal misses a core requirement.\nREJECT"

        self.assertEqual(extract_output_contract_verdict(_reviewer_role(), output), "REJECT")

    def test_markdown_heading_with_verdict_on_next_line(self) -> None:
        # Regression test: mtg_6b0c464bc9's real round-9 Skeptic turn used this
        # exact shape ("## Verdict" heading, verdict token on the following
        # line) and the old parser missed it entirely for round 11 (returned
        # None) because _EXPLICIT_VERDICT_RE only matched a token on the SAME
        # line as "VERDICT".
        output = (
            "## Verdict\n\n"
            "**ACCEPT** -- the architecture specification is complete after 11 rounds."
        )

        self.assertEqual(extract_output_contract_verdict(_reviewer_role(), output), "ACCEPT")

    def test_prose_use_of_enum_word_does_not_hijack_real_verdict(self) -> None:
        # Regression test: mtg_6b0c464bc9's real round-9 Skeptic turn wrote
        # "## Verdict\n\n**REVISE** -- three material corrections ..." but the
        # line just above it, in ordinary prose, said "...now fully measured --
        # accept it and design around X". The old fallback scanned the last 5
        # lines for ANY line containing exactly one enum word anywhere in it,
        # so it matched "accept" in that prose sentence and returned ACCEPT
        # instead of the real REVISE verdict two lines above.
        output = (
            "## Verdict\n\n"
            "**REVISE** -- three material corrections from Round 9:\n"
            "1. bright_fraction gate at 0.85, not 0.50\n"
            "2. The detection ceiling is now fully measured -- accept it and design around X\n"
            "3. Truro Path C produces exactly the gallery the task's preference asks for\n\n"
            "Phase 1 calibration is the ONLY remaining blocker."
        )

        self.assertEqual(extract_output_contract_verdict(_reviewer_role(), output), "REVISE")


if __name__ == "__main__":
    unittest.main()
