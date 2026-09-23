from __future__ import annotations

import unittest

import reliability_checker


class ReliabilityScorerTests(unittest.TestCase):
    def test_bracketed_commas_remain_inside_one_skill(self) -> None:
        self.assertEqual(
            reliability_checker.split_bracketed_delimiters("a, [b, c], d"),
            ["a", "b, c", "d"],
        )

    def test_exact_deduplication_preserves_nested_skills(self) -> None:
        self.assertEqual(
            reliability_checker.unique_normalized_items(
                ["Python", " python ", "experience with Python"]
            ),
            ["Python", "experience with Python"],
        )

    def test_exact_first_matching_protects_later_exact_pair(self) -> None:
        counts = reliability_checker.match_counts(
            ["python skills", "python"],
            ["python", "python skills experience"],
        )
        self.assertEqual(counts.exact_tp, 1)
        self.assertEqual(counts.containment_extra_tp, 1)

    def test_matching_is_one_to_one(self) -> None:
        counts = reliability_checker.match_counts(
            ["python", "advanced python"],
            ["python programming"],
        )
        self.assertEqual(counts.exact_tp + counts.containment_extra_tp, 1)

    def test_revision_macro_resolves_to_current_value(self) -> None:
        self.assertEqual(
            reliability_checker.resolve_simple_oursrev(r"value=\OURSREV{0.562}{0.489}"),
            "value=0.489",
        )

    def test_workbooks_have_expected_valid_overlap_ids(self) -> None:
        expected_overlap_ids = {"software_developers": 32, "journalists": 35}
        for config in reliability_checker.workbook_configs():
            _, workbook_summary, workbook_audit = reliability_checker.score_workbook(
                config
            )
            self.assertEqual(
                workbook_audit.valid_overlap_ids, expected_overlap_ids[config.domain]
            )
            self.assertTrue(workbook_audit.exact_id_set_match)
            self.assertEqual(
                {row["records"] for row in workbook_summary},
                {expected_overlap_ids[config.domain]},
            )


if __name__ == "__main__":
    unittest.main()
