"""Unit tests for analysis and presentation tools in the tools module."""

import pytest

from sequentialized_barnard_tests.tools.plotting import compact_letter_display

##### Compact Letter Display Test #####


@pytest.mark.parametrize(
    ("significant_model_list", "model_name_list"),
    [
        # fmt: off
        ([], ["A", "B", "C", "D", "E"]),
        ([("A", "B")], ["A", "B", "C"]),
        ([("A", "B"), ("B", "C")], ["A", "B", "C", "D"]),
        ([("A", "B"), ("A", "F"), ("B", "C"), ("C", "F"), ("C", "D")], ["A", "B", "C", "D", "E", "F"]),
        ([("A", "B"), ("C", "D"), ("F", "H")], ["A", "B", "C", "D", "E", "F", "G", "H"]),
        ([("A", "B"), ("B", "C"), ("C", "D"), ("A", "E")], ["A", "B", "C", "D", "E"]),
        ([("A", "B"), ("C", "D"), ("D", "E"), ("C", "F"), ("B", "F")], ["A", "B", "C", "D", "E", "F"]),
        # fmt: on
    ],
)
def test_compact_letter_display(significant_model_list, model_name_list):
    cld_list = compact_letter_display(significant_model_list, model_name_list)
    for i, model_0 in enumerate(model_name_list):
        for j, model_1 in enumerate(model_name_list[i + 1 :]):
            letters_0 = set(list(cld_list[i]))
            letters_1 = set(list(cld_list[i + 1 + j]))
            if (model_0, model_1) in significant_model_list or (
                model_1,
                model_0,
            ) in significant_model_list:
                # Should not share any letter.
                assert letters_0.isdisjoint(letters_1)
            else:
                # Should share at least one letter.
                assert not letters_0.isdisjoint(letters_1)
