import pytest

# Replace `your_module` with the actual module containing these classes.
from src.grammar import LiteralState, LiteralChoiceState, NumberState


class TestLiteralState:
    def test_is_complete_when_remaining_is_empty(self):
        state = LiteralState("")

        assert state.is_complete is True

    def test_is_not_complete_when_characters_remain(self):
        state = LiteralState("abc")

        assert state.is_complete is False

    def test_step_consumes_matching_character(self):
        state = LiteralState("abc")

        next_state = state.step("a")

        assert isinstance(next_state, LiteralState)
        assert next_state.remaining == "bc"
        assert next_state.is_complete is False

    def test_step_consumes_last_character(self):
        state = LiteralState("a")

        next_state = state.step("a")

        assert isinstance(next_state, LiteralState)
        assert next_state.remaining == ""
        assert next_state.is_complete is True

    def test_step_returns_none_for_wrong_character(self):
        state = LiteralState("abc")

        assert state.step("x") is None

    def test_step_on_empty_state_returns_none(self):
        state = LiteralState("")

        assert state.step("a") is None

    def test_step_does_not_mutate_original_state(self):
        state = LiteralState("abc")

        next_state = state.step("a")

        assert state.remaining == "abc"
        assert isinstance(next_state, LiteralState)
        assert next_state.remaining == "bc"


class TestLiteralChoiceState:
    def test_single_literal_is_not_complete_initially(self):
        state = LiteralChoiceState.from_options(["true"])

        assert state.is_complete is False

    def test_single_literal_becomes_complete_after_all_characters(self):
        state = LiteralChoiceState.from_options(["true"])

        for char in "tru":
            state = state.step(char)
            assert state is not None
            assert state.is_complete is False

        state = state.step("e")

        assert state is not None
        assert state.is_complete is True

    def test_false_literal_can_be_completed(self):
        state = LiteralChoiceState.from_options(["false"])

        for char in "false":
            state = state.step(char)

            assert state is not None

        assert state.is_complete is True

    def test_step_returns_none_for_invalid_first_character(self):
        state = LiteralChoiceState.from_options(["true", "false"])

        assert state.step("x") is None

    def test_step_returns_none_for_invalid_character(self):
        state = LiteralChoiceState.from_options(["true"])

        state = state.step("t")
        assert state is not None

        assert state.step("x") is None

    def test_only_matching_literal_can_be_completed(self):
        state = LiteralChoiceState.from_options(["true", "false"])

        for char in "true":
            state = state.step(char)
            assert state is not None

        assert state.is_complete is True

    def test_choice_state_is_not_complete_after_partial_match(self):
        state = LiteralChoiceState.from_options(["true", "false"])

        state = state.step("t")

        assert state is not None
        assert state.is_complete is False

    def test_multiple_options_can_be_selected(self):
        for literal in ("true", "false"):
            state = LiteralChoiceState.from_options(["true", "false"])

            for char in literal:
                state = state.step(char)
                assert state is not None

            assert state.is_complete is True

    def test_prefix_ambiguity_raises_value_error(self):
        with pytest.raises(ValueError, match="Ambiguous literal choices"):
            LiteralChoiceState.from_options(["a", "ab"])

    def test_prefix_ambiguity_is_detected_regardless_of_order(self):
        with pytest.raises(ValueError, match="Ambiguous literal choices"):
            LiteralChoiceState.from_options(["foobar", "foo"])

    def test_non_prefix_options_are_accepted(self):
        state = LiteralChoiceState.from_options(
            ["true", "false", "null"]
        )

        assert state.is_complete is False

    def test_duplicate_options_are_accepted(self):
        state = LiteralChoiceState.from_options(["true", "true"])

        for char in "true":
            state = state.step(char)
            assert state is not None

        assert state.is_complete is True

    def test_empty_options_are_not_complete(self):
        state = LiteralChoiceState.from_options([])

        assert state.is_complete is False

    def test_empty_string_option_is_complete(self):
        state = LiteralChoiceState.from_options([""])

        assert state.is_complete is True

    def test_step_does_not_mutate_previous_state(self):
        state = LiteralChoiceState.from_options(["true", "false"])

        next_state = state.step("t")

        assert next_state is not None
        assert state.is_complete is False
        assert next_state.is_complete is False

    def test_case_sensitive_matching(self):
        state = LiteralChoiceState.from_options(["true"])

        assert state.step("T") is None

    def test_extra_character_after_complete_literal_is_invalid(self):
        state = LiteralChoiceState.from_options(["true"])

        for char in "true":
            state = state.step(char)
            assert state is not None

        assert state.is_complete is True
        assert state.step("x") is None




class TestNumberState:
    @pytest.mark.parametrize("value", [
        "0",
        "1",
        "5",
        "9",
        "10",
        "123",
        "999",
    ])
    def test_valid_integer_is_complete(self, value: str):
        state = NumberState()

        for char in value:
            state = state.step(char)
            assert state is not None

        assert state.is_complete is True

    @pytest.mark.parametrize("value", [
        "0.1",
        "1.0",
        "1.5",
        "10.25",
        "123.456",
        "0.999",
    ])
    def test_valid_fraction_is_complete(self, value: str):
        state = NumberState()

        for char in value:
            state = state.step(char)
            assert state is not None

        assert state.is_complete is True

    def test_zero_is_complete(self):
        state = NumberState()

        state = state.step("0")

        assert state is not None
        assert state.is_complete is True

    def test_nonzero_integer_remains_complete_after_more_digits(self):
        state = NumberState()

        for char in "12":
            state = state.step(char)
            assert state is not None

        assert state.is_complete is True

        state = state.step("3")

        assert state is not None
        assert state.is_complete is True

    def test_number_is_not_complete_after_minus(self):
        state = NumberState()

        state = state.step("-")

        assert state is not None
        assert state.is_complete is False

    def test_number_is_not_complete_after_decimal_point(self):
        state = NumberState()

        for char in "1.":
            state = state.step(char)
            assert state is not None

        assert state.is_complete is False

    def test_fraction_becomes_complete_after_first_digit(self):
        state = NumberState()

        for char in "1.2":
            state = state.step(char)
            assert state is not None

        assert state.is_complete is True

    def test_negative_integer_is_valid(self):
        state = NumberState()

        for char in "-123":
            state = state.step(char)
            assert state is not None

        assert state.is_complete is True

    def test_negative_zero_is_valid(self):
        state = NumberState()

        for char in "-0":
            state = state.step(char)
            assert state is not None

        assert state.is_complete is True

    def test_negative_fraction_is_valid(self):
        state = NumberState()

        for char in "-12.34":
            state = state.step(char)
            assert state is not None

        assert state.is_complete is True

    def test_leading_zero_is_rejected(self):
        state = NumberState()

        state = state.step("0")
        assert state is not None
        assert state.is_complete is True

        assert state.step("1") is None

    @pytest.mark.parametrize("value", [
        "01",
        "00",
        "09",
        "-01",
        "-00",
    ])
    def test_leading_zero_numbers_are_rejected(self, value: str):
        state = NumberState()

        for char in value[:-1]:
            state = state.step(char)
            assert state is not None

        assert state.step(value[-1]) is None

    @pytest.mark.parametrize("value", [
        "",
        "-",
        "1.",
        "-1.",
        "0.",
        "-0.",
    ])
    def test_incomplete_numbers_are_not_complete(self, value: str):
        state = NumberState()

        for char in value:
            state = state.step(char)
            assert state is not None

        assert state.is_complete is False

    @pytest.mark.parametrize("value", [
        ".1",
        "-.1",
        "1..2",
        "1.2.",
        "1.2.3",
    ])
    def test_invalid_number_syntax_is_rejected(self, value: str):
        state = NumberState()

        for char in value:
            next_state = state.step(char)

            if next_state is None:
                break

            state = next_state
        else:
            pytest.fail(f"{value!r} was incorrectly accepted")

    @pytest.mark.parametrize("char", [
        "a",
        "x",
        " ",
        "+",
        "e",
        "E",
    ])
    def test_invalid_start_character_is_rejected(self, char: str):
        state = NumberState()

        assert state.step(char) is None

    def test_plus_sign_is_not_allowed(self):
        state = NumberState()

        assert state.step("+") is None

    def test_exponents_are_not_allowed(self):
        state = NumberState()

        for char in "12":
            state = state.step(char)
            assert state is not None

        assert state.step("e") is None

    def test_exponent_marker_uppercase_is_not_allowed(self):
        state = NumberState()

        for char in "12":
            state = state.step(char)
            assert state is not None

        assert state.step("E") is None

    def test_decimal_point_requires_digit(self):
        state = NumberState()

        for char in "12.":
            state = state.step(char)
            assert state is not None

        assert state.is_complete is False
        assert state.step(".") is None

    def test_fraction_accepts_multiple_digits(self):
        state = NumberState()

        for char in "12.345678":
            state = state.step(char)
            assert state is not None

        assert state.is_complete is True

    def test_invalid_character_after_fraction_is_rejected(self):
        state = NumberState()

        for char in "12.34":
            state = state.step(char)
            assert state is not None

        assert state.is_complete is True
        assert state.step("x") is None

    def test_state_can_be_used_incrementally(self):
        state = NumberState()

        state = state.step("-")
        assert state is not None
        assert state.is_complete is False

        state = state.step("1")
        assert state is not None
        assert state.is_complete is True

        state = state.step("2")
        assert state is not None
        assert state.is_complete is True

        state = state.step(".")
        assert state is not None
        assert state.is_complete is False

        state = state.step("5")
        assert state is not None
        assert state.is_complete is True
