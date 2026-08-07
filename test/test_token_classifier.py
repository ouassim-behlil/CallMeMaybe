import unittest
from TokenClassifier import TokenClassifier # adjust import path if needed


class TestTokenClassifier(unittest.TestCase):

    def setUp(self):
        self.vocab = {
            0: "{",
            1: "}",
            2: "[",
            3: "]",
            4: ":",
            5: ",",
            6: "\"",
            7: "a",
            8: "Z",
            9: "5",
            10: "-",
            11: ".",
            12: " ",
            13: "\n",
        }
        self.cls = TokenClassifier(id_to_token=self.vocab)

    # ---------- token_str ----------
    def test_token_str(self):
        self.assertEqual(self.cls.token_str(0), "{")
        self.assertEqual(self.cls.token_str(7), "a")
        self.assertEqual(self.cls.token_str(999), "")

    # ---------- structural tokens ----------
    def test_structural_tokens(self):
        expected = {0, 1, 2, 3, 4, 5, 6}
        self.assertEqual(self.cls.structural_tokens(), expected)

    # ---------- structural checks ----------
    def test_braces_and_brackets(self):
        self.assertTrue(self.cls.is_left_brace(0))
        self.assertFalse(self.cls.is_left_brace(1))

        self.assertTrue(self.cls.is_right_brace(1))
        self.assertFalse(self.cls.is_right_brace(0))

        self.assertTrue(self.cls.is_left_bracket(2))
        self.assertFalse(self.cls.is_left_bracket(3))

        self.assertTrue(self.cls.is_right_bracket(3))
        self.assertFalse(self.cls.is_right_bracket(2))

    def test_colon_comma_quote(self):
        self.assertTrue(self.cls.is_colon(4))
        self.assertFalse(self.cls.is_colon(5))

        self.assertTrue(self.cls.is_comma(5))
        self.assertFalse(self.cls.is_comma(4))

        self.assertTrue(self.cls.is_quote(6))
        self.assertFalse(self.cls.is_quote(7))

    # ---------- number tokens ----------
    def test_number_tokens(self):
        self.assertTrue(self.cls.is_digit(9))
        self.assertFalse(self.cls.is_digit(7))

        self.assertTrue(self.cls.is_minus(10))
        self.assertFalse(self.cls.is_minus(9))

        self.assertTrue(self.cls.is_dot(11))
        self.assertFalse(self.cls.is_dot(10))

        self.assertTrue(self.cls.is_number_char(9))
        self.assertTrue(self.cls.is_number_char(10))
        self.assertTrue(self.cls.is_number_char(11))
        self.assertFalse(self.cls.is_number_char(7))

        self.assertEqual(self.cls.digit_tokens(), {9})

    # ---------- whitespace ----------
    def test_whitespace(self):
        self.assertTrue(self.cls.is_white_space(12))
        self.assertTrue(self.cls.is_white_space(13))
        self.assertFalse(self.cls.is_white_space(7))

    # ---------- string content ----------
    def test_string_char(self):
        self.assertTrue(self.cls.is_string_char(7))
        self.assertTrue(self.cls.is_string_char(8))
        self.assertFalse(self.cls.is_string_char(0))
        self.assertFalse(self.cls.is_string_char(6))

    def test_string_tokens(self):
        expected = {7, 8, 9, 10, 11, 12, 13}
        self.assertEqual(self.cls.string_tokens(), expected)


if __name__ == '__main__':
    unittest.main()

