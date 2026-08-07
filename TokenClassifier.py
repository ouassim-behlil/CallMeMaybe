from pydantic import BaseModel, Field

class TokenClassifier(BaseModel):
    """
    Pydantic-compliant token classifier.
    Holds the vocab mapping and exposes classification helpers.
    """

    id_to_token: dict[int, str] = Field(...)

    class Config:
        arbitrary_types_allowed = True  # allows dict[int, str]

    def token_str(self, token_id: int) -> str:
        return self.id_to_token.get(token_id, "")

    # ---------- JSON structural tokens ----------
    def is_left_brace(self, token_id: int) -> bool:
        return self.token_str(token_id) == "{"

    def is_right_brace(self, token_id: int) -> bool:
        return self.token_str(token_id) == "}"

    def is_left_bracket(self, token_id: int) -> bool:
        return self.token_str(token_id) == "["

    def is_right_bracket(self, token_id: int) -> bool:
        return self.token_str(token_id) == "]"

    def is_colon(self, token_id: int) -> bool:
        return self.token_str(token_id) == ":"

    def is_comma(self, token_id: int) -> bool:
        return self.token_str(token_id) == ","

    def is_quote(self, token_id: int) -> bool:
        return self.token_str(token_id) == "\""

    # ---------- Number tokens ----------
    def is_digit(self, token_id: int) -> bool:
        return self.token_str(token_id).isdigit()

    def is_minus(self, token_id: int) -> bool:
        return self.token_str(token_id) == "-"

    def is_dot(self, token_id: int) -> bool:
        return self.token_str(token_id) == "."

    # ---------- Whitespace ----------
    def is_white_space(self, token_id: int) -> bool:
        return self.token_str(token_id).isspace()

    # ---------- String content ----------
    def is_string_char(self, token_id: int) -> bool:
        s = self.token_str(token_id)
        if s in ['"', '{', '}', '[', ']', ':', ',']:
            return False
        return True

    def is_number_char(self, token_id: int) -> bool:
        return self.is_digit(token_id) or self.is_minus(token_id) or self.is_dot(token_id)

    # ---------- Precomputed sets ----------
    def structural_tokens(self) -> set[int]:
        return {
            tid for tid, tok in self.id_to_token.items()
            if tok in ['{', '}', '[', ']', ':', ',', '"']
        }

    def digit_tokens(self) -> set[int]:
        return {
            tid for tid, tok in self.id_to_token.items()
            if tok.isdigit()
        }

    def string_tokens(self) -> set[int]:
        return {
            tid for tid, tok in self.id_to_token.items()
            if tok not in ['{', '}', '[', ']', ':', ',', '"']
        }
