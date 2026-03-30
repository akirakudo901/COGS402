import unittest

from llm_prolog import system_prompts as sp


class SystemPromptHashingTests(unittest.TestCase):
    def test_hashes_roundtrip_lookup(self) -> None:
        for name, prompt_text in sp.SYSTEM_PROMPTS_BY_NAME.items():
            expected_hash = sp.SYSTEM_PROMPT_HASHES_BY_NAME[name]
            actual_hash = sp.hash_system_prompt_text(prompt_text)
            self.assertEqual(actual_hash, expected_hash)

            looked_up = sp.get_system_prompt_by_hash(expected_hash)
            self.assertEqual(looked_up, prompt_text)

            self.assertEqual(sp.get_canonical_system_prompt_name_by_hash(expected_hash), name)

    def test_unknown_hash_returns_none(self) -> None:
        unknown = "0" * 64
        # Very unlikely to collide with a canonical prompt hash.
        if unknown in sp.SYSTEM_PROMPT_HASHES_BY_NAME.values():
            # Skip collision case.
            return
        self.assertIsNone(sp.get_system_prompt_by_hash(unknown))
        self.assertIsNone(sp.get_canonical_system_prompt_name_by_hash(unknown))


if __name__ == "__main__":
    unittest.main()

