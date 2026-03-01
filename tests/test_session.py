from __future__ import annotations

import unittest

from pipi.session import SessionManager
from pipi.types import ChatMessage, text_part


class SessionManagerTest(unittest.TestCase):
    def test_build_session_context_with_branch_summary(self) -> None:
        manager = SessionManager.in_memory()
        manager.append_model_change("openai", "gpt-4o-mini")
        user_id = manager.append_message(ChatMessage.user([text_part("hello")]))
        assistant_id = manager.append_message(ChatMessage.assistant([text_part("hi")], provider="openai", model="gpt-4o-mini"))
        manager.append_message(ChatMessage.user([text_part("old branch")]))
        manager.branch(assistant_id)
        manager.append_branch_summary("previous branch summary", assistant_id)
        manager.append_message(ChatMessage.user([text_part("new branch")]))

        context = manager.build_session_context()

        self.assertEqual(context.model, {"provider": "openai", "modelId": "gpt-4o-mini"})
        self.assertEqual(context.messages[0].text(), "hello")
        self.assertEqual(context.messages[1].text(), "hi")
        self.assertIn("previous branch summary", context.messages[2].text())
        self.assertEqual(context.messages[3].text(), "new branch")
        self.assertEqual(user_id is not None, True)


if __name__ == "__main__":
    unittest.main()
