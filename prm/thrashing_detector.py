import re

from prm.base import PRMJudgment, ProcessRewardModel, register_prm


@register_prm
class ThrashingDetector(ProcessRewardModel):
    """Rule-based PRM that detects agent thrashing.

    Checks every `check_interval` turns (starting after `min_turns`).
    Compares the last `window_size` assistant actions for repetitive patterns
    using Jaccard similarity on extracted command tokens.
    """

    def __init__(
        self,
        window_size: int = 6,
        similarity_threshold: float = 0.5,
        min_turns: int = 10,
        check_interval: int = 3,
        **kwargs,
    ):
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.min_turns = min_turns
        self.check_interval = check_interval

    @classmethod
    def name(cls) -> str:
        return "thrashing_detector"

    def should_terminate(self, turn, trajectory_steps, messages):
        if turn < self.min_turns:
            return False
        if turn % self.check_interval != 0:
            return False

        # Extract assistant message content from last window_size turns
        assistant_contents = []
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                assistant_contents.append(msg.get("content", ""))
            if len(assistant_contents) >= self.window_size:
                break
        assistant_contents.reverse()

        if len(assistant_contents) < self.window_size:
            return False

        return self._is_repetitive(assistant_contents)

    def judge(self, trajectory, reward, metadata):
        # Post-hoc: check if trajectory was thrashing
        assistant_msgs = [
            m["content"] for m in trajectory if m.get("role") == "assistant"
        ]
        if len(assistant_msgs) >= self.window_size:
            is_thrashing = self._is_repetitive(assistant_msgs[-self.window_size :])
        else:
            is_thrashing = False

        return PRMJudgment(
            adjusted_reward=reward,
            is_thrashing=is_thrashing,
            productive_turns=len(assistant_msgs),
        )

    def _is_repetitive(self, messages: list[str]) -> bool:
        """Check if messages show repetitive command patterns."""
        token_sets = [self._extract_command_tokens(m) for m in messages]
        if not token_sets or not any(token_sets):
            return False

        # Average pairwise Jaccard similarity
        similarities = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                if token_sets[i] and token_sets[j]:
                    intersection = token_sets[i] & token_sets[j]
                    union = token_sets[i] | token_sets[j]
                    similarities.append(
                        len(intersection) / len(union) if union else 0
                    )

        if not similarities:
            return False
        avg_similarity = sum(similarities) / len(similarities)
        return avg_similarity >= self.similarity_threshold

    @staticmethod
    def _extract_command_tokens(content: str) -> set[str]:
        """Extract command-like tokens from assistant message content."""
        # Look for shell commands in keystrokes fields or code blocks
        commands = re.findall(r'"keystrokes"\s*:\s*"([^"]*)"', content)
        commands.extend(
            re.findall(r"```(?:bash|sh|python)?\n(.*?)```", content, re.DOTALL)
        )
        # Tokenize
        tokens = set()
        for cmd in commands:
            tokens.update(cmd.split())
        return tokens
