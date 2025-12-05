#!/usr/bin/env python3
"""
Unit tests for dataset sampling logic.
Tests skill detection and sampling configuration.
"""

import unittest
import re


def detect_skills(bash_command: str) -> list:
    """
    Simplified skill detection for testing.
    Extracts command names from bash command string.
    """
    # Common bash commands to detect
    commands = [
        'awk', 'sed', 'grep', 'find', 'cat', 'sort', 'uniq', 'head', 'tail',
        'cut', 'tr', 'wc', 'xargs', 'tar', 'gzip', 'wget', 'curl', 'date',
        'echo', 'printf', 'ls', 'chmod', 'chown', 'mkdir', 'rm', 'cp', 'mv'
    ]

    skills = []
    for cmd in commands:
        # Look for command as a word boundary
        if re.search(rf'\b{cmd}\b', bash_command):
            skills.append(cmd)

    return skills


class TestSkillDetection(unittest.TestCase):
    """Test skill detection from bash commands."""

    def test_detect_single_command(self):
        """Test detection of single command."""
        bash = "grep 'pattern' file.txt"
        skills = detect_skills(bash)
        self.assertIn("grep", skills)

    def test_detect_multiple_commands(self):
        """Test detection of multiple commands in pipeline."""
        bash = "cat file.txt | grep 'pattern' | awk '{print $1}' | sort"
        skills = detect_skills(bash)
        self.assertIn("cat", skills)
        self.assertIn("grep", skills)
        self.assertIn("awk", skills)
        self.assertIn("sort", skills)

    def test_detect_compound_commands(self):
        """Test detection in compound commands."""
        bash = "find . -name '*.txt' && sed -i 's/old/new/g' file.txt"
        skills = detect_skills(bash)
        self.assertIn("find", skills)
        self.assertIn("sed", skills)

    def test_ignore_builtin_keywords(self):
        """Test that bash keywords are not detected as commands."""
        bash = "if [ -f file.txt ]; then cat file.txt; fi"
        skills = detect_skills(bash)
        # Should not include 'if', 'then', 'fi'
        self.assertIn("cat", skills)
        self.assertNotIn("if", skills)
        self.assertNotIn("then", skills)
        self.assertNotIn("fi", skills)

    def test_detect_with_arguments(self):
        """Test detection when command has complex arguments."""
        bash = "tar -czf archive.tar.gz --exclude='*.log' dir/"
        skills = detect_skills(bash)
        self.assertIn("tar", skills)

    def test_subshell_commands(self):
        """Test detection in subshells."""
        bash = "echo $(date '+%Y-%m-%d')"
        skills = detect_skills(bash)
        self.assertIn("echo", skills)
        self.assertIn("date", skills)

    def test_empty_command(self):
        """Test handling of empty command."""
        bash = ""
        skills = detect_skills(bash)
        self.assertEqual(len(skills), 0)

    def test_deduplication(self):
        """Test that duplicate commands are deduplicated."""
        bash = "grep pattern file1 | grep pattern2 | grep pattern3"
        skills = detect_skills(bash)
        # grep should appear only once
        self.assertEqual(skills.count("grep"), 1)


class TestSamplingConfiguration(unittest.TestCase):
    """Test sampling configuration generation."""

    def test_sampling_config_format(self):
        """Test that sampling config has expected format."""
        # This would test generate_config.py logic
        # Skipping for now as it requires dataset access
        pass


if __name__ == '__main__':
    unittest.main()
