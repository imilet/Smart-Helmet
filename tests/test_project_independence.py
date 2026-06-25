from pathlib import Path
import unittest


class ProjectIndependenceTest(unittest.TestCase):
    def test_source_does_not_reference_external_project_code(self):
        project_a = "".join(["Pytorch", "-", "Mobile", "Face", "Net"])
        project_b = "".join(["Smart", "_", "Construction", "-", "master"])
        forbidden = [
            "/".join(["", "home", "lowkeng", "code", "LVAN", project_a]),
            "/".join(["", "home", "lowkeng", "code", "LVAN", project_b]),
            project_a,
            project_b,
            ".".join(["sys", "path", "insert"]),
        ]
        files = [
            path
            for path in Path(".").rglob("*")
            if path.suffix in {".py", ".md"}
            and ".git" not in path.parts
            and "__pycache__" not in path.parts
        ]

        violations = []
        for path in files:
            text = path.read_text(encoding="utf-8")
            for item in forbidden:
                if item in text:
                    violations.append(f"{path}: {item}")

        self.assertEqual([], violations)


if __name__ == "__main__":
    unittest.main()
