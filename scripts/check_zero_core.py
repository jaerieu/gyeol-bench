#!/usr/bin/env python3
"""Static AST + regex zero-core & hygiene verifier for gyeol-bench (`Apache-2.0`).

Fails closed (exit 1, grade `[금지]`) if:
1. Any private engine module/symbol (`gyeol_lon`, `muk`, `chae`, `calcN`, or Korean private terms) appears.
2. Any visual-spectrum vocabulary appears anywhere in the repository.
3. Any Markdown file contains tilde (`~`) outside fenced code blocks or box-border characters.
4. Any Git remote is configured while in pre-Clock-A local staging mode (`--allow-remotes` overrides).
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List

DEFAULT_ROOT = Path(__file__).resolve().parents[1]

# Constructed cleanly so grep over non-audit files has zero false matches and audit rules stay explicit.
FORBIDDEN_PRIVATE_PATTERNS = [
    (r"(?:^|[^a-zA-Z0-9])gyeol_lon(?:$|[^a-zA-Z0-9])", "private_module:gyeol_lon"),
    (r"(?:^|[^a-zA-Z0-9])muk(?:$|[^a-zA-Z0-9]|_)", "private_symbol:muk"),
    (r"(?:^|[^a-zA-Z0-9])chae(?:$|[^a-zA-Z0-9]|_)", "private_symbol:chae"),
    (r"(?:^|[^a-zA-Z0-9])calcN(?:$|[^a-zA-Z0-9]|_)", "private_engine:calcN"),
    (r"먹", "private_ko_symbol:muk"),
    (r"채", "private_ko_symbol:chae"),
]

# Spectrum vocabulary regex built via unicode/token joins so even this file carries 0 raw spectrum words.
_EN_SPECTRUM_WORDS = [
    "r" + "ed",
    "gr" + "een",
    "yel" + "low",
    "am" + "ber",
    "ora" + "nge",
    "bl" + "ue",
    "pur" + "ple",
    "mage" + "nta",
    "cy" + "an",
    "crim" + "son",
    "emer" + "ald",
    "gr" + "ay",
    "gr" + "ey",
    "wh" + "ite",
    "bl" + "ack",
]
_KO_SPECTRUM_WORDS = [
    "\ubc18"[:0] + "\ube68\uac15",
    "\ube68\uac04",
    "\ub178\ub791",
    "\ub178\ub780",
    "\ucd08\ub85d",
    "\ub179\uc0c9",
    "\ud30c\ub791",
    "\ud30c\ub780",
    "\uc8fc\ud669",
    "\ubcf4\ub77c",
    "\ud68c\uc0c9",
    "\uac80\uc815",
    "\ud558\uc591",
    "\uc2e0\ud638\ub4f1",
]

SPECTRUM_RE = re.compile(
    r"\b(?:" + "|".join(_EN_SPECTRUM_WORDS) + r")\b|(?:" + "|".join(_KO_SPECTRUM_WORDS) + r")",
    re.IGNORECASE,
)

BOX_BORDER_RE = re.compile(r"[\u2500-\u257f]|^\s*\+[-=]{2,}\+", re.MULTILINE)
FENCED_BLOCK_RE = re.compile(r"```[\s\S]*?```")

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "build", "dist", ".pytest_cache"}


def _check_python_ast(py_path: Path, rel_str: str, violations: List[str]) -> None:
    source = py_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(py_path))
    except SyntaxError as exc:
        violations.append(f"{rel_str}:{exc.lineno}: Python syntax error: {exc.msg}")
        return

    forbidden_exact = {"gyeol_lon", "muk", "chae", "calcN"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                if root_pkg in forbidden_exact or "gyeol_lon" in alias.name:
                    violations.append(f"{rel_str}:{node.lineno}: forbidden import '{alias.name}'")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root_pkg = mod.split(".")[0] if mod else ""
            if root_pkg in forbidden_exact or "gyeol_lon" in mod:
                violations.append(f"{rel_str}:{node.lineno}: forbidden from-import '{mod}'")
            for alias in node.names:
                if alias.name in forbidden_exact:
                    violations.append(f"{rel_str}:{node.lineno}: forbidden imported symbol '{alias.name}'")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            low = node.name.lower()
            if any(tok.lower() in low for tok in forbidden_exact):
                violations.append(f"{rel_str}:{node.lineno}: forbidden definition name '{node.name}'")


def _check_markdown_hygiene(md_path: Path, rel_str: str, violations: List[str]) -> None:
    raw = md_path.read_text(encoding="utf-8")
    stripped = FENCED_BLOCK_RE.sub(lambda m: "\n" * m.group(0).count("\n"), raw)
    for idx, line in enumerate(stripped.splitlines(), start=1):
        if "~" in line:
            violations.append(f"{rel_str}:{idx}: tilde ('~') outside fenced code block")
        if BOX_BORDER_RE.search(line):
            violations.append(f"{rel_str}:{idx}: forbidden box-border character detected")


def audit_directory(root_dir: Path, allow_remotes: bool = False) -> Dict[str, Any]:
    root = root_dir.resolve()
    self_path = Path(__file__).resolve()
    violations: List[str] = []
    scanned_files = 0
    python_files = 0

    if not root.exists():
        return {
            "ok": False,
            "grade": "[금지]",
            "root": str(root),
            "scannedFiles": 0,
            "pythonFiles": 0,
            "violations": [f"Directory does not exist: {root}"],
        }

    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in path.parts):
            continue
        if not path.is_file():
            continue

        rel_str = str(path.relative_to(root))
        scanned_files += 1
        text = path.read_text(encoding="utf-8")

        if path.suffix == ".py":
            python_files += 1
            _check_python_ast(path, rel_str, violations)

        if path.suffix == ".md":
            _check_markdown_hygiene(path, rel_str, violations)

        # Spectrum check applies to EVERY file in the repo (including this script!).
        spec_hit = SPECTRUM_RE.search(text)
        if spec_hit:
            violations.append(f"{rel_str}: forbidden spectrum word '{spec_hit.group(0)}'")

        # Private-core regex check applies to all files except this static checker's pattern list.
        if path.resolve() != self_path:
            for pat, label in FORBIDDEN_PRIVATE_PATTERNS:
                flags = re.IGNORECASE if "private_" in label and "ko_symbol" not in label else 0
                if re.search(pat, text, flags=flags):
                    violations.append(f"{rel_str}: {label}")

    if not allow_remotes and (root / ".git").exists():
        proc = subprocess.run(
            ["git", "-C", str(root), "remote"],
            capture_output=True,
            text=True,
            check=False,
        )
        remotes = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        if remotes:
            violations.append(f".git/config: unexpected remotes configured before Clock A: {remotes}")

    ok = len(violations) == 0 and scanned_files > 0
    return {
        "ok": ok,
        "grade": "[안전]" if ok else "[금지]",
        "root": str(root),
        "scannedFiles": scanned_files,
        "pythonFiles": python_files,
        "violations": violations,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify zero private core and vocabulary hygiene.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Target repository directory")
    parser.add_argument("--allow-remotes", action="store_true", help="Skip zero-remote check")
    args = parser.parse_args(argv)

    report = audit_directory(args.root, allow_remotes=args.allow_remotes)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
