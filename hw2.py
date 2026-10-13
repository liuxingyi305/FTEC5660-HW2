#!/usr/bin/env python3
"""FTEC5660 HW2 student starter: build an agent that verifies CVs via MCP."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


MCP_URL = "https://ftec5660.ngrok.app/mcp"
MODEL_NAME = "deepseek-v4-flash"
THRESHOLD = 0.5


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def cv_files(folder: Path) -> list[Path]:
    """Return PDFs directly inside *folder*, sorted numerically (CV_2 before CV_10)."""

    def key(path: Path) -> tuple[int, str]:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return (int(digits) if digits else math.inf, path.name)

    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=key,
    )


def cv_text(path: Path) -> str:
    """Convert one CV PDF to markdown text."""
    from markitdown import MarkItDown

    return MarkItDown(enable_plugins=False).convert(str(path)).text_content


async def load_mcp_tools() -> list[Any]:
    """Connect to the course MCP server and return its tools as LangChain tools."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "social_graph": {
                "transport": "http",
                "url": MCP_URL,
                "headers": {"ngrok-skip-browser-warning": "true"},
            }
        }
    )
    return await client.get_tools()


def build_agent(tools: list[Any]) -> Any:
    """Create and return your agent once.

    ``tools`` are the six SocialGraph MCP tools (Facebook + LinkedIn search and
    profile lookup), already wrapped as LangChain tools. You may add your own
    local tools as well.

    Suggested imports:
        from langchain_deepseek import ChatDeepSeek
        from langchain.agents import create_agent

    Use the DeepSeek model named by ``MODEL_NAME``. The API key is loaded
    from .env.
    """
    ### YOUR CODE HERE
    from langchain_deepseek import ChatDeepSeek
    from langchain.agents import create_agent

    model = ChatDeepSeek(model=MODEL_NAME, temperature=0)

    system_prompt = """
You are a rigorous KYC / CV verification agent. You may ONLY use the provided SocialGraph MCP tools. Never use web search or any other data source.

Goal: decide whether the key claims in the CV are consistent with the candidate's LinkedIn and Facebook profiles. LinkedIn is the primary source; Facebook is used for cross-checking.

FIELDS TO VERIFY (structured fields only):
- name
- city
- jobs: company, title, seniority, start_year, end_year
- education: degree, school, field, graduation year
- skills

NOT discrepancies:
- wording differences: "Bachelor of Science" vs "BSc", "UI/UX Design" vs "UI/UX", "Senior Engineer" vs "Engineer"
- the CV lists FEWER skills than LinkedIn
- differences in job description, headline, or hometown

DISCREPANCIES (any one is enough to fail the CV):
- INFLATED JOB TITLE: the CV title implies HIGHER seniority than LinkedIn.
  Compare BOTH the LinkedIn title AND the LinkedIn seniority field.
  Examples that ARE discrepancies:
    * CV "Senior Analyst" vs LinkedIn "Analyst" with seniority "junior"
    * CV "Senior Engineer" vs LinkedIn "Engineer" with seniority "mid"
    * CV "Manager" vs LinkedIn "Analyst"
  Rule: if the CV title adds words like Senior / Lead / Head / Manager / Director / Principal
  that are NOT supported by the LinkedIn title or seniority, it is an inflated title -> discrepancy.
  Only ignore wording when seniority is the SAME, e.g. CV "BSc" vs LinkedIn "Bachelor of Science".
- employment or graduation YEARS that differ by even one year (e.g. CV 2015 vs LinkedIn 2014)
- an upgraded degree (e.g. CV says MSc, LinkedIn says BSc)
- a fake school or employer
- a wrong location
- a skill listed on the CV that is NOT in the LinkedIn skills list (e.g. CV lists Docker, LinkedIn has no Docker)

PROCEDURE (do this step by step, do not skip):
1. Extract from the CV: full name, city, every job (company, title, start_year, end_year), every education entry (school, degree, field, graduation year), every skill.
2. Search LinkedIn with the name and location. If several people match, prefer the one whose company / school / years match. Call get_linkedin_profile on the right person_id.
3. Compare each field one by one:
   - name: exact or clearly same person
   - city: exact
   - each job: company exact, title consistent (ignore mere wording), start_year EXACT, end_year EXACT
   - each education: school exact, degree consistent, field consistent, graduation year EXACT
   - each skill listed on the CV: MUST appear on LinkedIn skills (wording-only variants OK)
4. Cross-check with Facebook (search by display name, then get_facebook_profile).
5. Decide:
   - If ANY structured field conflicts (year, company, school, degree, location, extra skill): score in [0.0, 0.2]
   - If everything matches (or only wording/fewer-skills differences): score in [0.8, 1.0]
   - If genuinely uncertain because the person cannot be identified: score around 0.5

Your FINAL answer MUST be a single JSON object with a short reason, e.g.:
{"reason": "CV says start_year 2015, LinkedIn says 2014", "score": 0.1}
Nothing else after the JSON.
"""

    agent = create_agent(model, tools, system_prompt=system_prompt)
    return agent


async def score_cvs(agent: Any, cvs: dict[str, str]) -> dict[str, float | None]:
    """Run your agent and return one reliability score per CV.

    ``cvs`` maps each file name to its text, e.g. ``{"CV_1.pdf": "...", ...}``.
    Return a float in [0, 1] for every file name: higher means the CV is more
    likely consistent with the candidate's LinkedIn/Facebook data. A score
    above 0.5 counts as "valid", 0.5 or below counts as "has discrepancy".

        {"CV_1.pdf": 0.9, "CV_4.pdf": 0.1, ...}

    Catch errors per CV (e.g. a failed API call) and still return a score for
    it: an exception here means no results.csv, which scores zero.

    MCP tools are async, so call your agent with ``await agent.ainvoke(...)``.
    You may verify CVs in parallel (e.g. ``asyncio.gather``), but keep at most
    about 3 CVs in flight (e.g. with ``asyncio.Semaphore(3)``): the MCP server is
    shared by the whole class.
    """
    ### YOUR CODE HERE
    import re as _re

    sem = asyncio.Semaphore(3)

    def _extract_score(text):
        m = _re.search(r'\{\s*"score"\s*:\s*([01](?:\.\d+)?|0?\.\d+)\s*\}', text)
        if m:
            try:
                val = float(m.group(1))
                if 0.0 <= val <= 1.0:
                    return val
            except ValueError:
                pass
        m2 = _re.search(r'"score"\s*:\s*([01](?:\.\d+)?|0?\.\d+)', text)
        if m2:
            try:
                val = float(m2.group(1))
                if 0.0 <= val <= 1.0:
                    return val
            except ValueError:
                pass
        return None

    async def _score_one(name, text):
        async with sem:
            prompt = (
                "Verify the following CV against the candidate's LinkedIn and Facebook profiles. "
                "Use only the SocialGraph MCP tools. "
                'Reply with a single JSON object only, e.g. {"score": 0.9}.\n\n'
                f"CV file name: {name}\nCV content:\n{text}"
            )
            for _attempt in range(3):
                try:
                    result = await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
                    messages = result.get("messages", []) if isinstance(result, dict) else []
                    if not messages:
                        raise ValueError("no messages")
                    last = messages[-1]
                    content = getattr(last, "content", last)
                    if isinstance(content, list):
                        content = " ".join(
                            str(part.get("text", part)) if isinstance(part, dict) else str(part)
                            for part in content
                        )
                    score = _extract_score(str(content))
                    if score is not None:
                        return name, score
                except Exception:
                    await asyncio.sleep(1.0)
            return name, 0.0

    results = await asyncio.gather(*(_score_one(n, t) for n, t in cvs.items()))
    return dict(results)


# Everything below is provided runner/scoring code. No edits are needed.

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_score(value: Any) -> float | None:
    """Accept a float/int, or text containing exactly one number, in [0, 1]."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
    else:
        text = str(getattr(value, "content", value))
        matches = _NUMBER_RE.findall(text)
        if len(matches) != 1:
            return None
        score = float(matches[0])
    if math.isnan(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def read_ground_truth(folder: Path) -> dict[str, dict[str, Any]]:
    """Read labels (1 = valid CV, 0 = has discrepancy) and reasons from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    return {
        name: entry if isinstance(entry, dict) else {"label": entry}
        for name, entry in json.loads(path.read_text(encoding="utf-8")).items()
    }


def correctness_text(score: float | None, expected: dict[str, Any] | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if score is None:
        return "incorrect: score is missing or not a number in [0, 1]"
    if expected is None:
        return "not graded: no ground truth for this CV"
    label = int(expected["label"])
    predicted = 1 if score > THRESHOLD else 0
    if predicted == label:
        return "correct"
    reason = f" ({expected['reason']})" if expected.get("reason") else ""
    return f"incorrect: expected {label}{reason}, predicted {predicted}"


def write_results(names: list[str], scores: dict[str, Any], truth: dict[str, dict[str, Any]]) -> tuple[Path, int]:
    """Write the required results.csv file and return how many CVs were correct."""
    output = Path("results.csv")
    correct = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cv", "score", "correctness"])
        for name in names:
            score = parse_score(scores.get(name))
            verdict = correctness_text(score, truth.get(name))
            correct += verdict == "correct"
            writer.writerow([name, "" if score is None else f"{score:.4f}", verdict])
    return output, correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW2 on CV PDFs")
    parser.add_argument(
        "--cv-folder",
        required=True,
        type=Path,
        help="folder containing CV PDF files",
    )
    return parser.parse_args()


async def run(folder: Path) -> int:
    paths = cv_files(folder)
    if not paths:
        raise SystemExit(f"no PDF files found in {folder}")

    load_env_file()
    cvs = {path.name: cv_text(path) for path in paths}
    tools = await load_mcp_tools()
    agent = build_agent(tools)
    scores = await score_cvs(agent, cvs)
    if not isinstance(scores, dict):
        raise TypeError("score_cvs() must return a dictionary")

    truth = read_ground_truth(folder)
    output, correct = write_results(list(cvs), scores, truth)
    summary = f" Accuracy: {correct}/{len(cvs)}." if truth else ""
    print(f"Processed {len(cvs)} CV(s). Wrote {output}.{summary}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.cv_folder.is_dir():
        raise SystemExit(f"not a folder: {args.cv_folder}")
    return asyncio.run(run(args.cv_folder))


if __name__ == "__main__":
    raise SystemExit(main())
