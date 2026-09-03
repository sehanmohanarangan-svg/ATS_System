"""
Batch Resume Scorer
====================
Compare a folder of parsed resume JSONs against a single job posting.
Outputs a ranked Excel file with scores, grades, and gap analysis.

Usage:
    python batch_match.py resumes/ job_posting.pdf
    python batch_match.py resumes/ job_posting.pdf --output results.xlsx
    python batch_match.py resumes/ job_posting.pdf --workers 4
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import openpyxl
from openpyxl.styles import (
    Font, PatternFill, Alignment, Border, Side, GradientFill
)
from openpyxl.utils import get_column_letter

# Import your existing modules
from SemanticEngine import HDCSearch
from resume_matcher import (
    TextReader, JobParser,
    score_skills, score_experience, score_education,
    score_summary, score_certifications, score_projects,
    WEIGHTS, _grade, _recommendation,
)


# ---------------------------------------------------------------------------
# Batch scorer
# ---------------------------------------------------------------------------

class BatchScorer:
    def __init__(self) -> None:
        print("Loading semantic model...", flush=True)
        self._sem     = HDCSearch()
        self._reader  = TextReader()
        self._jparser = JobParser()
        print("Model ready.", flush=True)

    def load_job(self, job_path: Path) -> dict[str, Any]:
        """Parse and pre-encode the job posting once for all resumes."""
        raw   = self._reader.read(job_path)
        clean = self._reader.clean(raw)
        job   = self._jparser.parse(clean)

        # Pre-encode job texts into cache
        desc = job.get("description") or job.get("full_text") or ""
        if desc:
            self._sem.encode(desc[:1500])

        job_skills = job.get("required_skills") or job.get("all_skills") or []
        if job_skills:
            self._sem.encode_batch(job_skills)

        print(f"Job: {job.get('title') or job_path.name}", flush=True)
        return job

    def score_one(self, resume: dict, job: dict) -> dict[str, Any]:
        """Score a single resume against the pre-loaded job."""
        # Warm cache for this resume
        self._sem.warm_batch(resume)

        sections = {
            "skills":         score_skills(resume, job, self._sem),
            "experience":     score_experience(resume, job, self._sem),
            "education":      score_education(resume, job, self._sem),
            "summary":        score_summary(resume, job, self._sem),
            "certifications": score_certifications(resume, job, self._sem),
            "projects":       score_projects(resume, job, self._sem),
        }

        total = sum(s.weighted for s in sections.values())

        matched_skills = sections["skills"].matched[:5]
        missing_skills = sections["skills"].missing[:5]
        gaps = [
            f"{k} ({WEIGHTS[k]}pts)"
            for k, s in sections.items()
            if s.raw < 0.50 and WEIGHTS[k] >= 10
        ]
        for s in sections.values():
            gaps += s.missing[:2]

        derived  = resume.get("derived") or {}
        contact  = resume.get("contact") or {}

        return {
            # Identity
            "name":             resume.get("name") or "Unknown",
            "email":            contact.get("email") or "",
            "phone":            contact.get("phone") or "",
            "location":         contact.get("location") or "",
            "source_file":      resume.get("source_file") or "",
            # Overall
            "total_score":      round(total, 1),
            "grade":            _grade(total),
            "recommendation":   _recommendation(total, sections),
            # Section scores (weighted points)
            "skills_score":     round(sections["skills"].weighted, 1),
            "experience_score": round(sections["experience"].weighted, 1),
            "education_score":  round(sections["education"].weighted, 1),
            "summary_score":    round(sections["summary"].weighted, 1),
            "certs_score":      round(sections["certifications"].weighted, 1),
            "projects_score":   round(sections["projects"].weighted, 1),
            # Raw ratios (0–1) for conditional formatting
            "skills_raw":       round(sections["skills"].raw, 3),
            "experience_raw":   round(sections["experience"].raw, 3),
            "education_raw":    round(sections["education"].raw, 3),
            "summary_raw":      round(sections["summary"].raw, 3),
            # Details
            "years_experience": derived.get("years_experience") or "",
            "education_level":  derived.get("education_level") or "",
            "matched_skills":   ", ".join(matched_skills),
            "missing_skills":   ", ".join(missing_skills),
            "top_gaps":         " | ".join(gaps[:4]),
            "notes":            " | ".join(
                n for s in sections.values() for n in s.notes[:1]
            ),
        }

    def score_folder(
        self,
        folder: Path,
        job: dict,
        workers: int = 1,
    ) -> list[dict[str, Any]]:
        json_files = sorted(folder.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {folder}")

        print(f"Scoring {len(json_files)} resumes...", flush=True)
        results: list[dict[str, Any]] = []
        failed: list[str] = []

        # NOTE: HDCSearch has a shared cache + lock — workers > 1 is safe
        # but on CPU it won't be faster since encode is the bottleneck.
        # Keep workers=1 unless you have a GPU.
        def _process(path: Path) -> dict[str, Any]:
            raw = path.read_bytes()
            for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
                try:
                    resume = json.loads(raw.decode(enc))
                    break
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
            else:
                raise ValueError(f"Could not decode {path.name}")
            return self.score_one(resume, job)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process, p): p for p in json_files}
            done = 0
            for future in as_completed(futures):
                path = futures[future]
                done += 1
                try:
                    result = future.result()
                    results.append(result)
                    print(
                        f"  [{done}/{len(json_files)}] "
                        f"{result['name']:<30} {result['total_score']:5.1f}  {result['grade']}",
                        flush=True,
                    )
                except Exception as e:
                    failed.append(path.name)
                    print(f"  [{done}/{len(json_files)}] FAILED {path.name}: {e}", flush=True)

        if failed:
            print(f"\nFailed files ({len(failed)}): {', '.join(failed)}", flush=True)

        # Sort by score descending
        results.sort(key=lambda r: r["total_score"], reverse=True)
        return results


# ---------------------------------------------------------------------------
# Excel writer
# ---------------------------------------------------------------------------

def _hex(h: str) -> str:
    return h.lstrip("#")

def _fill(hex_color: str) -> PatternFill:
    return PatternFill("solid", fgColor=_hex(hex_color))

def _font(bold=False, color="000000", size=11) -> Font:
    return Font(bold=bold, color=_hex(color), size=size, name="Arial")

def _border() -> Border:
    s = Side(style="thin", color="CCCCCC")
    return Border(left=s, right=s, top=s, bottom=s)

def _score_color(raw: float) -> str:
    """Green → Yellow → Red based on 0–1 raw score."""
    if raw >= 0.70: return "C6EFCE"   # green
    if raw >= 0.50: return "FFEB9C"   # yellow
    return "FFC7CE"                   # red

def _grade_color(grade: str) -> str:
    return {
        "A": "375623", "B": "006100", "C": "9C5700",
        "D": "9C0006", "F": "9C0006",
    }.get(grade, "000000")


def write_excel(results: list[dict], job: dict, output: Path) -> None:
    wb = openpyxl.Workbook()

    # ------------------------------------------------------------------ Sheet 1: Rankings
    ws = wb.active
    ws.title = "Rankings"

    # Title row
    job_title = job.get("title") or "Job Posting"
    ws.merge_cells("A1:T1")
    ws["A1"] = f"Resume Match Results — {job_title}"
    ws["A1"].font      = Font(bold=True, size=14, name="Arial", color="FFFFFF")
    ws["A1"].fill      = _fill("1F3864")
    ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 28

    # Column headers
    headers = [
        ("Rank",             6),
        ("Name",            22),
        ("Score /100",      11),
        ("Grade",            7),
        ("Skills /35",      10),
        ("Exp /20",          9),
        ("Edu /15",          9),
        ("Summary /15",     11),
        ("Certs /10",        9),
        ("Projects /5",     10),
        ("Yrs Exp",          8),
        ("Edu Level",       11),
        ("Matched Skills",  28),
        ("Missing Skills",  28),
        ("Top Gaps",        30),
        ("Recommendation",  35),
        ("Email",           24),
        ("Phone",           14),
        ("Location",        18),
        ("Source File",     20),
    ]

    HDR_FILL = _fill("2E75B6")
    for col, (hdr, width) in enumerate(headers, 1):
        cell = ws.cell(row=2, column=col, value=hdr)
        cell.font      = Font(bold=True, color="FFFFFF", name="Arial", size=10)
        cell.fill      = HDR_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border    = _border()
        ws.column_dimensions[get_column_letter(col)].width = width
    ws.row_dimensions[2].height = 30

    # Data rows
    fields = [
        None,  # rank — computed
        "name", "total_score", "grade",
        "skills_score", "experience_score", "education_score",
        "summary_score", "certs_score", "projects_score",
        "years_experience", "education_level",
        "matched_skills", "missing_skills", "top_gaps",
        "recommendation", "email", "phone", "location", "source_file",
    ]

    # Raw keys for conditional formatting (parallel to score columns)
    raw_keys = {
        5: "skills_raw", 6: "experience_raw",
        7: "education_raw", 8: "summary_raw",
    }

    for row_idx, res in enumerate(results, 3):
        alt = (row_idx % 2 == 0)
        for col_idx, field in enumerate(fields, 1):
            if field is None:
                value = row_idx - 2   # rank
            else:
                value = res.get(field, "")

            cell = ws.cell(row=row_idx, column=col_idx, value=value)
            cell.font      = _font(size=10)
            cell.border    = _border()
            cell.alignment = Alignment(
                vertical="center", wrap_text=(col_idx >= 13)
            )

            # Alternating row background
            if alt:
                cell.fill = _fill("EBF3FB")

            # Score column (col 3) — color by total
            if col_idx == 3:
                total = res.get("total_score", 0)
                cell.fill = _fill(_score_color(total / 100))
                cell.font = Font(bold=True, name="Arial", size=10)
                cell.alignment = Alignment(horizontal="center", vertical="center")

            # Grade column (col 4)
            if col_idx == 4:
                cell.font = Font(
                    bold=True, name="Arial", size=10,
                    color=_grade_color(str(value))
                )
                cell.alignment = Alignment(horizontal="center", vertical="center")

            # Section score columns (5–10) — color by raw ratio
            if col_idx in raw_keys:
                raw = res.get(raw_keys[col_idx], 0)
                cell.fill = _fill(_score_color(raw))
                cell.alignment = Alignment(horizontal="center", vertical="center")
            elif col_idx in (5, 6, 7, 8, 9, 10):
                cell.alignment = Alignment(horizontal="center", vertical="center")

        ws.row_dimensions[row_idx].height = 18

    # Freeze panes + auto-filter
    ws.freeze_panes = "A3"
    ws.auto_filter.ref = f"A2:{get_column_letter(len(headers))}2"

    # ------------------------------------------------------------------ Sheet 2: Summary stats
    ws2 = wb.create_sheet("Summary")
    ws2.column_dimensions["A"].width = 28
    ws2.column_dimensions["B"].width = 18

    def _stat_row(r, label, value, bold=False):
        a = ws2.cell(row=r, column=1, value=label)
        b = ws2.cell(row=r, column=2, value=value)
        a.font = _font(bold=bold, size=11)
        b.font = _font(bold=bold, size=11)
        a.border = _border()
        b.border = _border()
        b.alignment = Alignment(horizontal="center")

    ws2["A1"] = "Batch Match Summary"
    ws2["A1"].font = Font(bold=True, size=14, name="Arial")
    ws2.merge_cells("A1:B1")
    ws2.row_dimensions[1].height = 24

    scores = [r["total_score"] for r in results]
    grades = [r["grade"] for r in results]

    stats = [
        ("Job Title",           job.get("title") or "—"),
        ("Total Resumes",       len(results)),
        ("Average Score",       f"{sum(scores)/len(scores):.1f}" if scores else "—"),
        ("Highest Score",       f"{max(scores):.1f}" if scores else "—"),
        ("Lowest Score",        f"{min(scores):.1f}" if scores else "—"),
        ("Grade A (≥85)",       grades.count("A")),
        ("Grade B (70–84)",     grades.count("B")),
        ("Grade C (55–69)",     grades.count("C")),
        ("Grade D (40–54)",     grades.count("D")),
        ("Grade F (<40)",       grades.count("F")),
        ("Top Candidate",       results[0]["name"] if results else "—"),
        ("Top Score",           f"{results[0]['total_score']:.1f}" if results else "—"),
    ]

    for i, (label, value) in enumerate(stats, 2):
        _stat_row(i, label, value, bold=(i == 2))

    # ------------------------------------------------------------------ Sheet 3: Raw data (CSV-friendly)
    ws3 = wb.create_sheet("Raw Data")
    raw_headers = [
        "rank", "name", "total_score", "grade",
        "skills_score", "experience_score", "education_score",
        "summary_score", "certs_score", "projects_score",
        "years_experience", "education_level",
        "matched_skills", "missing_skills", "top_gaps",
        "recommendation", "email", "phone", "location", "source_file",
    ]
    ws3.append(raw_headers)
    for rank, res in enumerate(results, 1):
        ws3.append([rank] + [res.get(h, "") for h in raw_headers[1:]])

    wb.save(output)
    print(f"\nSaved: {output}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score a folder of resume JSONs against one job posting."
    )
    ap.add_argument("resumes_folder", type=Path,
                    help="Folder containing parsed resume .json files")
    ap.add_argument("job_posting",    type=Path,
                    help="Job posting PDF or text file")
    ap.add_argument("--output",  type=Path, default=None,
                    help="Output .xlsx path (default: results.xlsx next to job posting)")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel workers (default 1; GPU: try 2–4)")
    args = ap.parse_args()

    if not args.resumes_folder.is_dir():
        ap.error(f"Not a directory: {args.resumes_folder}")
    if not args.job_posting.is_file():
        ap.error(f"Job posting not found: {args.job_posting}")

    output = args.output or args.job_posting.with_name(
        args.job_posting.stem + "_results.xlsx"
    )

    t0      = time.perf_counter()
    scorer  = BatchScorer()
    job     = scorer.load_job(args.job_posting)
    results = scorer.score_folder(args.resumes_folder, job, workers=args.workers)

    if not results:
        ap.error("No resumes were scored successfully.")

    write_excel(results, job, output)
    elapsed = time.perf_counter() - t0
    print(f"Done. {len(results)} resumes in {elapsed:.1f}s "
          f"({elapsed/len(results):.1f}s each)", flush=True)


if __name__ == "__main__":
    main()