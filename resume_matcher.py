"""
Hybrid Resume-to-Job Match Scorer
==================================
Scores a parsed resume JSON against a job posting PDF (or text).

Scoring breakdown (100 pts total):
  - Skills match       35 pts  hybrid: semantic + keyword overlap
  - Experience         20 pts  title/highlight relevance + years
  - Education          15 pts  degree level + field relevance
  - Summary/Profile    15 pts  semantic similarity to job description
  - Certifications     10 pts  keyword match against requirements
  - Projects            5 pts  semantic relevance to job

Usage:
    python resume_matcher.py resume.json job_posting.pdf
    python resume_matcher.py resume.json job_posting.txt
    python resume_matcher.py resume.json job_posting.pdf --detail
    python resume_matcher.py resume.json job_posting.pdf --output report.json

Dependencies:
    pip install pdfplumber sentence-transformers
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import numpy as np
from SemanticEngine import HDCSearch

# ---------------------------------------------------------------------------
# Scoring weights (must sum to 100)
# ---------------------------------------------------------------------------

WEIGHTS = {
    "skills":          35,
    "experience":      20,
    "education":       15,
    "summary":         15,
    "certifications":  10,
    "projects":         5,
}
assert sum(WEIGHTS.values()) == 100


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SectionScore:
    raw: float          # 0.0 – 1.0
    weighted: float     # raw * weight
    max_points: int
    notes: list[str] = field(default_factory=list)
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


@dataclass
class MatchReport:
    candidate_name: str
    total_score: float          # 0–100
    grade: str                  # A / B / C / D / F
    sections: dict[str, SectionScore]
    top_strengths: list[str]
    gaps: list[str]
    recommendation: str


# ---------------------------------------------------------------------------
# PDF / text reader  (same lightweight reader as resume_parser)
# ---------------------------------------------------------------------------

class TextReader:
    def read(self, path: Path) -> str:
        if path.suffix.lower() != ".pdf":
            return path.read_text(encoding="utf-8", errors="replace")
        try:
            import pdfplumber
        except ImportError:
            raise RuntimeError("pip install pdfplumber")
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)

    def clean(self, text: str) -> str:
        for bad, good in [
            ("\u2022","•"),("\u00a0"," "),("\u2013","-"),("\u2014","-"),
            ("\u2019","'"),("\u2018","'"),
        ]:
            text = text.replace(bad, good)
        text = re.sub(r"[ \t]+"," ", text)
        text = re.sub(r" *\n *","\n", text)
        text = re.sub(r"\n{3,}","\n\n", text)
        return text.strip()


# ---------------------------------------------------------------------------
# Job posting parser  (extracts structured fields from raw JD text)
# ---------------------------------------------------------------------------

class JobParser:
    """Heuristically extracts key fields from a job posting."""

    # Degree level keywords mapped to numeric level
    _DEGREE_LEVELS = {
        "phd": 5, "ph.d": 5, "doctorate": 5,
        "master": 4, "m.sc": 4, "m.eng": 4, "mba": 4,
        "bachelor": 3, "b.sc": 3, "b.eng": 3, "b.a": 3, "undergraduate": 3,
        "diploma": 2, "college": 2,
        "high school": 1, "secondary": 1,
    }

    def parse(self, text: str) -> dict[str, Any]:
        return {
            "full_text":        text,
            "title":            self._job_title(text),
            "required_skills":  self._skills(text, required=True),
            "preferred_skills": self._skills(text, required=False),
            "all_skills":       self._all_skills(text),
            "min_years_exp":    self._years_exp(text),
            "degree_required":  self._degree_level(text),
            "degree_field":     self._degree_field(text),
            "certifications":   self._certs(text),
            "description":      self._description(text),
        }

    def _job_title(self, text: str) -> str | None:
        # First non-empty line is usually the job title
        for line in text.splitlines():
            line = line.strip()
            if line and len(line) < 80:
                return line
        return None

    def _section(self, text: str, *headings: str) -> str:
        """Extract text under any of the given headings."""
        pattern = "|".join(re.escape(h) for h in headings)
        m = re.search(
            rf"(?im)^(?:{pattern})[:\s]*\n(.*?)(?=\n[A-Z][A-Za-z ]+[:\n]|\Z)",
            text, re.DOTALL
        )
        return m.group(1).strip() if m else ""

    def _skills(self, text: str, *, required: bool) -> list[str]:
        heading_sets = (
            ["Required Skills", "Required Qualifications", "Must Have", "Requirements"],
            ["Preferred Skills", "Nice to Have", "Preferred Qualifications", "Assets"],
        )
        headings = heading_sets[0] if required else heading_sets[1]
        block = self._section(text, *headings)
        return self._extract_list(block) if block else []

    def _all_skills(self, text: str) -> list[str]:
        """Grab all skill-like tokens from the full posting."""
        # Common tech / skill terms — broad sweep
        tokens = re.findall(
            r"\b([A-Za-z][A-Za-z0-9+#.\-]{1,30})\b", text
        )
        stop = {
            "the","and","or","in","of","to","a","an","for","with","at","by",
            "from","is","are","be","we","you","our","will","have","has","this",
            "that","on","as","it","not","but","all","can","may","any","your",
            "their","they","who","what","how","when","where","each","been",
            "would","should","could","must","than","then","into","more","also",
            "other","both","most","some","such","well","per","etc","via","us",
            "its","was","were","had","did","do","does","if","up","about","after",
            "before","between","during","through","over","under","above",
            "company", "role", "team", "work", "experience", "skills",
            "ability", "strong", "excellent", "proven", "using", "use",
            "including", "within", "across", "ensure", "build", "help",
            "make", "take", "give", "get", "set", "new", "good", "best",
            "high", "large", "small", "based", "related", "focused",
        }
        seen, out = set(), []
        for t in tokens:
            tl = t.lower()
            if tl not in stop and len(t) > 1 and t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def _years_exp(self, text: str) -> int | None:
        m = re.search(r"(\d+)\+?\s*(?:or more\s*)?years?\s+(?:of\s+)?experience", text, re.IGNORECASE)
        return int(m.group(1)) if m else None

    def _degree_level(self, text: str) -> int:
        lower = text.lower()
        best = 0
        for kw, level in self._DEGREE_LEVELS.items():
            if kw in lower:
                best = max(best, level)
        return best

    def _degree_field(self, text: str) -> list[str]:
        m = re.findall(
            r"(?:degree|diploma|b\.?sc|m\.?sc|b\.?eng|m\.?eng)\s+in\s+([A-Za-z ]{3,40})",
            text, re.IGNORECASE
        )
        return [f.strip() for f in m]

    def _certs(self, text: str) -> list[str]:
        block = self._section(
            text,
            "Certifications", "Licenses", "Credentials",
            "Required Certifications", "Professional Certifications",
        )
        if block:
            return self._extract_list(block)
        # Fallback: scan for known cert patterns anywhere
        found = re.findall(
            r"\b((?:PMP|AWS|GCP|Azure|CPA|CFA|CKA|CISSP|CCNA|CCNP|Six Sigma|"
            r"Scrum|Agile|LEED|P\.?Eng|ISO \d+)[^\n,;]{0,40})\b",
            text, re.IGNORECASE
        )
        return list(dict.fromkeys(f.strip() for f in found))

    def _description(self, text: str) -> str:
        """Best paragraph for semantic comparison — responsibilities or overview."""
        block = self._section(
            text,
            "Responsibilities", "About the Role", "Role Overview",
            "What You Will Do", "Job Description", "Overview",
        )
        return block if block else text[:2000]

    def _extract_list(self, text: str) -> list[str]:
        items = re.split(r"(?m)(?:^|\n)\s*[•●▪\-*]\s*", text)
        out = []
        for item in items:
            item = re.sub(r"\s+", " ", item).strip(" .")
            if item and len(item) > 2:
                out.append(item)
        return out

# ---------------------------------------------------------------------------
# Individual scorers
# ---------------------------------------------------------------------------

_DEGREE_LEVEL_MAP = {
    "phd": 5, "ph.d": 5, "doctorate": 5,
    "master": 4, "m.sc": 4, "m.eng": 4, "mba": 4,
    "bachelor": 3, "b.sc": 3, "b.eng": 3, "b.a": 3,
    "diploma": 2, "college": 2,
    "high school": 1, "secondary": 1,
}


def _candidate_degree_level(education: list[dict]) -> int:
    best = 0
    for edu in education:
        degree = (edu.get("degree") or "").lower()
        for kw, lvl in _DEGREE_LEVEL_MAP.items():
            if kw in degree:
                best = max(best, lvl)
    return best


def score_skills(resume: dict, job: dict, sem: HDCSearch) -> SectionScore:
    # Prefer cleaned derived skills, fall back to raw skills list
    resume_skills = (
        (resume.get("derived") or {}).get("skills_normalized")
        or resume.get("skills")
        or []
    )
    resume_skills = [s for s in resume_skills if s and len(s.strip()) > 1]

    required  = job.get("required_skills") or []
    preferred = job.get("preferred_skills") or []
    all_job   = job.get("all_skills") or []
    job_skills = required or (preferred + all_job[:20])

    if not resume_skills or not job_skills:
        return SectionScore(0.0, 0.0, WEIGHTS["skills"],
                            notes=["No skills data available"])

    # --- Pure semantic: for each job skill find best resume skill match ---
    sims = sem.bulk_similarities(job_skills, resume_skills)

    MATCH_THRESHOLD  = 0.50   # cosine >= this = matched
    STRONG_THRESHOLD = 0.65   # cosine >= this = strong match

    matched, missing = [], []
    for skill, score in zip(job_skills, sims):
        if score >= MATCH_THRESHOLD:
            matched.append(skill)
        elif skill in (required or []):   # only flag required as missing
            missing.append(skill)

    strong_ratio = sum(s >= STRONG_THRESHOLD for s in sims) / len(sims)
    weak_ratio   = sum(MATCH_THRESHOLD <= s < STRONG_THRESHOLD for s in sims) / len(sims)
    raw          = min(strong_ratio + 0.5 * weak_ratio, 1.0)

    notes = [
        f"Semantic match: {len(matched)}/{len(job_skills)} job skills",
        f"Strong matches (≥{STRONG_THRESHOLD}): {sum(s >= STRONG_THRESHOLD for s in sims)}",
    ]

    return SectionScore(
        raw=raw,
        weighted=raw * WEIGHTS["skills"],
        max_points=WEIGHTS["skills"],
        notes=notes,
        matched=matched[:10],
        missing=missing[:10],
    )


def score_experience(resume: dict, job: dict, sem: HDCSearch) -> SectionScore:
    work   = resume.get("work_experience") or []
    notes, matched, missing = [], [], []

    # --- Years (from derived, already computed correctly) ---
    min_years     = job.get("min_years_exp")
    derived       = resume.get("derived") or {}
    candidate_yrs = derived.get("years_experience")
    years_score   = 1.0

    if min_years:
        if candidate_yrs:
            years_score = min(candidate_yrs / min_years, 1.0)
            notes.append(f"{candidate_yrs} yrs vs {min_years} required")
            if candidate_yrs < min_years:
                missing.append(f"{min_years - candidate_yrs:.0f}+ more years needed")
        else:
            years_score = 0.4
            notes.append("Years of experience not parseable from resume")

    # --- Semantic relevance of each role to job description ---
    job_desc = job.get("description") or job.get("full_text") or ""
    if not job_desc.strip() or not work:
        return SectionScore(
            raw=0.5 * years_score, 
            weighted=0.5 * years_score * WEIGHTS["experience"],
            max_points=WEIGHTS["experience"],
            notes=notes + ["No work experience or job description to compare"],
        )

    # Build one text blob per role (title + highlights)
    role_texts = []
    for role in work:
        parts = [role.get("title") or "", role.get("company") or ""]
        parts += role.get("highlights") or []
        role_texts.append(" ".join(p for p in parts if p))

    role_sims  = sem.bulk_similarities(role_texts, [job_desc[:1500]])
    best_idx   = int(np.argmax(role_sims))
    relevance  = role_sims[best_idx]
    avg_rel    = sum(role_sims) / len(role_sims)
    # Best role weighted more than average
    rel_score  = 0.7 * relevance + 0.3 * avg_rel

    best_role = work[best_idx]
    matched.append(f"{best_role.get('title')} @ {best_role.get('company')}")
    notes.append(f"Best role relevance: {relevance:.0%}, avg: {avg_rel:.0%}")

    raw = 0.4 * years_score + 0.6 * rel_score
    return SectionScore(
        raw=min(raw, 1.0),
        weighted=min(raw, 1.0) * WEIGHTS["experience"],
        max_points=WEIGHTS["experience"],
        notes=notes, matched=matched, missing=missing,
    )

def score_education(
    resume: dict, job: dict, _sem: HDCSearch
) -> SectionScore:
    notes, matched, missing = [], [], []

    candidate_lvl = _candidate_degree_level(resume.get("education") or [])
    required_lvl  = job.get("degree_required") or 0
    degree_fields = job.get("degree_field") or []

    # Level score
    if required_lvl == 0:
        lvl_score = 1.0
        notes.append("No specific degree level required")
    elif candidate_lvl >= required_lvl:
        lvl_score = 1.0
        matched.append(f"Degree level met (candidate={candidate_lvl}, required={required_lvl})")
    else:
        lvl_score = candidate_lvl / required_lvl if required_lvl else 1.0
        missing.append(f"Degree level gap (candidate={candidate_lvl}, required={required_lvl})")

    # Field match
    field_score = 1.0
    if degree_fields:
        candidate_fields = " ".join(
            (e.get("field_of_study") or "") + " " + (e.get("degree") or "")
            for e in (resume.get("education") or [])
        ).lower()
        hits = [f for f in degree_fields if f.lower() in candidate_fields]
        field_score = len(hits) / len(degree_fields)
        if hits:
            matched.append(f"Field match: {', '.join(hits)}")
        else:
            missing.append(f"Preferred fields: {', '.join(degree_fields)}")

    raw = 0.6 * lvl_score + 0.4 * field_score
    return SectionScore(
        raw=min(raw, 1.0),
        weighted=min(raw, 1.0) * WEIGHTS["education"],
        max_points=WEIGHTS["education"],
        notes=notes, matched=matched, missing=missing,
    )


def score_summary(
    resume: dict, job: dict, sem: HDCSearch
) -> SectionScore:
    candidate_summary = resume.get("summary") or ""
    # Enrich with highlights from top work role
    work = resume.get("work_experience") or []
    if work:
        top = work[0]
        candidate_summary += " " + " ".join(top.get("highlights") or [])

    job_desc = job.get("description") or job.get("full_text") or ""

    if not candidate_summary.strip() or not job_desc.strip():
        return SectionScore(0.5, 0.5 * WEIGHTS["summary"], WEIGHTS["summary"],
                            notes=["Missing summary or job description"])

    sim = sem.similarity(candidate_summary.strip(), job_desc[:1500])
    notes = [f"Profile-to-JD semantic similarity: {sim:.0%}"]

    return SectionScore(
        raw=sim,
        weighted=sim * WEIGHTS["summary"],
        max_points=WEIGHTS["summary"],
        notes=notes,
    )


def score_certifications(
    resume: dict, job: dict, _sem: HDCSearch
) -> SectionScore:
    resume_certs = [
        (c.get("name") or "").lower()
        for c in (resume.get("certifications") or [])
        if c.get("name")
    ]
    job_certs = [c.lower() for c in (job.get("certifications") or [])]

    if not job_certs:
        # No certs required → full marks; note any the candidate has
        note = (f"No certifications required by job. "
                f"Candidate holds: {len(resume_certs)} cert(s)") if resume_certs else \
               "No certifications required."
        return SectionScore(1.0, WEIGHTS["certifications"], WEIGHTS["certifications"],
                            notes=[note])

    if not resume_certs:
        return SectionScore(0.0, 0.0, WEIGHTS["certifications"],
                            notes=["Job requires certifications but none found on resume"],
                            missing=job_certs)

    matched, missing = [], []
    for jc in job_certs:
        jc_tokens = set(re.findall(r"\b[a-z0-9]{2,}\b", jc))
        hit = any(
            jc_tokens & set(re.findall(r"\b[a-z0-9]{2,}\b", rc))
            for rc in resume_certs
        )
        (matched if hit else missing).append(jc)

    raw = len(matched) / len(job_certs)
    return SectionScore(
        raw=raw,
        weighted=raw * WEIGHTS["certifications"],
        max_points=WEIGHTS["certifications"],
        matched=matched, missing=missing,
    )


def score_projects(
    resume: dict, job: dict, sem: HDCSearch
) -> SectionScore:
    projects = resume.get("projects") or []
    job_desc = job.get("description") or job.get("full_text") or ""

    if not projects or not job_desc:
        return SectionScore(0.3, 0.3 * WEIGHTS["projects"], WEIGHTS["projects"],
                            notes=["No projects or job description to compare"])

    proj_texts = [
        " ".join(filter(None, [p.get("title"), p.get("description")]))
        for p in projects
    ]
    sims = [sem.similarity(pt, job_desc[:1000]) for pt in proj_texts]
    best_sim = max(sims)
    best_proj = projects[sims.index(best_sim)]

    notes = [f"Most relevant project: '{best_proj.get('title')}' ({best_sim:.0%} match)"]
    avg_sim = sum(sims) / len(sims)
    raw = 0.6 * best_sim + 0.4 * avg_sim

    return SectionScore(
        raw=min(raw, 1.0),
        weighted=min(raw, 1.0) * WEIGHTS["projects"],
        max_points=WEIGHTS["projects"],
        notes=notes,
        matched=[best_proj.get("title") or ""],
    )


# ---------------------------------------------------------------------------
# Grade + recommendation
# ---------------------------------------------------------------------------

def _grade(score: float) -> str:
    if score >= 85: return "A"
    if score >= 70: return "B"
    if score >= 55: return "C"
    if score >= 40: return "D"
    return "F"


def _recommendation(score: float, sections: dict[str, SectionScore]) -> str:
    if score >= 85:
        return "Strong match. Recommend advancing to interview."
    if score >= 70:
        return "Good match. A few gaps worth discussing in interview."
    if score >= 55:
        return "Partial match. Candidate meets some requirements but has notable gaps."
    if score >= 40:
        return "Weak match. Significant gaps in key areas."
    return "Poor match. Candidate profile does not align well with this role."


# ---------------------------------------------------------------------------
# Main matcher
# ---------------------------------------------------------------------------

class ResumeMatcher:
    def __init__(self) -> None:
        print("Loading semantic model (first run downloads ~80MB)...", flush=True)
        self._sem    = HDCSearch()
        self._reader = TextReader()
        self._jparser = JobParser()
        print("Model ready.", flush=True)

    def match(
        self,
        resume: dict[str, Any],
        job_path: Path,
    ) -> MatchReport:
        raw_text   = self._reader.read(job_path)
        clean_text = self._reader.clean(raw_text)
        job        = self._jparser.parse(clean_text)

        # Pre-encode everything once before scoring
        self._sem.warm_batch(resume)

        desc = job.get("description") or job.get("full_text") or ""
        if desc:
            self._sem.encode(desc[:1500])   # cache job description

        job_skills = job.get("required_skills") or job.get("all_skills") or []
        if job_skills:
            self._sem.encode_batch(job_skills)  # cache job skills

        scorers = {
            "skills":         score_skills,
            "experience":     score_experience,
            "education":      score_education,
            "summary":        score_summary,
            "certifications": score_certifications,
            "projects":       score_projects,
        }

        sections: dict[str, SectionScore] = {}
        for key, fn in scorers.items():
            print(f"  Scoring {key}...", flush=True)
            sections[key] = fn(resume, job, self._sem)

        total = sum(s.weighted for s in sections.values())

        # Top strengths: sections where raw >= 0.7
        strengths = [
            k for k, s in sections.items() if s.raw >= 0.70
        ]
        # Gaps: sections where raw < 0.5 and weight is significant
        gaps = [
            f"{k} ({WEIGHTS[k]}pts)" for k, s in sections.items()
            if s.raw < 0.50 and WEIGHTS[k] >= 10
        ]
        # Add specific missing items
        for s in sections.values():
            gaps += s.missing[:3]

        return MatchReport(
            candidate_name=resume.get("name") or "Candidate",
            total_score=round(total, 1),
            grade=_grade(total),
            sections=sections,
            top_strengths=strengths,
            gaps=gaps[:8],
            recommendation=_recommendation(total, sections),
        )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def print_report(report: MatchReport) -> None:
    W = 60
    print("\n" + "=" * W)
    print(f"  RESUME MATCH REPORT — {report.candidate_name}")
    print("=" * W)
    print(f"  Overall Score : {report.total_score:.1f} / 100  (Grade: {report.grade})")
    print(f"  Recommendation: {report.recommendation}")
    print("-" * W)

    for key, sec in report.sections.items():
        bar_filled = int(sec.raw * 20)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        print(f"\n  {key.upper():<16} {bar}  {sec.weighted:.1f}/{sec.max_points}pts")
        for note in sec.notes:
            print(f"    ℹ  {note}")
        if sec.matched:
            print(f"    ✓  {', '.join(sec.matched[:5])}")
        if sec.missing:
            print(f"    ✗  {', '.join(sec.missing[:5])}")

    print("\n" + "-" * W)
    if report.top_strengths:
        print(f"  Strengths : {', '.join(report.top_strengths)}")
    if report.gaps:
        print(f"  Gaps      : {', '.join(report.gaps[:5])}")
    print("=" * W + "\n")


def report_to_dict(report: MatchReport) -> dict[str, Any]:
    return {
        "candidate_name": report.candidate_name,
        "total_score":    report.total_score,
        "grade":          report.grade,
        "recommendation": report.recommendation,
        "top_strengths":  report.top_strengths,
        "gaps":           report.gaps,
        "sections": {
            k: {
                "score":      round(v.weighted, 2),
                "max_points": v.max_points,
                "raw":        round(v.raw, 3),
                "notes":      v.notes,
                "matched":    v.matched,
                "missing":    v.missing,
            }
            for k, v in report.sections.items()
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score a resume JSON against a job posting PDF/text."
    )
    ap.add_argument("resume_json", type=Path, help="Parsed resume JSON file")
    ap.add_argument("job_posting", type=Path, help="Job posting PDF or text file")
    ap.add_argument("--output", type=Path, help="Save JSON report to this path")
    ap.add_argument("--detail", action="store_true", help="Print full report to console")
    args = ap.parse_args()

    if not args.resume_json.is_file():
        ap.error(f"Resume JSON not found: {args.resume_json}")
    if not args.job_posting.is_file():
        ap.error(f"Job posting not found: {args.job_posting}")

    resume = json.loads(args.resume_json.read_text(encoding="utf-8-sig", errors="replace"))

    matcher = ResumeMatcher()
    report  = matcher.match(resume, args.job_posting)

    print_report(report)

    if args.output:
        args.output.write_text(
            json.dumps(report_to_dict(report), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Report saved to: {args.output}")


# ---------------------------------------------------------------------------
# Importable API
# ---------------------------------------------------------------------------

def match_resume(
    resume: dict[str, Any],
    job_path: str | Path,
) -> dict[str, Any]:
    """Importable one-call interface. Returns the report as a dict."""
    matcher = ResumeMatcher()
    report  = matcher.match(resume, Path(job_path))
    return report_to_dict(report)


if __name__ == "__main__":
    main()