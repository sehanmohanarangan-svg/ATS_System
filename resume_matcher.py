"""
Hybrid Resume-to-Job Match Scorer  (optimized + tailored)
=========================================================
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

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

import datetime

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

MIN_TOKEN_LEN = 3
STOPWORDS = {"and", "or", "in", "of", "the", "for", "with"}

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
# PDF / text reader
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
            ("\u2022", "•"), ("\u00a0", " "), ("\u2013", "-"), ("\u2014", "-"),
            ("\u2019", "'"), ("\u2018", "'"),
        ]:
            text = text.replace(bad, good)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


# ---------------------------------------------------------------------------
# Job posting parser
# ---------------------------------------------------------------------------

class JobParser:
    """Heuristically extracts key fields from a job posting."""

    _DEGREE_LEVELS = {
        "phd": 5, "ph.d": 5, "doctorate": 5,
        "master": 4, "m.sc": 4, "m.eng": 4, "mba": 4, "m.s": 4,
        "bachelor": 3, "b.sc": 3, "b.eng": 3, "b.a": 3, "b.tech": 3,
        "undergraduate": 3,
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

    # REPLACE _job_title entirely:
    def _job_title(self, text: str) -> str | None:
        NOISE = re.compile(
            r"^(about|overview|summary|location|remote|hybrid|"
            r"company|who we are|the role|our team)\b",
            re.IGNORECASE,
        )
        TITLE_SIGNALS = re.compile(
            r"\b(engineer|developer|manager|analyst|designer|"
            r"architect|lead|director|scientist|specialist|"
            r"coordinator|consultant|officer)\b",
            re.IGNORECASE,
        )
        
        for line in text.splitlines()[:20]:   # only scan top 20 lines
            line = line.strip()
            if 5 < len(line) < 80 and not NOISE.match(line) and TITLE_SIGNALS.search(line):
                return line
        return None

    def _section(self, text: str, *headings: str) -> str:
        pattern = "|".join(re.escape(h) for h in headings)
        m = re.search(
            rf"(?im)^(?:{pattern})[:\s]*\n(.*?)(?=\n[A-Z][A-Za-z ]+[:\n]|\Z)",
            text, re.DOTALL,
        )
        return m.group(1).strip() if m else ""

    def _skills(self, text: str, *, required: bool) -> list[str]:
        headings = (
            ["Required Skills", "Required Qualifications", "Must Have", "Requirements"]
            if required else
            ["Preferred Skills", "Nice to Have", "Preferred Qualifications", "Assets"]
        )
        block = self._section(text, *headings)
        if not block:
            return []
        
        sentences = self._extract_list(block)
        
        # Pull out actual skill tokens from each sentence
        SKILL_PATTERN = re.compile(
            r"\b(JavaScript|Node\.js|Angular[\w\s]*?JS|PostgreSQL|MySQL|MongoDB|"
            r"AWS|GCP|Azure|Docker|Kubernetes|CI[/\-]CD|GitHub|GitLab|"
            r"Python|Java|Go|Rust|TypeScript|React|Vue|Django|FastAPI|"
            r"ECS|EKS|Fargate|Lambda|S3|RDS|Redis|Kafka|"
            r"TLS|HTTPS|REST|GraphQL|gRPC|OAuth|JWT|"
            r"Linux|Terraform|Ansible|Nginx|[A-Z]{2,}(?:\s[A-Z][a-z]+)?)\b",
            re.IGNORECASE,
        )
        
        skills = []
        seen = set()
        for sentence in sentences:
            found = SKILL_PATTERN.findall(sentence)
            if found:
                for f in found:
                    fl = f.lower()
                    if fl not in seen:
                        seen.add(fl)
                        skills.append(f)
            else:
                # Fallback: keep the sentence for semantic scoring
                if sentence not in seen:
                    seen.add(sentence)
                    skills.append(sentence)
        
        return skills

    def _all_skills(self, text: str) -> list[str]:
        # In _all_skills, change the token pattern to:
        tokens = re.findall(r"\b([A-Za-z][A-Za-z0-9+#./\-]{1,30})\b", text)
        stop = {
            "the", "and", "or", "in", "of", "to", "a", "an", "for", "with", "at", "by",
            "from", "is", "are", "be", "we", "you", "our", "will", "have", "has", "this",
            "that", "on", "as", "it", "not", "but", "all", "can", "may", "any", "your",
            "their", "they", "who", "what", "how", "when", "where", "each", "been",
            "would", "should", "could", "must", "than", "then", "into", "more", "also",
            "other", "both", "most", "some", "such", "well", "per", "etc", "via", "us",
            "its", "was", "were", "had", "did", "do", "does", "if", "up", "about", "after",
            "before", "between", "during", "through", "over", "under", "above",
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
        m = re.search(
            r"(\d+)\+?\s*(?:or more\s*)?years?\s+(?:of\s+)?(?:\w+\s+)?experience",
            text, re.IGNORECASE,
        )
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
            r"(?:degree|diploma|b\.?sc|m\.?sc|b\.?eng|m\.?eng|b\.?tech)\s+in\s+([A-Za-z ]{3,40})",
            text, re.IGNORECASE,
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
        found = re.findall(
            r"\b((?:PMP|AWS|GCP|Azure|CPA|CFA|CKA|CISSP|CCNA|CCNP|Six Sigma|"
            r"Scrum|Agile|LEED|P\.?Eng|ISO \d+)[^\n,;]{0,40})\b",
            text, re.IGNORECASE,
        )
        return list(dict.fromkeys(f.strip() for f in found))

    def _description(self, text: str) -> str:
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
# Resume helpers tailored to the provided JSON shape
# ---------------------------------------------------------------------------

_DEGREE_LEVEL_MAP = {
    "phd": 5, "ph.d": 5, "doctorate": 5,
    "master": 4, "m.sc": 4, "m.eng": 4, "mba": 4, "m.s": 4,
    "bachelor": 3, "b.sc": 3, "b.eng": 3, "b.a": 3, "b.tech": 3,
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


def _normalize_skill(s: str) -> str:
    """Clean noisy skill strings that appear in the sample resume."""
    s = s.strip().lower()
    # Drop pure filler fragments
    if len(s) < 2 or s in {"and", "in", "of", "the", "options in", "courses coursework"}:
        return ""
    # Remove leading/trailing noise words
    s = re.sub(r"^(and|including|using|with)\s+", "", s)
    s = re.sub(r"\s+(and|including)$", "", s)
    return s.strip()


def _get_resume_skills(resume: dict) -> list[str]:
    """Prefer derived.skills_normalized; otherwise clean the raw skills list."""
    derived = resume.get("derived") or {}
    skills = derived.get("skills_normalized") or resume.get("skills") or []

    cleaned = []
    seen = set()
    for raw in skills:
        s = _normalize_skill(str(raw))
        if s and s not in seen:
            seen.add(s)
            cleaned.append(s)
    return cleaned


def _estimate_years(resume: dict) -> float | None:
    """
    Try derived.years_experience first.
    Fall back to a rough count of distinct work roles (sample has placeholder years).
    """
    derived = resume.get("derived") or {}
    yrs = derived.get("years_experience")
    if yrs is not None:
        try:
            return float(yrs)
        except (TypeError, ValueError):
            pass

    # Heuristic: each distinct role ≈ 1 year if dates are placeholders
    work = resume.get("work_experience") or []
    if not work:
        return None

    # If real years appear somewhere, try to parse them
    year_nums = []
    for role in work:
        for key in ("start_date", "end_date"):
            val = str(role.get(key) or "")
            m = re.search(r"(20\d{2}|19\d{2})", val)
            if m:
                year_nums.append(int(m.group(1)))

    if len(year_nums) >= 2:
        return float(max(year_nums) - min(year_nums) + 1)

    # Last resort: number of roles as a weak proxy (capped)
    return float(len(work)) * 1.5 

def _recency_weight(role: dict, half_life_years: float = 7.0) -> float:
    """Roles decay in relevance with a configurable half-life."""
    current_year = datetime.date.today().year
    for key in ("end_date", "start_date"):
        val = str(role.get(key) or "")
        m = re.search(r"(20\d{2}|19\d{2})", val)
        if m:
            age = current_year - int(m.group(1))
            return 0.5 ** (age / half_life_years)
    return 0.75  # unknown date: mild penalty

def _synthesize_profile(resume: dict) -> str:
    """Build a profile text when summary is missing."""
    parts: list[str] = []

    if resume.get("summary"):
        parts.append(resume["summary"])

    # Latest role highlights
    work = resume.get("work_experience") or []
    if work:
        top = work[0]
        parts.append(top.get("title") or "")
        parts += top.get("highlights") or []

    # Leadership highlights (valuable signal for this resume)
    for lead in (resume.get("leadership") or [])[:3]:
        parts.append(lead.get("role") or "")
        parts += (lead.get("highlights") or [])[:2]

    # Awards
    awards = resume.get("awards") or []
    if awards:
        parts.append(" ".join(str(a) for a in awards[:2]))

    return " ".join(p for p in parts if p).strip()


def _extract_projects_from_skills(resume: dict) -> list[dict]:
    """When projects[] is empty, try to recover project-like items from the skills blob."""
    if resume.get("projects"):
        return resume["projects"]

    projects = []
    for raw in resume.get("skills") or []:
        raw = str(raw)
        if "project" in raw.lower() or "simulated" in raw.lower() or "researched" in raw.lower():
            # crude split
            for piece in re.split(r"[;.]", raw):
                piece = piece.strip()
                if len(piece) > 15:
                    projects.append({"title": piece[:60], "description": piece})
    return projects


# ---------------------------------------------------------------------------
# Individual scorers
# ---------------------------------------------------------------------------

def score_skills(resume: dict, job: dict, sem: HDCSearch) -> SectionScore:
    resume_skills = _get_resume_skills(resume)

    required  = job.get("required_skills") or []
    preferred = job.get("preferred_skills") or []
    all_job   = job.get("all_skills") or []

    # Prefer explicit required skills; otherwise take preferred + top of all_skills
    required_weight  = 1.0
    preferred_weight = 0.6

    job_skills   = []
    job_weights  = []

    if required:
        job_skills  += required
        job_weights += [required_weight] * len(required)
    if preferred:
        job_skills  += preferred
        job_weights += [preferred_weight] * len(preferred)
    if not job_skills:
        job_skills  = all_job[:25]
        job_weights = [0.4] * len(all_job[:25])

    if not resume_skills or not job_skills:
        return SectionScore(
            0.0, 0.0, WEIGHTS["skills"],
            notes=["No skills data available"],
        )

    # Semantic: for each job skill find best resume skill match
    sims = sem.bulk_similarities(job_skills, resume_skills)

    MATCH_THRESHOLD  = 0.48
    STRONG_THRESHOLD = 0.62

    # REPLACE the matched/missing loop + raw calculation:
    # Replace the matched/missing loop + raw calculation:
    matched, missing = [], []
    weighted_strong, weighted_total = 0.0, 0.0
    count_strong, count_moderate = 0, 0          # ← add these

    for skill, score, w in zip(job_skills, sims, job_weights):
        weighted_total += w
        if score >= STRONG_THRESHOLD:
            weighted_strong += w
            matched.append(skill)
            count_strong += 1                    # ← track it
        elif score >= MATCH_THRESHOLD:
            weighted_strong += w * 0.55
            matched.append(skill)
            count_moderate += 1                  # ← track it
        elif required and skill in required:
            missing.append(skill)

    raw = min(weighted_strong / max(weighted_total, 1.0), 1.0)
    notes = [
        f"Semantic match: {len(matched)}/{len(job_skills)} job skills",
        f"Strong (≥{STRONG_THRESHOLD:.2f}): {count_strong}, moderate: {count_moderate}",  # ← fixed
    ]

    return SectionScore(
        raw=raw,
        weighted=raw * WEIGHTS["skills"],
        max_points=WEIGHTS["skills"],
        notes=notes,
        matched=matched[:12],
        missing=missing[:10],
    )


def score_experience(resume: dict, job: dict, sem: HDCSearch) -> SectionScore:
    work = resume.get("work_experience") or []
    notes, matched, missing = [], [], []

    # --- Years ---
    min_years = job.get("min_years_exp")
    candidate_yrs = _estimate_years(resume)
    years_score = 1.0

    if min_years:
        if candidate_yrs is not None:
            years_score = min(candidate_yrs / min_years, 1.0)
            notes.append(f"{candidate_yrs:.1f} yrs estimated vs {min_years} required")
            if candidate_yrs < min_years:
                missing.append(f"{min_years - candidate_yrs:.0f}+ more years needed")
        else:
            years_score = 0.35
            notes.append("Years of experience not parseable")

    # --- Semantic relevance of each role ---
    job_desc = job.get("description") or job.get("full_text") or ""
    if not job_desc.strip() or not work:
        raw = 0.5 * years_score
        return SectionScore(
            raw=raw,
            weighted=raw * WEIGHTS["experience"],
            max_points=WEIGHTS["experience"],
            notes=notes + ["No work experience or job description to compare"],
            matched=matched,
            missing=missing,
        )

    role_texts = []
    for role in work:
        parts = [role.get("title") or "", role.get("company") or ""]
        parts += role.get("highlights") or []
        role_texts.append(" ".join(p for p in parts if p))

    role_sims = sem.bulk_similarities(role_texts, [job_desc[:1500]])
    best_idx  = int(np.argmax(role_sims))
    relevance = role_sims[best_idx]
    weights = [_recency_weight(r) for r in work]
    weighted_sum = sum(s * w for s, w in zip(role_sims, weights))
    avg_rel = weighted_sum / max(sum(weights), 1e-9)
    rel_score = 0.7 * relevance + 0.3 * avg_rel

    best_role = work[best_idx]
    matched.append(f"{best_role.get('title')} @ {best_role.get('company')}")
    notes.append(f"Best role relevance: {relevance:.0%}, avg: {avg_rel:.0%}")

    # Small leadership bonus (this resume has strong leadership signal)
    leadership = resume.get("leadership") or []
    if leadership and relevance > 0.4:
        rel_score = min(rel_score + 0.05, 1.0)
        notes.append(f"+ leadership bonus ({len(leadership)} roles)")

    raw = 0.4 * years_score + 0.6 * rel_score
    return SectionScore(
        raw=min(raw, 1.0),
        weighted=min(raw, 1.0) * WEIGHTS["experience"],
        max_points=WEIGHTS["experience"],
        notes=notes,
        matched=matched,
        missing=missing,
    )


def score_education(resume: dict, job: dict, _sem: HDCSearch) -> SectionScore:
    notes, matched, missing = [], [], []

    candidate_lvl = _candidate_degree_level(resume.get("education") or [])
    required_lvl  = job.get("degree_required") or 0
    degree_fields = job.get("degree_field") or []

    # Level
    if required_lvl == 0:
        lvl_score = 1.0
        notes.append("No specific degree level required")
    elif candidate_lvl >= required_lvl:
        lvl_score = 1.0
        matched.append(f"Degree level met (cand={candidate_lvl}, req={required_lvl})")
    else:
        lvl_score = candidate_lvl / required_lvl if required_lvl else 1.0
        missing.append(f"Degree level gap (cand={candidate_lvl}, req={required_lvl})")

    # Field
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

    raw = 0.65 * lvl_score + 0.35 * field_score
    return SectionScore(
        raw=min(raw, 1.0),
        weighted=min(raw, 1.0) * WEIGHTS["education"],
        max_points=WEIGHTS["education"],
        notes=notes,
        matched=matched,
        missing=missing,
    )


def score_summary(resume: dict, job: dict, sem: HDCSearch) -> SectionScore:
    candidate_summary = _synthesize_profile(resume)
    job_desc = job.get("description") or job.get("full_text") or ""

    if not candidate_summary or not job_desc.strip():
        return SectionScore(
            0.4, 0.4 * WEIGHTS["summary"], WEIGHTS["summary"],
            notes=["Missing usable profile text or job description"],
        )

    sim = sem.similarity(candidate_summary, job_desc[:1500])
    notes = [f"Profile-to-JD semantic similarity: {sim:.0%}"]
    if not resume.get("summary"):
        notes.append("Used synthesized profile (summary was null)")

    return SectionScore(
        raw=sim,
        weighted=sim * WEIGHTS["summary"],
        max_points=WEIGHTS["summary"],
        notes=notes,
    )


def score_certifications(resume: dict, job: dict, _sem: HDCSearch) -> SectionScore:
    resume_certs = [
        (c.get("name") or "").lower()
        for c in (resume.get("certifications") or [])
        if c.get("name")
    ]
    # Also check awards for cert-like strings
    for a in resume.get("awards") or []:
        resume_certs.append(str(a).lower())

    job_certs = [c.lower() for c in (job.get("certifications") or [])]

    if not job_certs:
        note = (
            f"No certifications required. Candidate holds: {len(resume_certs)} item(s)"
            if resume_certs else "No certifications required."
        )
        return SectionScore(
            1.0, WEIGHTS["certifications"], WEIGHTS["certifications"],
            notes=[note],
        )

    if not resume_certs:
        return SectionScore(
            0.7, 0.7, WEIGHTS["certifications"],
            notes=["Job requires certifications but none found on resume"],
            missing=job_certs,
        )
    
    def _cert_tokens(s: str) -> set[str]:
        return {
            t for t in re.findall(r"\b[a-z0-9]{2,}\b", s)
            if len(t) >= MIN_TOKEN_LEN and t not in STOPWORDS
        }
    matched, missing = [], []
    for jc in job_certs:
        jc_tokens = _cert_tokens(jc)
        hit = any(
            len(jc_tokens & _cert_tokens(rc)) >= max(1, len(jc_tokens) // 3)
            for rc in resume_certs
        )
        (matched if hit else missing).append(jc)

    raw = len(matched) / len(job_certs)
    return SectionScore(
        raw=raw,
        weighted=raw * WEIGHTS["certifications"],
        max_points=WEIGHTS["certifications"],
        matched=matched,
        missing=missing,
    )


def score_projects(resume: dict, job: dict, sem: HDCSearch) -> SectionScore:
    projects = _extract_projects_from_skills(resume)
    job_desc = job.get("description") or job.get("full_text") or ""

    if not projects or not job_desc:
        return SectionScore(
            0.25, 0.25 * WEIGHTS["projects"], WEIGHTS["projects"],
            notes=["No projects or job description to compare"],
        )

    proj_texts = [
        " ".join(filter(None, [p.get("title"), p.get("description")]))
        for p in projects
    ]
    sims = sem.bulk_similarities(proj_texts, [job_desc[:1000]])
    best_sim = max(sims) if sims else 0.0
    best_idx = int(np.argmax(sims)) if sims else 0
    best_proj = projects[best_idx]

    notes = [f"Most relevant: '{best_proj.get('title', '')[:50]}' ({best_sim:.0%})"]
    avg_sim = sum(sims) / len(sims) if sims else 0.0
    raw = 0.65 * best_sim + 0.35 * avg_sim

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
        print("Loading semantic model (first run downloads ~80 MB)...", flush=True)
        self._sem     = HDCSearch()
        self._reader  = TextReader()
        self._jparser = JobParser()
        print("Model ready.", flush=True)

    def match(self, resume: dict[str, Any], job_path: Path) -> MatchReport:
        raw_text   = self._reader.read(job_path)
        clean_text = self._reader.clean(raw_text)
        job        = self._jparser.parse(clean_text)

        # ---- one-time batch encoding ----
        self._sem.warm_resume(resume)

        desc = job.get("description") or job.get("full_text") or ""
        if desc:
            self._sem.encode(desc[:1500])

        job_skills = (
            job.get("required_skills")
            or job.get("preferred_skills")
            or job.get("all_skills")
            or []
        )
        if job_skills:
            self._sem.encode_batch(job_skills)

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

        strengths = [k for k, s in sections.items() if s.raw >= 0.70]
        gaps = [
            f"{k} ({WEIGHTS[k]}pts)"
            for k, s in sections.items()
            if s.raw < 0.50 and WEIGHTS[k] >= 10
        ]
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

    resume = json.loads(
        args.resume_json.read_text(encoding="utf-8-sig", errors="replace")
    )

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