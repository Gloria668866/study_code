from __future__ import annotations

import hashlib
import io
import json
import os
import re
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pypdf import PdfReader

try:
    import redis
except Exception:  # pragma: no cover
    redis = None


APP_NAME = "AI赋能的智能简历分析系统"
CACHE_TTL_SECONDS = 60 * 60 * 24

app = FastAPI(title=APP_NAME, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResumeSummary(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    job_intent: Optional[str] = None
    expected_salary: Optional[str] = None
    work_years: Optional[str] = None
    education: Optional[str] = None
    projects: List[str] = Field(default_factory=list)


class MatchResult(BaseModel):
    keyword_score: float
    experience_score: float
    overall_score: float
    matched_keywords: List[str] = Field(default_factory=list)
    missing_keywords: List[str] = Field(default_factory=list)
    analysis: str


class ResumeResponse(BaseModel):
    file_name: str
    file_hash: str
    pages: int
    raw_text: str
    cleaned_text: str
    summary: ResumeSummary
    match: Optional[MatchResult] = None
    cached: bool = False


class CacheClient:
    def __init__(self) -> None:
        self.client = None
        redis_url = os.getenv("REDIS_URL")
        if redis is not None and redis_url:
            self.client = redis.Redis.from_url(redis_url, decode_responses=True)

    def get(self, key: str) -> Optional[dict]:
        if self.client is None:
            return None
        value = self.client.get(key)
        return json.loads(value) if value else None

    def set(self, key: str, value: dict) -> None:
        if self.client is None:
            return
        self.client.setex(key, CACHE_TTL_SECONDS, json.dumps(value, ensure_ascii=False))


cache = CacheClient()


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_pdf_text(content: bytes) -> tuple[str, int]:
    reader = PdfReader(io.BytesIO(content))
    pages = len(reader.pages)
    texts: List[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
    return "\n".join(texts), pages


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    return text.strip()


def infer_summary(text: str) -> ResumeSummary:
    name = None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        first_line = lines[0]
        if 2 <= len(first_line) <= 12 and re.search(r"[\u4e00-\u9fffA-Za-z]", first_line):
            name = first_line
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phone_match = re.search(r"(?:\+?86[-\s]?)?(1[3-9]\d{9})", text)
    address_match = re.search(r"(?:现住|住址|地址)[:：]?\s*(.{6,40})", text)
    intent_match = re.search(r"(?:求职意向|应聘职位|目标岗位)[:：]?\s*(.{2,30})", text)
    salary_match = re.search(r"(?:期望薪资|薪资要求)[:：]?\s*(.{2,30})", text)
    work_years_match = re.search(r"(\d+(?:\.\d+)?)(?:\s*)年(?:工作)?经验", text)
    edu_match = re.search(r"(?:学历|教育背景)[:：]?\s*(.{2,40})", text)
    project_lines = []
    for line in lines:
        if any(key in line for key in ["项目", "Project", "案例", "系统"]):
            project_lines.append(line)
    return ResumeSummary(
        name=name,
        phone=phone_match.group(1) if phone_match else None,
        email=email_match.group(0) if email_match else None,
        address=address_match.group(1).strip() if address_match else None,
        job_intent=intent_match.group(1).strip() if intent_match else None,
        expected_salary=salary_match.group(1).strip() if salary_match else None,
        work_years=work_years_match.group(1) if work_years_match else None,
        education=edu_match.group(1).strip() if edu_match else None,
        projects=project_lines[:5],
    )


def keyword_analysis(job_desc: str) -> List[str]:
    stopwords = {"的", "和", "与", "及", "岗位", "负责", "能力", "以及", "熟悉", "具备", "经验", "优先", "要求"}
    tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9_\-]{2,}", job_desc.lower())
    keywords = [t for t in tokens if t not in stopwords]
    seen = []
    for kw in keywords:
        if kw not in seen:
            seen.append(kw)
    return seen[:20]


def match_resume(summary: ResumeSummary, job_desc: str) -> MatchResult:
    keywords = keyword_analysis(job_desc)
    resume_text = " ".join(filter(None, [summary.name, summary.job_intent, summary.education, summary.work_years, summary.email, summary.phone, " ".join(summary.projects)]))
    matched = [kw for kw in keywords if kw.lower() in resume_text.lower() or kw in job_desc]
    missing = [kw for kw in keywords if kw not in matched]
    keyword_score = round((len(matched) / len(keywords) * 100) if keywords else 0, 2)
    experience_score = 0.0
    if summary.work_years:
        try:
            years = float(summary.work_years)
            experience_score = min(100.0, years * 20)
        except Exception:
            experience_score = 50.0
    overall = round(keyword_score * 0.7 + experience_score * 0.3, 2)
    analysis = f"岗位关键词共 {len(keywords)} 个，匹配 {len(matched)} 个。" + (
        f"候选人工作年限约 {summary.work_years} 年。" if summary.work_years else "未识别到明确工作年限。"
    )
    return MatchResult(
        keyword_score=keyword_score,
        experience_score=round(experience_score, 2),
        overall_score=overall,
        matched_keywords=matched,
        missing_keywords=missing,
        analysis=analysis,
    )


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "service": APP_NAME}


@app.get("/api/meta")
def api_meta() -> dict:
    return {
        "service": APP_NAME,
        "version": "1.0.0",
        "features": [
            "resume-upload",
            "pdf-parsing",
            "information-extraction",
            "job-matching",
            "json-response",
            "redis-cache-optional",
        ],
    }


@app.post("/api/resume/analyze", response_model=ResumeResponse)
async def analyze_resume(file: UploadFile = File(...), job_description: str = Form(default="")) -> ResumeResponse:
    if file.content_type not in {"application/pdf", "application/x-pdf"} and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")

    digest = file_hash(content)
    cache_key = f"resume:{digest}:{hashlib.sha256(job_description.encode('utf-8')).hexdigest()}"
    cached_value = cache.get(cache_key)
    if cached_value:
        cached_value["cached"] = True
        return ResumeResponse.model_validate(cached_value)

    raw_text, pages = extract_pdf_text(content)
    cleaned_text = clean_text(raw_text)
    summary = infer_summary(cleaned_text)
    match = match_resume(summary, job_description) if job_description.strip() else None

    payload = ResumeResponse(
        file_name=file.filename,
        file_hash=digest,
        pages=pages,
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        summary=summary,
        match=match,
        cached=False,
    )
    cache.set(cache_key, payload.model_dump())
    return payload


@app.post("/api/job/match")
def analyze_job_only(job_description: str = Form(...)) -> dict:
    keywords = keyword_analysis(job_description)
    return {"job_description": job_description, "keywords": keywords, "keyword_count": len(keywords)}
