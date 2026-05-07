import json
import os
import shutil
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile

from __002__db_helper_parse.db_helper import my_db_helper
from common.path_utils import get_file_extension, get_file_path
from common.time_utils import get_current_time, get_datetime_from_str
from src.core.workflow import interview_voice_analyse


USE_BACKGROUND_ANALYSIS = True


def _log(message: str):
    print(f"[interview-voice-api] {message}", flush=True)


def _success(message="ok", **extra):
    return {"message": message, **extra}


app = FastAPI(title="Interview Voice API")
UPLOAD_DIR = Path(get_file_path("uploaded_files"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


STATUS_TEXT_MAP = {
    0: "未处理",
    1: "处理中",
    2: "已完成",
    -1: "处理失败",
}


@app.get("/health")
async def health_check():
    return _success(status="ok", service="interview-voice-api")


@app.get("/interview_records")
async def get_interview_records():
    records = my_db_helper.get_all_interview_records(exclude_fields=["markdown_text"])
    _log(f"query records count={len(records)}")
    return _success(data=records)


@app.get("/get_interview_records_by_record_id")
async def get_interview_records_by_record_id(record_id: int):
    records = my_db_helper.get_all_interview_records({"id": record_id})
    _log(f"query record detail record_id={record_id}, found={bool(records)}")
    return _success(data=records[0] if records else {})


@app.delete("/clear_interview_records")
@app.post("/clear_interview_records")
@app.get("/clear_interview_records")
@app.delete("/api/clear_interview_records")
@app.post("/api/clear_interview_records")
@app.get("/api/clear_interview_records")
async def clear_interview_records():
    _log("clear records requested")
    my_db_helper.clear_interview_records()
    return _success()


@app.post("/add_interview_record")
async def add_interview_record(
    background_tasks: BackgroundTasks,
    json_data_str: str = Form(...),
    file: UploadFile = File(...),
):
    data_dict = json.loads(json_data_str)
    name = data_dict.get("name", "")
    company = data_dict.get("company", "")
    subject = data_dict.get("subject", "")
    interview_date_str = data_dict.get("interview_date_str", "")
    priority_level = int(data_dict.get("priority_level", 0) or 0)

    extension = get_file_extension(file.filename)
    save_file_name = f"{get_current_time()}_{name}_{company}{extension}"
    relative_path = os.path.join("uploaded_files", save_file_name)
    file_location = UPLOAD_DIR / save_file_name

    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)

    record_id = my_db_helper.add_interview_record(
        name=name,
        company_name=company,
        subject=subject,
        interview_time=get_datetime_from_str(interview_date_str),
        recording_url=relative_path,
        priority_level=priority_level,
    )

    _log(f"record created record_id={record_id}, file={file_location}")
    if record_id:
        my_db_helper.update_interview_record(
            record_id,
            {"processing_status": 1, "processing_tips": "已提交，等待后台执行（如果长时间不变，说明任务未被调度）"},
        )
        _log(f"record submitted record_id={record_id}, status={STATUS_TEXT_MAP[1]}")
    else:
        _log("record submitted failed, record_id is None")

    if USE_BACKGROUND_ANALYSIS and record_id:
        background_tasks.add_task(
            _run_analysis,
            record_id,
            str(file_location),
            name,
            company,
            subject,
            interview_date_str,
        )
        return {"message": "ok", "record_id": record_id, "recording_url": relative_path, "status": "submitted"}

    return {"message": "ok", "record_id": record_id, "recording_url": relative_path, "status": "stored"}


async def _run_analysis(record_id: int, file_path: str, name: str, company: str, subject: str, interview_date_str: str):
    try:
        _log(f"analysis start record_id={record_id}")
        if not record_id:
            raise ValueError("record_id 不能为空")
        my_db_helper.update_interview_record(record_id, {"processing_status": 1, "processing_tips": "后台任务已开始，正在进行语音分析"})
        result = await interview_voice_analyse(
            file_path,
            record_id,
            {
                "name": name,
                "company": company,
                "subject": subject,
                "interview_date_str": interview_date_str,
            },
        )
        _log(f"analysis done record_id={record_id}")
        my_db_helper.update_interview_record(record_id, {"processing_status": 2, "processing_tips": "处理完成"})
        return result
    except Exception as e:
        _log(f"analysis failed record_id={record_id}, error={e}")
        if record_id:
            my_db_helper.update_interview_record(record_id, {"processing_status": -1, "processing_tips": f"处理失败: {e}"})
        return None


@app.post("/analyse_record")
async def analyse_record(
    background_tasks: BackgroundTasks,
    record_id: int,
    file_path: str,
    name: str,
    company: str,
    subject: str,
    interview_date_str: str,
):
    background_tasks.add_task(_run_analysis, record_id, file_path, name, company, subject, interview_date_str)
    print(f"[analysis submitted] record_id={record_id}, file_path={file_path}")
    return {"record_id": record_id, "status": "submitted"}


@app.post("/analyse_record/")
async def analyse_record_slash(
    background_tasks: BackgroundTasks,
    record_id: int,
    file_path: str,
    name: str,
    company: str,
    subject: str,
    interview_date_str: str,
):
    background_tasks.add_task(_run_analysis, record_id, file_path, name, company, subject, interview_date_str)
    return {"record_id": record_id, "status": "submitted"}
