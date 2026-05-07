from __001__langgraph_more_node.langgraph_agent import interview_voice_analyse
from common.path_utils import get_file_path
from common.time_utils import get_datetime_str_from_datetime
from __002__db_helper_parse.db_helper import my_db_helper
import time
import asyncio


def run_langgraph():
    while True:
        all_interview_records = my_db_helper.get_all_interview_records({"processing_status": 0})
        if all_interview_records:
            interview_record = all_interview_records[0]
            print(interview_record)
            # 更新面试记录为处理中
            my_db_helper.update_interview_record(interview_record['id'], {"processing_status": 1})
            my_db_helper.delete_analysis_details_by_record_id(interview_record['id'])
            # "input_audio_path": file_location, "name": name, "company": company, "subject": subject,
            # "interview_date_str": interview_date_str
            interview_info_dict = {
                "name": interview_record["name"],
                "company": interview_record["company_name"],
                "subject": interview_record["subject"],
                "interview_date_str": get_datetime_str_from_datetime(interview_record['interview_time'])
            }
            asyncio.run(interview_voice_analyse(get_file_path(interview_record['recording_url']),
                                                interview_record['id'], interview_info_dict))
            # 更新面试记录为处理完成
            my_db_helper.update_interview_record(interview_record['id'], {"processing_status": 2})
        time.sleep(30)


run_langgraph()
