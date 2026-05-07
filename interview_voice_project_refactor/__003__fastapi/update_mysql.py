from __002__db_helper_parse.db_helper import my_db_helper


async def update_mysql(msg: str, record_id=None, processing_status=None):
    update_fields = {"processing_tips": msg}
    if processing_status is not None:
        update_fields["processing_status"] = processing_status
    print(f"[record_id={record_id}] {msg}")
    my_db_helper.update_interview_record(record_id, update_fields)
