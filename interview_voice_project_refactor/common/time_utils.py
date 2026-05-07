from datetime import datetime


def get_current_time():
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    return datetime_str


def get_current_date():
    datetime_str = datetime.now().strftime("%Y%m%d")
    return datetime_str


def get_datetime_str_from_datetime(date_time):
    return date_time.strftime("%Y%m%d")


def get_datetime_from_str(date_str):
    return datetime.strptime(date_str, "%Y%m%d")
