from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, desc, inspect, text
from sqlalchemy.orm import sessionmaker

from common.config import Config
from common.path_utils import get_file_path
from __002__db_helper_parse.model.base import Base
from __002__db_helper_parse.model.tb_interview_recording_analysis import TbInterviewRecordingAnalysis
from __002__db_helper_parse.model.tb_interview_recording_analysis_detail import TbInterviewRecordingAnalysisDetail
from __002__db_helper_parse.model.tb_user import TbUser

conf = Config()


class DatabaseHelper:
    def __init__(self, db_url):
        self.engine = create_engine(db_url, pool_size=10, max_overflow=20, pool_recycle=3600)
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        self._ensure_schema()

    def _ensure_schema(self):
        inspector = inspect(self.engine)
        columns = {column["name"] for column in inspector.get_columns(TbInterviewRecordingAnalysis.__tablename__)}
        migrations = []
        if "priority_level" not in columns:
            migrations.append("ALTER TABLE tb_interview_recording_analysis ADD COLUMN priority_level SMALLINT NOT NULL DEFAULT 0 COMMENT '优先级（0：普通，1：优先）'")
        if migrations:
            session = self.Session()
            try:
                for sql in migrations:
                    session.execute(text(sql))
                session.commit()
            except Exception as e:
                session.rollback()
                print(f"初始化数据库结构失败: {e}")
            finally:
                session.close()

    def get_user_by_id(self, user_id):
        session = self.Session()
        try:
            user = session.query(TbUser).filter(TbUser.user_id == user_id).first()
            return user
        except Exception as e:
            print(f"查询失败: {e}")
            return None
        finally:
            session.close()

    def add_interview_record(self, name, interview_time=None, company_name=None, recording_url=None, subject=None, priority_level=0):
        session = self.Session()
        try:
            if interview_time is None:
                interview_time = datetime.now()
            if not company_name:
                company_name = "未知公司"
            if recording_url is None:
                recording_url = ""
            if not subject:
                subject = "未知学科"

            new_record = TbInterviewRecordingAnalysis(
                name=name,
                interview_time=interview_time,
                company_name=company_name,
                recording_url=recording_url,
                processing_status=0,
                priority_level=priority_level,
                subject=subject,
            )
            session.add(new_record)
            session.commit()
            return new_record.id
        except Exception as e:
            print(f"添加面试记录失败: {e}")
            session.rollback()
            return None
        finally:
            session.close()

    def update_interview_record(self, record_id, update_fields):
        if not record_id:
            print("面试记录ID不能为空")
            return
        session = self.Session()
        try:
            record = session.query(TbInterviewRecordingAnalysis).filter(TbInterviewRecordingAnalysis.id == record_id).first()
            if record:
                for field, value in update_fields.items():
                    if hasattr(record, field):
                        setattr(record, field, value)
                session.commit()
        except Exception as e:
            print(f"更新面试记录失败: {e}")
            session.rollback()
        finally:
            session.close()

    def get_all_interview_records(self, filters=None, exclude_fields=None):
        session = self.Session()
        try:
            query = session.query(TbInterviewRecordingAnalysis)
            if filters:
                query = query.filter_by(**filters)
            if exclude_fields:
                all_columns = [column.name for column in TbInterviewRecordingAnalysis.__table__.columns]
                included_columns = [column for column in all_columns if column not in exclude_fields]
                query = query.with_entities(*[getattr(TbInterviewRecordingAnalysis, col) for col in included_columns])
            query = query.order_by(desc(TbInterviewRecordingAnalysis.priority_level), desc(TbInterviewRecordingAnalysis.update_time))
            records = query.all()
            records_dict = [record._asdict() if hasattr(record, "_asdict") else record.__dict__ for record in records]
            for record in records_dict:
                record.pop("_sa_instance_state", None)
            return records_dict
        except Exception as e:
            print(f"查询面试记录失败: {e}")
            return []
        finally:
            session.close()

    def add_interview_analysis_detail(self, interview_record_analysis_id, interview_question=None,
                                      interviewee_answer=None,
                                      reference_answer=None, point_analysis=None, answer_thoughts=None,
                                      answer_evaluation=None, answer_score=None):
        if not interview_record_analysis_id:
            print("面试记录分析ID不能为空")
            return
        session = self.Session()
        try:
            new_detail = TbInterviewRecordingAnalysisDetail(
                interview_record_analysis_id=interview_record_analysis_id,
                interview_question=interview_question,
                interviewee_answer=interviewee_answer,
                reference_answer=reference_answer,
                point_analysis=point_analysis,
                answer_thoughts=answer_thoughts,
                answer_evaluation=answer_evaluation,
                answer_score=answer_score,
            )
            session.add(new_detail)
            session.commit()
        except Exception as e:
            print(f"添加面试记录分析详情失败: {e}")
            session.rollback()
        finally:
            session.close()

    def get_analysis_details_by_record_id(self, interview_record_analysis_id):
        session = self.Session()
        try:
            details = session.query(TbInterviewRecordingAnalysisDetail).filter(
                TbInterviewRecordingAnalysisDetail.interview_record_analysis_id == interview_record_analysis_id).all()
            details_dict = [detail.__dict__ for detail in details]
            for detail in details_dict:
                detail.pop("_sa_instance_state", None)
            return details_dict
        except Exception as e:
            print(f"查询面试记录分析详情失败: {e}")
            return []
        finally:
            session.close()

    def delete_analysis_details_by_record_id(self, record_id):
        if not record_id:
            print("面试记录分析ID不能为空")
            return
        session = self.Session()
        try:
            session.query(TbInterviewRecordingAnalysisDetail).filter(
                TbInterviewRecordingAnalysisDetail.interview_record_analysis_id == record_id).delete()
            session.commit()
        except Exception as e:
            print(f"删除面试记录分析详情失败: {e}")
            session.rollback()
        finally:
            session.close()

    def clear_interview_records(self):
        session = self.Session()
        try:
            session.query(TbInterviewRecordingAnalysisDetail).delete()
            session.query(TbInterviewRecordingAnalysis).delete()
            session.commit()
            print("已清空所有面试记录")
        except Exception as e:
            print(f"清空面试记录失败: {e}")
            session.rollback()
        finally:
            session.close()


def get_db_helper(host, user_name, password, db_name):
    if not host or not user_name or not db_name:
        sqlite_path = Path(get_file_path("interview_voice.db"))
        return DatabaseHelper(f"sqlite:///{sqlite_path}")

    mysql_host = host
    mysql_port = 3306
    if ":" in host:
        mysql_host, port_str = host.rsplit(":", 1)
        if port_str.isdigit():
            mysql_port = int(port_str)

    database_url = f"mysql+pymysql://{user_name}:{password}@{mysql_host}:{mysql_port}/{db_name}"
    return DatabaseHelper(database_url)


my_db_helper = get_db_helper(conf.MYSQL_HOST, conf.MYSQL_USER, conf.MYSQL_PASSWORD, conf.MYSQL_DATABASE_NAME)
