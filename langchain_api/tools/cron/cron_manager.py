import os
from datetime import datetime
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select

from langchain_api.constant import home_path


class CronJob(SQLModel, table=True):
    __tablename__ = "cron_jobs"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    cron_expression: str
    command: str
    description: Optional[str] = None
    enabled: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class CronManager:
    def __init__(self, db_url: Optional[str] = None):
        if db_url is None:
            os.makedirs(home_path, exist_ok=True)
            db_url = f"sqlite:///{os.path.join(home_path, 'cron.db')}"

        self.engine = create_engine(db_url, echo=False)
        SQLModel.metadata.create_all(self.engine)

    def add(self, name: str, cron_expression: str, command: str, description: Optional[str] = None) -> CronJob:
        with Session(self.engine) as session:
            job = CronJob(name=name, cron_expression=cron_expression, command=command, description=description)
            session.add(job)
            session.commit()
            session.refresh(job)
            return job

    def list(self, enabled: Optional[bool] = None) -> list[CronJob]:
        with Session(self.engine) as session:
            statement = select(CronJob)
            if enabled is not None:
                statement = statement.where(CronJob.enabled == enabled)
            statement = statement.order_by(CronJob.created_at.desc())
            return session.exec(statement).all()

    def get(self, job_id: Optional[int] = None, name: Optional[str] = None) -> Optional[CronJob]:
        with Session(self.engine) as session:
            if job_id is not None:
                return session.get(CronJob, job_id)
            if name:
                statement = select(CronJob).where(CronJob.name == name)
                return session.exec(statement).first()
        return None

    def remove(self, job_id: Optional[int] = None, name: Optional[str] = None) -> bool:
        with Session(self.engine) as session:
            if job_id is not None:
                job = session.get(CronJob, job_id)
            elif name:
                statement = select(CronJob).where(CronJob.name == name)
                job = session.exec(statement).first()
            else:
                return False

            if job is None:
                return False

            session.delete(job)
            session.commit()
            return True

    def update(self, job_id: Optional[int] = None, name: Optional[str] = None, **kwargs) -> Optional[CronJob]:
        with Session(self.engine) as session:
            if job_id is not None:
                job = session.get(CronJob, job_id)
            elif name:
                statement = select(CronJob).where(CronJob.name == name)
                job = session.exec(statement).first()
            else:
                return None

            if job is None:
                return None

            for key, value in kwargs.items():
                if hasattr(job, key) and value is not None:
                    setattr(job, key, value)

            job.updated_at = datetime.utcnow()
            session.add(job)
            session.commit()
            session.refresh(job)
            return job


_manager: Optional[CronManager] = None


def get_cron_manager(db_url: Optional[str] = None) -> CronManager:
    global _manager
    if _manager is None or db_url is not None:
        _manager = CronManager(db_url)
    return _manager
