from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String

from database.models.base import Base

class Message(Base):
  """
  This table is used to store the messages between the user and the bot.
  """

  __tablename__ = "messages"

  id = Column(Integer(), primary_key=True)
  user_id = Column(Integer(), nullable=False)
  question = Column(String(), nullable=False)
  answer = Column(String(), nullable=False)
  created_at = Column(DateTime(), default=datetime.now())

  def __repr__(self):
    return f"<Message(id={self.id}, user_id={self.user_id}, question={self.question}, answer={self.answer}, created_at={self.created_at})>"
