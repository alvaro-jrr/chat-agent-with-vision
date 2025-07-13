from sqlalchemy.orm import Session

from database.models.message import Message

class MessageController:
  """
  The class to handle the messages.
  """

  def __init__(self, session: Session):
    self.session = session

  def add_message(self, user_id: int, question: str, answer: str) -> None:
    """Add a message to the database."""

    message = Message(user_id=user_id, question=question, answer=answer)
    self.session.add(message)
    self.session.commit()

  def get_messages_by_user(self, user_id: int) -> list[Message]:
    """Get all messages for an user."""
    return self.session.query(Message).filter(Message.user_id == user_id).all()