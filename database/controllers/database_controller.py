from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from database.controllers.message_controller import MessageController
from database.models.base import Base

class DatabaseController:
  """
  The class to handle the database.
  """

  # The database engine.
  ENGINE = create_engine('sqlite:///app.db')

  # The session class that is used to interact with the database.
  session: Session

  # The message controller.
  messages: MessageController

  def __init__(self):
    # Create the session.
    _Session = sessionmaker(bind=self.ENGINE)
    self.session = _Session()

    # Create the controllers.
    self.messages = MessageController(self.session)

  def create_tables(self, delete_existing: bool = False):
    """Create the tables in the database."""

    if delete_existing:
      Base.metadata.drop_all(self.ENGINE)

    Base.metadata.create_all(self.ENGINE)