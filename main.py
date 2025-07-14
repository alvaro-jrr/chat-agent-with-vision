from database.controllers.database_controller import DatabaseController
from slm import SLM
from vision import Vision

if __name__ == "__main__":
  db = DatabaseController()
  db.create_tables()

  # Load the SLM and vision model.
  vision = Vision()
  slm = SLM(db)

  # Get the user.
  user_id = 1

  # Handle the conversation.
  slm.handle_conversation(user_id)