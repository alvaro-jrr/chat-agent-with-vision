from database.controllers.database_controller import DatabaseController
from slm import SLM
from vision import Vision

if __name__ == "__main__":
  print("----- Chatbot -----\n")

  # Create the database.
  db = DatabaseController()
  db.create_tables()

  # Load the SLM and vision model.
  vision = Vision()
  slm = SLM(db, vision)

  # Handle the conversation.
  slm.handle_conversation()