from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from database.controllers.database_controller import DatabaseController
from database.models.message import Message
from vision import Vision

class SLM:
  """
  The class to handle the SLM.
  """

  # The database controller.
  db: DatabaseController

  # The vision model.
  vision: Vision

  def __init__(self, db: DatabaseController, vision: Vision):
    self.db = db
    self.vision = vision

  def __build_template(self, user_id: int) -> str:
    """Build the template for the conversation."""

    templates = {
      "helpful": """
      You are a helpful and friendly assistant that can answer questions and help with tasks.
      Always be polite, clear, and supportive in your responses.

      The user is feeling {emotion}. Adapt your response accordingly:
      - If they're happy/laughing: Match their positive energy and enthusiasm
      - If they're sad: Be extra gentle, supportive, and offer comfort
      - If they're angry: Stay calm, be understanding, and help them feel heard
      - If they're tired: Be patient, concise, and considerate of their energy
      - If they're surprised: Be excited with them and help process the surprise
      - If they're thoughtful: Give them space to think and provide thoughtful responses
      - If they're neutral: Be balanced and professional
      
      Here is the conversation history: {context}

      Question: {question}

      Answer:
      """,
      
      "sarcastic": """
      You are a witty and sarcastic assistant with a sharp sense of humor.
      Respond with clever remarks and playful sarcasm while still being helpful.
      Use emojis occasionally and don't be afraid to make jokes.

      The user is feeling {emotion}. Adapt your response accordingly:
      - If they're happy/laughing: Match their positive energy and enthusiasm
      - If they're sad: Be extra gentle, supportive, and offer comfort
      - If they're angry: Stay calm, be understanding, and help them feel heard
      - If they're tired: Be patient, concise, and considerate of their energy
      - If they're surprised: Be excited with them and help process the surprise
      - If they're thoughtful: Give them space to think and provide thoughtful responses
      - If they're neutral: Be balanced and professional

      Here is the conversation history: {context}

      Question: {question}

      Answer:
      """,
    }

    if user_id == 0:
      return templates['sarcastic']
    else:
      return templates['helpful']

  def __build_chain(self, template: str):
    """Build the chain for the conversation."""
    llm = OllamaLLM(model="gemma3:latest")
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm

    return chain

  def __build_context(self, messages: list[Message]) -> str:
    """Build the context for the conversation."""
    context = ""

    for message in messages:
      context += f"User: {message.question}\nAI: {message.answer}\n"

    return context

  def __log_message(self, question, answer):
    """Log the message to the console."""
    print(f"You: {question}\n")
    print(f"Bot: {answer.strip()}\n")

  def __log_messages(self, messages: list[Message]):
    """Log the messages to the console."""
    for message in messages:
      self.__log_message(message.question, message.answer)

  def __log_answer(self, answer: str):
    """Log the answer to the console."""
    print(f"\nBot: {answer.strip()}\n")

  def __get_user(self) -> int:
    """Get the user."""

    print("To start the conversation, please provide a photo so I can identify you 😊\n")

    while True:
      image_path = input("Enter the image path: ")

      try:
        prediction = self.vision.predict(image_path)
        print()
        break
      except:
        print("Enter a valid image path.\n")

    return prediction[0]

  def __get_emotion(self) -> str:
    """Get the emotion."""

    print("\nTo recognize your emotion, please provide a photo 😊\n")

    while True:
      image_path = input("Enter the image path: ")

      try:
        prediction = self.vision.predict(image_path)
        break
      except:
        print("Enter a valid image path.\n")

    return self.vision.get_expression_name(prediction[1])

  def handle_conversation(self):
    """Handle the conversation."""

    # Get the user.
    user_id = self.__get_user()
    user_name = self.vision.get_developer_name(user_id).title()

    # Get the user's chat history and build the context.
    messages = self.db.messages.get_messages_by_user(user_id)
    context = self.__build_context(messages)

    # Build the template for the conversation.
    template = self.__build_template(user_id)
    chain = self.__build_chain(template)

    # Log the history.
    if len(messages) > 0:
      print("# Conversation History\n")
      self.__log_messages(messages)

    # Log the welcome message.
    print(f"> Welcome {user_name}!\n")
    print("* Type 'emotion' to recognize your emotion from a photo.")
    print("* Type 'exit' to end the conversation.\n")

    # Set default emotion.
    emotion = "neutral"

    while True:
      # Get the question from the user.
      question = input("You: ")

      if question.lower() == "exit":
        break

      if question.lower() == "emotion":
        emotion = self.__get_emotion()
        print(f"You are feeling {emotion}.\n")
        continue

      # Run the chain.
      answer = chain.invoke({"context": context, "question": question, "emotion": emotion})

      # Log the answer.
      self.__log_answer(answer)

      # Update the context.
      context += f"User: {question}\nAI: {answer}"

      # Save to the database.
      self.db.messages.add_message(user_id, question, answer)