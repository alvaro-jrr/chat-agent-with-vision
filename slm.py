from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from database.controllers.database_controller import DatabaseController
from database.models.message import Message

class SLM:
  """
  The class to handle the SLM.
  """

  # The database controller.
  db: DatabaseController

  def __init__(self, db: DatabaseController):
    self.db = db

  def __build_template(self, user_id: int) -> str:
    """Build the template for the conversation."""

    template = """
    You are a helpful assistant that can answer questions and help with tasks.

    Here is the conversation history: {context}

    Question: {question}

    Answer:
    """

    return template

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
    print(f"Bot: {answer}\n")

  def __log_messages(self, messages: list[Message]):
    """Log the messages to the console."""
    for message in messages:
      self.__log_message(message.question, message.answer)

  def __log_answer(self, answer: str):
    """Log the answer to the console."""
    print(f"Bot: {answer}\n")

  def handle_conversation(self, user_id: int):
    """Handle the conversation."""

    # Get the user's chat history and build the context.
    messages = self.db.messages.get_messages_by_user(user_id)
    context = self.__build_context(messages)

    # Build the template for the conversation.
    template = self.__build_template(user_id)
    chain = self.__build_chain(template)

    # Log the history.
    self.__log_messages(messages)

    # Log the welcome message.
    if len(messages) > 0:
      print("> Welcome back! Type 'exit' to end the conversation.\n")
    else:
      print("> Welcome to the chatbot! Type 'exit' to end the conversation.\n")

    while True:
      # Get the question from the user.
      question = input("You: ")

      if question.lower() == "exit":
        break

      # Run the chain.
      answer = chain.invoke({"context": context, "question": question})

      # Log the answer.
      self.__log_answer(answer)

      # Update the context.
      context += f"User: {question}\nAI: {answer}"


      # Save to the database.
      self.db.messages.add_message(user_id, question, answer)