from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from database.controllers.database_controller import DatabaseController
from database.models.message import Message

def build_template(user_id: int) -> str:
  """Build the template for the conversation."""

  template = """
  You are a helpful assistant that can answer questions and help with tasks.

  Here is the conversation history: {context}

  Question: {question}

  Answer:
  """

  return template

def build_chain(template: str):
  """Build the chain for the conversation."""
  llm = OllamaLLM(model="gemma3:latest")
  prompt = ChatPromptTemplate.from_template(template)
  chain = prompt | llm

  return chain

def build_context(messages: list[Message]) -> str:
  """Build the context for the conversation."""
  context = ""

  for message in messages:
    context += f"User: {message.question}\nAI: {message.answer}\n"

  return context

if __name__ == "__main__":
  db = DatabaseController()
  db.create_tables()

  # Get the user.
  user_id = 1

  # Build the template for the conversation and the chain.
  template = build_template(user_id)
  chain = build_chain(template)

  # Get the user's chat history and build the context.
  messages = db.messages.get_messages_by_user(user_id)
  context = build_context(messages)

  # Print the chat history.
  for message in messages:
    print(f"You: {message.question}\n")
    print(f"Bot: {message.answer}\n")

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
    answer = chain.invoke({
      "context": context,
      "question": question
    })

    print(f"\nBot: {answer}\n")

    # Save the conversation history
    context += f"User: {question}\nAI: {answer}"

    # Save to the database.
    db.messages.add_message(user_id, question, answer)