from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """
You are a helpful assistant that can answer questions and help with tasks.

Here is the conversation history: {context}

Question: {question}

Answer:
"""

llm = OllamaLLM(model="gemma3:latest")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm

def handle_conversation():
  context: str = ""

  print("Welcome to the chatbot! Type 'exit' to end the conversation.")

  while True:
    question = input("You: ")

    if question.lower() == "exit":
      break

    result = chain.invoke({
      "context": context,
      "question": question
    })

    print(f"Bot: {result}\n")

    # Save the conversation history
    context += f"User: {question}\nAI: {result}"

if __name__ == "__main__":
  handle_conversation()