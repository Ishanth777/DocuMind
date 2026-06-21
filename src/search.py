import os
from dotenv import load_dotenv
from src.vector_store import vector_store_example
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, llm_model: str = "llama-3.1-8b-instant"):

        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")
        
    def search_and_summarize(self, query: str, top_k: int) -> str:
              results = vector_store_example(query)
              
              context = results
              if not context:
                     return "No relevant documents found."
              prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
              response = self.llm.invoke([prompt])
              return response.content
        
# #Example usage
# if __name__ == "__main__":
#     rag_search = RAGSearch()
#     query = "Who is ronaldo?"
#     summary = rag_search.search_and_summarize(query, top_k=2)
#     print("Summary:", summary)  