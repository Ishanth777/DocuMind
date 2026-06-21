from src.search import RAGSearch


# Example usage
if __name__ == "__main__":
    
    #print(store.query("What is attention mechanism?", top_k=3))
    rag_search = RAGSearch()
    while(True):
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        summary = rag_search.search_and_summarize(query, top_k=2)
        print("Summary:", summary)