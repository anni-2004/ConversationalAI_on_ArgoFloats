from rag_system import RAGSQLQueryExecutor

rag = RAGSQLQueryExecutor()

while True:
    q = input("Ask a question: ")
    result = rag.query_with_rag(q)
    print(result["generated_query"])
    print(result["enhanced_response"])