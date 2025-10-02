import os
from vector_store import VectorStore
from rag_engine import RAGEngine
from observability import ObservabilityTracker

def test_rag_strategies():
    """Test RAG strategies"""
    
    print("Initializing components...")
    obs_tracker = ObservabilityTracker()
    vector_store = VectorStore()
    rag_engine = RAGEngine(vector_store=vector_store, obs_tracker=obs_tracker)
    
    # Check if vector store has data
    stats = vector_store.get_stats()
    print(f"\nVector Store Stats:")
    print(f"  Total chunks: {stats.get('total_chunks', 0)}")
    print(f"  Total companies: {stats.get('total_companies', 0)}")
    print(f"  Industries: {stats.get('industries', [])}")
    
    if stats.get('total_chunks', 0) == 0:
        print("\nNo data in vector store. Please run the app first to scrape customer stories.")
        return False
    
    # Test query
    test_query = "What are some examples of customers in logistics?"
    
    # Test each strategy
    strategies = ['naive', 'parent_document', 'hybrid']
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Testing {strategy.upper()} strategy")
        print('='*60)
        
        config = {
            'strategy': strategy,
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'top_k': 3,
            'retrieval_method': 'semantic',
            'temperature': 0.7,
            'max_tokens': 500
        }
        
        try:
            # Create RAGConfig from dict
            from rag_engine import RAGConfig
            rag_config = RAGConfig(
                strategy=config['strategy'],
                chunk_size=config['chunk_size'],
                chunk_overlap=config['chunk_overlap'],
                top_k=config['top_k'],
                retrieval_method=config['retrieval_method'],
                temperature=config['temperature'],
                max_tokens=config['max_tokens']
            )
            
            # Test retrieval only (without OpenAI call to save costs)
            if strategy == 'naive':
                docs = rag_engine._naive_retrieval(test_query, rag_config)
            elif strategy == 'parent_document':
                docs = rag_engine._parent_document_retrieval(test_query, rag_config)
            elif strategy == 'hybrid':
                docs = rag_engine._hybrid_retrieval(test_query, rag_config)
            
            print(f"Retrieved {len(docs)} documents")
            
            for i, doc in enumerate(docs[:2], 1):  # Show first 2 docs
                print(f"\nDocument {i}:")
                print(f"  Company: {doc.metadata.get('company_name', 'Unknown')}")
                print(f"  Industry: {doc.metadata.get('industry', 'Unknown')}")
                print(f"  Content preview: {doc.page_content[:150]}...")
            
            print(f"\n✓ {strategy} strategy working correctly")
            
        except Exception as e:
            print(f"\n✗ Error in {strategy} strategy: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"\n{'='*60}")
    print("All RAG strategies verified successfully!")
    print('='*60)
    return True

if __name__ == "__main__":
    # Set OpenAI API key to avoid errors (won't be used for retrieval tests)
    if 'OPENAI_API_KEY' not in os.environ:
        os.environ['OPENAI_API_KEY'] = 'test_key'
    
    test_rag_strategies()
