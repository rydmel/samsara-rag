import os
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import streamlit as st

from vector_store import VectorStore
from observability import ObservabilityTracker

@dataclass
class RAGConfig:
    """Configuration for RAG operations"""
    strategy: str = "naive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    retrieval_method: str = "semantic"
    max_tokens: int = 2048

class RAGEngine:
    """RAG engine with multiple retrieval strategies"""
    
    def __init__(self, vector_store: VectorStore, obs_tracker: ObservabilityTracker):
        self.vector_store = vector_store
        self.obs_tracker = obs_tracker
        
        # Initialize OpenAI client
        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "default_key")
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    
    def query(self, question: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process a query using the specified RAG configuration"""
        
        rag_config = RAGConfig(
            strategy=config.get('strategy', 'naive'),
            chunk_size=config.get('chunk_size', 1000),
            chunk_overlap=config.get('chunk_overlap', 200),
            top_k=config.get('top_k', 5),
            retrieval_method=config.get('retrieval_method', 'semantic'),
            max_tokens=config.get('max_tokens', 2048)
        )
        
        # Start tracking this query
        trace_id = self.obs_tracker.start_trace(question, rag_config.__dict__)
        
        try:
            # Retrieve relevant documents based on strategy
            if rag_config.strategy == "naive":
                documents = self._naive_retrieval(question, rag_config)
            elif rag_config.strategy == "parent_document":
                documents = self._parent_document_retrieval(question, rag_config)
            elif rag_config.strategy == "hybrid":
                documents = self._hybrid_retrieval(question, rag_config)
            else:
                documents = self._naive_retrieval(question, rag_config)
            
            # Generate response
            response = self._generate_response(question, documents, rag_config)
            
            # End tracking
            self.obs_tracker.end_trace(trace_id, response, len(documents))
            
            return response
            
        except Exception as e:
            self.obs_tracker.log_error(trace_id, str(e))
            raise e
    
    def _naive_retrieval(self, question: str, config: RAGConfig) -> List[Document]:
        """Simple semantic similarity retrieval"""
        
        if config.retrieval_method == "semantic":
            results = self.vector_store.similarity_search(question, k=config.top_k)
        elif config.retrieval_method == "keyword":
            results = self.vector_store.keyword_search(question, k=config.top_k)
        else:  # hybrid
            # Combine semantic and keyword search
            semantic_results = self.vector_store.similarity_search(question, k=config.top_k//2)
            keyword_results = self.vector_store.keyword_search(question, k=config.top_k//2)
            
            # Merge and deduplicate
            all_results = semantic_results + keyword_results
            seen_content = set()
            results = []
            for doc in all_results:
                if doc.page_content not in seen_content:
                    results.append(doc)
                    seen_content.add(doc.page_content)
                if len(results) >= config.top_k:
                    break
        
        return results
    
    def _parent_document_retrieval(self, question: str, config: RAGConfig) -> List[Document]:
        """Parent document retrieval - find relevant chunks then return full documents"""
        
        # First, get relevant chunks
        chunk_results = self.vector_store.similarity_search(question, k=config.top_k * 2)
        
        # Get parent documents for these chunks
        parent_docs = []
        seen_sources = set()
        
        for chunk in chunk_results:
            source = chunk.metadata.get('source', '')
            if source and source not in seen_sources:
                # Get the full document
                full_doc = self.vector_store.get_full_document(source)
                if full_doc:
                    parent_docs.append(full_doc)
                    seen_sources.add(source)
                
                if len(parent_docs) >= config.top_k:
                    break
        
        return parent_docs
    
    def _hybrid_retrieval(self, question: str, config: RAGConfig) -> List[Document]:
        """Hybrid retrieval combining multiple strategies"""
        
        # Get results from both naive and parent document retrieval
        naive_results = self._naive_retrieval(question, config)
        parent_results = self._parent_document_retrieval(question, config)
        
        # Combine and rank results
        all_results = naive_results + parent_results
        
        # Simple deduplication based on content similarity
        unique_results = []
        for doc in all_results:
            is_duplicate = False
            for existing in unique_results:
                if self._calculate_similarity(doc.page_content, existing.page_content) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append(doc)
            
            if len(unique_results) >= config.top_k:
                break
        
        return unique_results
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Simple similarity calculation based on word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _generate_response(self, question: str, documents: List[Document], config: RAGConfig) -> Dict[str, Any]:
        """Generate response using OpenAI API"""
        
        # Prepare context from retrieved documents
        context = self._prepare_context(documents)
        
        # Create prompt
        prompt = self._create_prompt(question, context)
        
        try:
            # Generate response using OpenAI
            start_time = time.time()
            
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_completion_tokens=config.max_tokens,
                stream=False
            )
            
            end_time = time.time()
            
            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            # Extract sources
            sources = [doc.metadata.get('source', 'Unknown') for doc in documents]
            
            return {
                "answer": answer,
                "sources": sources,
                "tokens_used": tokens_used,
                "response_time": end_time - start_time,
                "context_length": len(context)
            }
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return {
                "answer": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "tokens_used": 0,
                "response_time": 0,
                "context_length": 0
            }
    
    def _prepare_context(self, documents: List[Document]) -> str:
        """Prepare context from retrieved documents"""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown Source')
            company = doc.metadata.get('company_name', 'Unknown Company')
            industry = doc.metadata.get('industry', 'Unknown Industry')
            
            context_part = f"--- Source {i}: {company} ({industry}) ---\n"
            context_part += f"URL: {source}\n"
            context_part += f"Content: {doc.page_content}\n\n"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create the prompt for the language model"""
        prompt = f"""
Based on the following customer stories and information about Samsara, please answer the user's question.

Context from Samsara Customer Stories:
{context}

User Question: {question}

Please provide a comprehensive answer based on the provided context. If the context doesn't contain enough information to fully answer the question, please say so and provide what information is available.

When mentioning specific customers or examples, please reference them clearly. If you mention metrics like ROI, cost savings, or efficiency improvements, please cite the specific customer story where this information came from.
"""
        return prompt
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the AI assistant"""
        return """You are an expert assistant specializing in Samsara's customer success stories and use cases. Your role is to help users understand how different companies have benefited from Samsara's solutions.

Key guidelines:
1. Always base your responses on the provided customer story context
2. Be specific about which customers you're referencing
3. Highlight relevant metrics, ROI, and business outcomes when available
4. If asked about specific industries, focus on customers from those sectors
5. When discussing competitors, only mention information explicitly stated in the customer stories
6. Be helpful and informative while staying accurate to the source material
7. If the context doesn't contain sufficient information, clearly state this limitation

Your responses should be professional, informative, and focused on helping the user understand Samsara's value proposition through real customer examples."""

    def update_text_splitter(self, chunk_size: int, chunk_overlap: int):
        """Update text splitter configuration"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
