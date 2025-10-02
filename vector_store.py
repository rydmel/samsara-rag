import os
import pickle
import hashlib
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st

class VectorStore:
    """Vector store implementation using ChromaDB"""
    
    def __init__(self, persist_directory: str = "./chromadb"):
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        self.collection_name = "samsara_customer_stories"
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except:
            # Create new collection if it doesn't exist
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        # Store for full documents (for parent document retrieval)
        self.full_documents = {}
        self._load_full_documents()
    
    def is_populated(self) -> bool:
        """Check if the vector store has any documents"""
        try:
            count = self.collection.count()
            return count > 0
        except:
            return False
    
    def populate_store(self, customer_stories: List[Dict[str, Any]]):
        """Populate the vector store with customer stories"""
        
        st.info("Processing and storing customer stories...")
        progress_bar = st.progress(0)
        
        all_documents = []
        
        for i, story in enumerate(customer_stories):
            try:
                # Create documents from the story
                documents = self._create_documents_from_story(story)
                all_documents.extend(documents)
                
                # Store full document for parent retrieval
                doc_id = self._generate_doc_id(story['url'])
                self.full_documents[story['url']] = story
                
                progress_bar.progress((i + 1) / len(customer_stories))
                
            except Exception as e:
                st.warning(f"Error processing story {story.get('title', 'Unknown')}: {str(e)}")
        
        # Add documents to ChromaDB
        if all_documents:
            self._add_documents_to_collection(all_documents)
            self._save_full_documents()
            st.success(f"Successfully stored {len(all_documents)} document chunks!")
        else:
            st.error("No documents were processed successfully")
    
    def _create_documents_from_story(self, story: Dict[str, Any]) -> List[Document]:
        """Create Document objects from a customer story"""
        documents = []
        
        # Main content
        content = story.get('content', '')
        if content:
            chunks = self.text_splitter.split_text(content)
            
            for i, chunk in enumerate(chunks):
                metadata = {
                    'source': story.get('url', ''),
                    'title': story.get('title', ''),
                    'company_name': story.get('company_name', ''),
                    'industry': story.get('industry', ''),
                    'chunk_index': i,
                    'content_type': 'main_content'
                }
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=metadata
                ))
        
        # Highlights
        highlights = story.get('highlights', [])
        if highlights:
            highlights_text = "\n".join([f"• {highlight}" for highlight in highlights])
            metadata = {
                'source': story.get('url', ''),
                'title': story.get('title', ''),
                'company_name': story.get('company_name', ''),
                'industry': story.get('industry', ''),
                'content_type': 'highlights'
            }
            
            documents.append(Document(
                page_content=f"Key highlights for {story.get('company_name', 'this company')}:\n{highlights_text}",
                metadata=metadata
            ))
        
        # ROI metrics
        roi_metrics = story.get('roi_metrics', [])
        if roi_metrics:
            metrics_text = "\n".join([f"• {metric}" for metric in roi_metrics])
            metadata = {
                'source': story.get('url', ''),
                'title': story.get('title', ''),
                'company_name': story.get('company_name', ''),
                'industry': story.get('industry', ''),
                'content_type': 'roi_metrics'
            }
            
            documents.append(Document(
                page_content=f"ROI and performance metrics for {story.get('company_name', 'this company')}:\n{metrics_text}",
                metadata=metadata
            ))
        
        # Challenges and solutions
        challenges = story.get('challenges', [])
        solutions = story.get('solutions', [])
        
        if challenges or solutions:
            challenge_solution_text = ""
            if challenges:
                challenge_solution_text += "Challenges:\n" + "\n".join([f"• {challenge}" for challenge in challenges]) + "\n\n"
            if solutions:
                challenge_solution_text += "Solutions:\n" + "\n".join([f"• {solution}" for solution in solutions])
            
            metadata = {
                'source': story.get('url', ''),
                'title': story.get('title', ''),
                'company_name': story.get('company_name', ''),
                'industry': story.get('industry', ''),
                'content_type': 'challenges_solutions'
            }
            
            documents.append(Document(
                page_content=challenge_solution_text,
                metadata=metadata
            ))
        
        # Competitor information
        competitor_info = story.get('competitor_info', '')
        if competitor_info:
            metadata = {
                'source': story.get('url', ''),
                'title': story.get('title', ''),
                'company_name': story.get('company_name', ''),
                'industry': story.get('industry', ''),
                'content_type': 'competitor_info'
            }
            
            documents.append(Document(
                page_content=f"{story.get('company_name', 'This company')} previously used or switched from: {competitor_info}",
                metadata=metadata
            ))
        
        return documents
    
    def _add_documents_to_collection(self, documents: List[Document]):
        """Add documents to ChromaDB collection"""
        
        if not documents:
            return
        
        # Prepare data for ChromaDB
        ids = []
        texts = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            # Create unique ID
            doc_id = self._generate_doc_id(f"{doc.metadata.get('source', '')}_{i}")
            ids.append(doc_id)
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            
            try:
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    metadatas=batch_metadatas
                )
            except Exception as e:
                st.warning(f"Error adding batch {i//batch_size + 1}: {str(e)}")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(k, self.collection.count())
            )
            
            documents = []
            if results['documents'] and results['documents'][0]:
                for i, (text, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                    documents.append(Document(
                        page_content=text,
                        metadata=metadata
                    ))
            
            return documents
            
        except Exception as e:
            st.warning(f"Error during similarity search: {str(e)}")
            return []
    
    def keyword_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform keyword-based search"""
        try:
            # ChromaDB doesn't have built-in keyword search, so we'll use similarity search
            # but with query preprocessing to emphasize keywords
            keywords = query.lower().split()
            expanded_query = " ".join(keywords + [f"important: {word}" for word in keywords])
            
            return self.similarity_search(expanded_query, k)
            
        except Exception as e:
            st.warning(f"Error during keyword search: {str(e)}")
            return []
    
    def get_full_document(self, source: str) -> Optional[Document]:
        """Get the full document for parent document retrieval"""
        if source in self.full_documents:
            story = self.full_documents[source]
            
            # Combine all content into one document
            full_content = f"Company: {story.get('company_name', 'Unknown')}\n"
            full_content += f"Industry: {story.get('industry', 'Unknown')}\n\n"
            full_content += f"Story: {story.get('content', '')}\n\n"
            
            if story.get('highlights'):
                full_content += "Key Highlights:\n"
                full_content += "\n".join([f"• {h}" for h in story['highlights']]) + "\n\n"
            
            if story.get('roi_metrics'):
                full_content += "ROI Metrics:\n"
                full_content += "\n".join([f"• {m}" for m in story['roi_metrics']]) + "\n\n"
            
            metadata = {
                'source': story.get('url', ''),
                'title': story.get('title', ''),
                'company_name': story.get('company_name', ''),
                'industry': story.get('industry', ''),
                'content_type': 'full_document'
            }
            
            return Document(
                page_content=full_content,
                metadata=metadata
            )
        
        return None
    
    def _generate_doc_id(self, content: str) -> str:
        """Generate unique document ID"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def _save_full_documents(self):
        """Save full documents to disk"""
        try:
            full_docs_path = os.path.join(self.persist_directory, "full_documents.pkl")
            with open(full_docs_path, 'wb') as f:
                pickle.dump(self.full_documents, f)
        except Exception as e:
            st.warning(f"Error saving full documents: {str(e)}")
    
    def _load_full_documents(self):
        """Load full documents from disk"""
        try:
            full_docs_path = os.path.join(self.persist_directory, "full_documents.pkl")
            if os.path.exists(full_docs_path):
                with open(full_docs_path, 'rb') as f:
                    self.full_documents = pickle.load(f)
        except Exception as e:
            st.warning(f"Error loading full documents: {str(e)}")
            self.full_documents = {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        try:
            count = self.collection.count()
            industries = set()
            companies = set()
            
            # Get some sample data to analyze
            if count > 0:
                sample_results = self.collection.get(limit=min(100, count))
                if sample_results['metadatas']:
                    for metadata in sample_results['metadatas']:
                        if metadata.get('industry'):
                            industries.add(metadata['industry'])
                        if metadata.get('company_name'):
                            companies.add(metadata['company_name'])
            
            return {
                'total_chunks': count,
                'total_companies': len(companies),
                'industries': list(industries),
                'full_documents': len(self.full_documents)
            }
            
        except Exception as e:
            return {
                'total_chunks': 0,
                'total_companies': 0,
                'industries': [],
                'full_documents': 0,
                'error': str(e)
            }
