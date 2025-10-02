import streamlit as st
import os
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

from scraper import SamsaraCustomerScraper
from rag_engine import RAGEngine
from vector_store import VectorStore
from observability import ObservabilityTracker
from evaluation import EvaluationMetrics

# Page configuration
st.set_page_config(
    page_title="Samsara RAG Chatbot",
    page_icon="🚚",
    layout="wide"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.messages = []
    st.session_state.query_history = []
    st.session_state.performance_data = []

def initialize_app():
    """Initialize the application components"""
    if not st.session_state.initialized:
        with st.spinner("Initializing application..."):
            try:
                # Initialize observability tracker
                st.session_state.obs_tracker = ObservabilityTracker()
                
                # Initialize vector store
                st.session_state.vector_store = VectorStore()
                
                # Initialize RAG engine
                st.session_state.rag_engine = RAGEngine(
                    vector_store=st.session_state.vector_store,
                    obs_tracker=st.session_state.obs_tracker
                )
                
                # Initialize evaluation metrics
                st.session_state.evaluator = EvaluationMetrics()
                
                # Check if data exists, if not scrape it
                if not st.session_state.vector_store.is_populated():
                    st.info("No existing data found. Scraping Samsara customer stories...")
                    scraper = SamsaraCustomerScraper()
                    stories = scraper.scrape_customer_stories()
                    
                    if stories:
                        st.session_state.vector_store.populate_store(stories)
                        st.success(f"Successfully scraped and indexed {len(stories)} customer stories!")
                    else:
                        st.error("Failed to scrape customer stories. Please check your connection.")
                        return False
                
                st.session_state.initialized = True
                return True
                
            except Exception as e:
                st.error(f"Failed to initialize application: {str(e)}")
                return False
    return True

def main():
    st.title("🚚 Samsara RAG Chatbot")
    st.markdown("Ask questions about Samsara's customer success stories and experiences.")
    
    # Initialize the application
    if not initialize_app():
        st.stop()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "⚙️ Configuration", "📊 Evaluation"])
    
    with tab1:
        chat_interface()
    
    with tab2:
        configuration_interface()
    
    with tab3:
        evaluation_interface()

def chat_interface():
    """Main chat interface"""
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "metadata" in message:
                with st.expander("View Sources"):
                    for i, source in enumerate(message["metadata"].get("sources", []), 1):
                        st.markdown(f"**Source {i}:** {source}")
    
    # Chat input
    if prompt := st.chat_input("Ask about Samsara customers..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get RAG configuration from session state
        config = st.session_state.get('rag_config', {
            'strategy': 'naive',
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'top_k': 5,
            'retrieval_method': 'semantic'
        })
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time = time.time()
                
                try:
                    response = st.session_state.rag_engine.query(prompt, config)
                    end_time = time.time()
                    
                    # Display response
                    st.markdown(response["answer"])
                    
                    # Store performance data
                    performance_data = {
                        "query": prompt,
                        "strategy": config["strategy"],
                        "response_time": end_time - start_time,
                        "timestamp": datetime.now(),
                        "tokens_used": response.get("tokens_used", 0),
                        "sources_count": len(response.get("sources", []))
                    }
                    st.session_state.performance_data.append(performance_data)
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "metadata": {
                            "sources": response.get("sources", []),
                            "performance": performance_data
                        }
                    })
                    
                    # Show sources
                    if response.get("sources"):
                        with st.expander("View Sources"):
                            for i, source in enumerate(response["sources"], 1):
                                st.markdown(f"**Source {i}:** {source}")
                
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

def configuration_interface():
    """RAG configuration interface"""
    st.header("RAG Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Retrieval Strategy")
        strategy = st.selectbox(
            "Select RAG Pattern",
            options=["naive", "parent_document", "hybrid"],
            index=0,
            help="Choose the RAG strategy to use for document retrieval"
        )
        
        st.subheader("Chunking Parameters")
        chunk_size = st.slider(
            "Chunk Size",
            min_value=200,
            max_value=2000,
            value=1000,
            step=100,
            help="Size of text chunks for processing"
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=200,
            step=50,
            help="Overlap between consecutive chunks"
        )
    
    with col2:
        st.subheader("Retrieval Parameters")
        top_k = st.slider(
            "Top-K Retrieval",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of top documents to retrieve"
        )
        
        retrieval_method = st.selectbox(
            "Retrieval Method",
            options=["semantic", "keyword", "hybrid"],
            index=0,
            help="Method for retrieving relevant documents"
        )
        
        st.subheader("Model Parameters")
        max_tokens = st.slider(
            "Max Completion Tokens",
            min_value=100,
            max_value=8192,
            value=2048,
            help="Maximum tokens for model response"
        )
    
    # Update configuration
    config = {
        'strategy': strategy,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'top_k': top_k,
        'retrieval_method': retrieval_method,
        'max_tokens': max_tokens
    }
    
    st.session_state.rag_config = config
    
    # Show current configuration
    st.subheader("Current Configuration")
    st.json(config)
    
    # Reset chat button
    if st.button("Reset Chat History"):
        st.session_state.messages = []
        st.success("Chat history cleared!")
        st.rerun()

def evaluation_interface():
    """Performance evaluation and monitoring interface"""
    st.header("Performance Evaluation & Monitoring")
    
    if not st.session_state.performance_data:
        st.info("No performance data available. Start chatting to see metrics!")
        return
    
    # Convert performance data to DataFrame
    df = pd.DataFrame(st.session_state.performance_data)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_response_time = df['response_time'].mean()
        st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
    
    with col2:
        total_queries = len(df)
        st.metric("Total Queries", total_queries)
    
    with col3:
        avg_tokens = df['tokens_used'].mean() if 'tokens_used' in df else 0
        st.metric("Avg Tokens Used", f"{avg_tokens:.0f}")
    
    with col4:
        avg_sources = df['sources_count'].mean()
        st.metric("Avg Sources Retrieved", f"{avg_sources:.1f}")
    
    # Performance over time
    st.subheader("Performance Over Time")
    fig_time = px.line(
        df, 
        x='timestamp', 
        y='response_time',
        color='strategy',
        title="Response Time Over Time by Strategy"
    )
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Strategy comparison
    st.subheader("Strategy Performance Comparison")
    strategy_stats = df.groupby('strategy').agg({
        'response_time': ['mean', 'std'],
        'tokens_used': 'mean',
        'sources_count': 'mean'
    }).round(3)
    
    st.dataframe(strategy_stats)
    
    # Response time distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig_dist = px.histogram(
            df, 
            x='response_time', 
            color='strategy',
            title="Response Time Distribution",
            nbins=20
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        fig_box = px.box(
            df, 
            x='strategy', 
            y='response_time',
            title="Response Time by Strategy"
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Query history
    st.subheader("Query History")
    
    # Display recent queries
    for i, data in enumerate(reversed(st.session_state.performance_data[-10:]), 1):
        with st.expander(f"Query {len(st.session_state.performance_data) - i + 1}: {data['query'][:60]}..."):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Strategy:** {data['strategy']}")
                st.write(f"**Response Time:** {data['response_time']:.2f}s")
                st.write(f"**Timestamp:** {data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            with col2:
                st.write(f"**Tokens Used:** {data.get('tokens_used', 'N/A')}")
                st.write(f"**Sources Retrieved:** {data['sources_count']}")
    
    # Export data
    st.subheader("Export Performance Data")
    if st.button("Download Performance Data as CSV"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"rag_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
