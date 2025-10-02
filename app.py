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
    page_title="Samsara Customer Assistant",
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
    # Display logo and title
    col1, col2 = st.columns([1, 12])
    with col1:
        st.image("attached_assets/image_1759439317085.png", width=80)
    with col2:
        st.title("Samsara Customer Assistant")
    
    st.markdown("Ask questions about Samsara's customer success stories and experiences.")
    
    # Initialize the application
    if not initialize_app():
        st.stop()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "⚙️ Configuration", "📊 Evaluation", "📚 Knowledge Base"])
    
    with tab1:
        chat_interface()
    
    with tab2:
        configuration_interface()
    
    with tab3:
        evaluation_interface()
    
    with tab4:
        knowledge_base_interface()

def chat_interface():
    """Main chat interface with elegant multi-turn conversation"""
    
    # Sidebar for conversation controls and settings
    with st.sidebar:
        st.subheader("💬 Conversation")
        
        # Show sources toggle
        show_sources = st.checkbox("Show sources", value=False, help="Display source documents for each response")
        
        # Show performance metrics toggle
        show_metrics = st.checkbox("Show metrics", value=False, help="Display performance metrics for each response")
        
        st.divider()
        
        # Quick actions
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # Show message count
        st.caption(f"💬 {len(st.session_state.messages)} messages in conversation")
    
    # Check if vector store has data
    stats = st.session_state.vector_store.get_stats()
    if stats.get('total_chunks', 0) == 0:
        st.warning("⚠️ The knowledge base is empty. Please go to the Configuration tab and click 'Refresh Database' to load customer stories.")
        return
    
    # Main chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show metadata if available and enabled
                if message["role"] == "assistant" and "metadata" in message:
                    metadata = message["metadata"]
                    
                    # Create a compact metadata section
                    if show_sources or show_metrics:
                        cols = []
                        if show_metrics and "performance" in metadata:
                            cols.append(1)
                        if show_sources and metadata.get("sources"):
                            cols.append(1)
                        
                        if cols:
                            col_objs = st.columns(len(cols))
                            col_idx = 0
                            
                            # Performance metrics
                            if show_metrics and "performance" in metadata:
                                with col_objs[col_idx]:
                                    perf = metadata["performance"]
                                    st.caption(f"⏱️ {perf['response_time']:.2f}s | 🔤 {perf.get('tokens_used', 'N/A')} tokens | 📚 {perf['sources_count']} sources")
                                col_idx += 1
                            
                            # Sources in compact expander
                            if show_sources and metadata.get("sources"):
                                with col_objs[col_idx]:
                                    with st.expander(f"📚 {len(metadata['sources'])} sources", expanded=False):
                                        for j, source in enumerate(metadata["sources"], 1):
                                            st.caption(f"**{j}.** {source}")
    
    # Pinned input at the bottom
    if prompt := st.chat_input("Ask about Samsara customers...", key="chat_input"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get RAG configuration from session state
        config = st.session_state.get('rag_config', {
            'strategy': 'naive',
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'top_k': 5,
            'retrieval_method': 'semantic',
            'temperature': 1.0,
            'max_tokens': 2048,
            'max_agent_steps': 3,
            'agent_confidence_threshold': 0.7,
            'enable_reflection': True
        })
        
        # Display user message immediately
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and stream response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                start_time = time.time()
                
                try:
                    # Show spinner while processing
                    with st.spinner(""):
                        response = st.session_state.rag_engine.query(prompt, config)
                    end_time = time.time()
                    
                    # Display the complete response
                    message_placeholder.markdown(response["answer"])
                    
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
                    
                    # Add assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "metadata": {
                            "sources": response.get("sources", []),
                            "performance": performance_data
                        }
                    })
                    
                    # Show inline metadata if enabled
                    if show_sources or show_metrics:
                        cols = []
                        if show_metrics:
                            cols.append(1)
                        if show_sources and response.get("sources"):
                            cols.append(1)
                        
                        if cols:
                            col_objs = st.columns(len(cols))
                            col_idx = 0
                            
                            if show_metrics:
                                with col_objs[col_idx]:
                                    st.caption(f"⏱️ {performance_data['response_time']:.2f}s | 🔤 {performance_data.get('tokens_used', 'N/A')} tokens | 📚 {performance_data['sources_count']} sources")
                                col_idx += 1
                            
                            if show_sources and response.get("sources"):
                                with col_objs[col_idx]:
                                    with st.expander(f"📚 {len(response['sources'])} sources", expanded=False):
                                        for j, source in enumerate(response["sources"], 1):
                                            st.caption(f"**{j}.** {source}")
                
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    message_placeholder.markdown(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
        
        # Rerun to update the conversation
        st.rerun()

def configuration_interface():
    """RAG configuration interface"""
    st.header("RAG Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Retrieval Strategy")
        strategy = st.selectbox(
            "Select RAG Pattern",
            options=["naive", "parent_document", "hybrid", "agentic"],
            index=0,
            help="Choose the RAG strategy to use for document retrieval"
        )
        
        # Agentic RAG specific parameters
        if strategy == "agentic":
            st.info("🤖 Agentic RAG uses multi-step reasoning for complex queries")
            max_agent_steps = st.slider(
                "Max Reasoning Steps",
                min_value=1,
                max_value=5,
                value=3,
                help="Maximum number of reasoning iterations the agent can perform"
            )
            agent_confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.5,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Stop early if agent confidence exceeds this threshold"
            )
            enable_reflection = st.checkbox(
                "Enable Reflection",
                value=True,
                help="Allow agent to reflect and synthesize findings"
            )
        else:
            max_agent_steps = 3
            agent_confidence_threshold = 0.7
            enable_reflection = True
        
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
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Controls randomness in responses (0=focused, 2=creative)"
        )
        
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
        'temperature': temperature,
        'max_tokens': max_tokens,
        'max_agent_steps': max_agent_steps,
        'agent_confidence_threshold': agent_confidence_threshold,
        'enable_reflection': enable_reflection
    }
    
    st.session_state.rag_config = config
    
    # Show current configuration
    st.subheader("Current Configuration")
    st.json(config)
    
    # Actions
    col1, col2 = st.columns(2)
    
    with col1:
        # Export configuration
        config_export = {
            'timestamp': datetime.now().isoformat(),
            'config': config
        }
        config_json = json.dumps(config_export, indent=2)
        st.download_button(
            label="⬇️ Export Configuration",
            data=config_json,
            file_name=f"rag_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            help="Download current RAG configuration as JSON"
        )
    
    with col2:
        # Reset chat button
        if st.button("🗑️ Reset Chat History"):
            st.session_state.messages = []
            st.success("Chat history cleared!")
            st.rerun()
    
    # Database Management Section
    st.divider()
    st.subheader("📚 Vector Database Management")
    
    # Show database stats
    stats = st.session_state.vector_store.get_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Chunks", stats.get('total_chunks', 0))
    with col2:
        st.metric("Companies", stats.get('total_companies', 0))
    with col3:
        st.metric("Full Documents", stats.get('full_documents', 0))
    with col4:
        industries_count = len(stats.get('industries', []))
        st.metric("Industries", industries_count)
    
    if stats.get('industries'):
        st.info(f"**Industries:** {', '.join(stats['industries'])}")
    
    # Database actions
    st.subheader("Database Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Database", help="Clear and re-scrape all customer stories"):
            with st.spinner("Refreshing database..."):
                from scraper import SamsaraCustomerScraper
                scraper = SamsaraCustomerScraper()
                customer_stories = scraper.scrape_customer_stories()
                
                if customer_stories:
                    st.session_state.vector_store.refresh_store(customer_stories)
                    st.success("Database refreshed successfully!")
                    st.rerun()
                else:
                    st.error("Failed to scrape customer stories")
    
    with col2:
        if st.button("➕ Update Database", help="Add new stories without removing existing ones"):
            with st.spinner("Updating database..."):
                from scraper import SamsaraCustomerScraper
                scraper = SamsaraCustomerScraper()
                customer_stories = scraper.scrape_customer_stories()
                
                if customer_stories:
                    st.session_state.vector_store.add_or_update_stories(customer_stories)
                    st.success("Database updated successfully!")
                    st.rerun()
                else:
                    st.error("Failed to scrape customer stories")
    
    # Warning for clear action
    with st.expander("⚠️ Danger Zone"):
        st.warning("**Clear Database**: This action will permanently delete all stored customer stories and embeddings.")
        if st.button("🗑️ Clear All Data", type="primary"):
            if st.session_state.vector_store.clear_store():
                st.success("Database cleared successfully!")
                st.rerun()
            else:
                st.error("Failed to clear database")

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
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV export
        csv = df.to_csv(index=False)
        st.download_button(
            label="📊 Download CSV",
            data=csv,
            file_name=f"rag_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Export performance metrics as CSV"
        )
    
    with col2:
        # JSON export
        json_data = json.dumps(st.session_state.performance_data, default=str, indent=2)
        st.download_button(
            label="📄 Download JSON",
            data=json_data,
            file_name=f"rag_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            help="Export performance metrics as JSON"
        )
    
    with col3:
        # Chat logs export
        if st.session_state.messages:
            chat_export = {
                'timestamp': datetime.now().isoformat(),
                'total_messages': len(st.session_state.messages),
                'conversation': st.session_state.messages
            }
            # Use default=str to handle datetime serialization
            chat_json = json.dumps(chat_export, default=str, indent=2)
            st.download_button(
                label="💬 Download Chat Logs",
                data=chat_json,
                file_name=f"chat_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Export conversation history as JSON"
            )
        else:
            st.button("💬 Download Chat Logs", disabled=True, help="No chat messages to export")

def knowledge_base_interface():
    """Display indexed customer stories database"""
    st.header("📚 Knowledge Base - Indexed Customer Stories")
    
    # Get statistics
    stats = st.session_state.vector_store.get_stats()
    
    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", stats.get('full_documents', 0))
    with col2:
        st.metric("Total Chunks", stats.get('total_chunks', 0))
    with col3:
        st.metric("Industries", len(stats.get('industries', [])))
    
    st.divider()
    
    # Get all full documents
    full_docs = st.session_state.vector_store.full_documents
    
    if not full_docs:
        st.warning("⚠️ No customer stories in the knowledge base. Go to Configuration tab to load data.")
        return
    
    # Convert to table format
    table_data = []
    for url, story in full_docs.items():
        # Extract path segment (last part of URL)
        path_segment = url.rstrip('/').split('/')[-1] if url else 'unknown'
        
        table_data.append({
            'Path Segment': path_segment,
            'Full URL': url,
            'Content Size': len(story.get('content', ''))
        })
    
    # Create DataFrame and sort by path segment
    df = pd.DataFrame(table_data)
    df = df.sort_values('Path Segment')
    
    # Display table
    st.dataframe(
        df,
        use_container_width=True,
        height=400,
        column_config={
            "Path Segment": st.column_config.TextColumn(
                "Path Segment",
                width="medium"
            ),
            "Full URL": st.column_config.LinkColumn(
                "Full URL",
                width="large"
            ),
            "Content Size": st.column_config.NumberColumn(
                "Content Size",
                format="%d chars",
                width="small"
            )
        }
    )
    
    # Export database
    st.divider()
    st.subheader("Export Knowledge Base")
    
    col1, col2 = st.columns(2)
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📊 Download as CSV",
            data=csv,
            file_name=f"knowledge_base_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Export customer stories database as CSV"
        )
    
    with col2:
        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            label="📄 Download as JSON",
            data=json_data,
            file_name=f"knowledge_base_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            help="Export customer stories database as JSON"
        )

if __name__ == "__main__":
    main()
