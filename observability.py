import os
import time
import json
from typing import Dict, Any, Optional
from datetime import datetime
import uuid
import streamlit as st

try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False

class ObservabilityTracker:
    """Observability and tracing for RAG operations"""
    
    def __init__(self):
        self.traces = {}
        self.metrics = []
        
        # Initialize Logfire if available
        if LOGFIRE_AVAILABLE:
            try:
                # Configure Logfire with token from environment
                logfire_token = os.getenv('LOGFIRE_TOKEN')
                if logfire_token:
                    logfire.configure(token=logfire_token)
                    self.logfire_enabled = True
                    print("✓ Pydantic Logfire initialized successfully with your account")
                else:
                    logfire.configure()
                    self.logfire_enabled = True
                    print("✓ Pydantic Logfire initialized (no token found)")
            except Exception as e:
                print(f"⚠ Logfire configuration failed: {str(e)}")
                self.logfire_enabled = False
        else:
            print("⚠ Pydantic Logfire not available - using local logging")
            self.logfire_enabled = False
    
    def start_trace(self, query: str, config: Dict[str, Any]) -> str:
        """Start tracing a RAG query"""
        trace_id = str(uuid.uuid4())
        
        trace_data = {
            'trace_id': trace_id,
            'query': query,
            'config': config,
            'start_time': time.time(),
            'timestamp': datetime.now(),
            'status': 'started'
        }
        
        self.traces[trace_id] = trace_data
        
        # Log to Logfire if available
        if self.logfire_enabled:
            try:
                with logfire.span('rag_query_start') as span:
                    span.set_attribute('trace_id', trace_id)
                    span.set_attribute('query', query)
                    span.set_attribute('rag_strategy', config.get('strategy', 'unknown'))
                    span.set_attribute('top_k', config.get('top_k', 0))
            except Exception as e:
                print(f"⚠ Logfire logging error: {str(e)}")
        
        return trace_id
    
    def end_trace(self, trace_id: str, response: Dict[str, Any], num_documents: int):
        """End tracing a RAG query"""
        if trace_id not in self.traces:
            return
        
        trace_data = self.traces[trace_id]
        end_time = time.time()
        
        trace_data.update({
            'end_time': end_time,
            'duration': end_time - trace_data['start_time'],
            'status': 'completed',
            'response_tokens': response.get('tokens_used', 0),
            'num_documents_retrieved': num_documents,
            'context_length': response.get('context_length', 0)
        })
        
        # Store metrics
        metric = {
            'trace_id': trace_id,
            'query': trace_data['query'],
            'strategy': trace_data['config'].get('strategy', 'unknown'),
            'duration': trace_data['duration'],
            'tokens_used': response.get('tokens_used', 0),
            'num_documents': num_documents,
            'timestamp': trace_data['timestamp'],
            'success': True
        }
        
        self.metrics.append(metric)
        
        # Log to Logfire if available
        if self.logfire_enabled:
            try:
                with logfire.span('rag_query_complete') as span:
                    span.set_attribute('trace_id', trace_id)
                    span.set_attribute('duration', trace_data['duration'])
                    span.set_attribute('tokens_used', response.get('tokens_used', 0))
                    span.set_attribute('num_documents', num_documents)
                    span.set_attribute('success', True)
            except Exception as e:
                st.warning(f"Logfire logging error: {str(e)}")
        
        return trace_data
    
    def log_error(self, trace_id: str, error_message: str):
        """Log an error for a trace"""
        if trace_id in self.traces:
            trace_data = self.traces[trace_id]
            trace_data.update({
                'status': 'error',
                'error_message': error_message,
                'end_time': time.time()
            })
            
            # Store error metric
            metric = {
                'trace_id': trace_id,
                'query': trace_data.get('query', ''),
                'strategy': trace_data.get('config', {}).get('strategy', 'unknown'),
                'duration': trace_data.get('end_time', time.time()) - trace_data.get('start_time', time.time()),
                'error': error_message,
                'timestamp': trace_data.get('timestamp', datetime.now()),
                'success': False
            }
            
            self.metrics.append(metric)
            
            # Log to Logfire if available
            if self.logfire_enabled:
                try:
                    with logfire.span('rag_query_error') as span:
                        span.set_attribute('trace_id', trace_id)
                        span.set_attribute('error_message', error_message)
                        span.record_exception(Exception(error_message))
                except Exception as e:
                    st.warning(f"Logfire error logging failed: {str(e)}")
    
    def log_retrieval_step(self, trace_id: str, step_name: str, duration: float, metadata: Dict[str, Any]):
        """Log a retrieval step within a trace"""
        if self.logfire_enabled:
            try:
                with logfire.span(f'retrieval_step_{step_name}') as span:
                    span.set_attribute('trace_id', trace_id)
                    span.set_attribute('step_name', step_name)
                    span.set_attribute('duration', duration)
                    
                    for key, value in metadata.items():
                        span.set_attribute(key, value)
                        
            except Exception as e:
                st.warning(f"Logfire step logging error: {str(e)}")
    
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get trace data by ID"""
        return self.traces.get(trace_id)
    
    def get_all_traces(self) -> Dict[str, Dict[str, Any]]:
        """Get all trace data"""
        return self.traces.copy()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary metrics"""
        if not self.metrics:
            return {
                'total_queries': 0,
                'avg_duration': 0,
                'success_rate': 0,
                'avg_tokens': 0,
                'strategy_breakdown': {}
            }
        
        successful_metrics = [m for m in self.metrics if m.get('success', False)]
        
        strategy_breakdown = {}
        for metric in self.metrics:
            strategy = metric.get('strategy', 'unknown')
            if strategy not in strategy_breakdown:
                strategy_breakdown[strategy] = {
                    'count': 0,
                    'total_duration': 0,
                    'total_tokens': 0,
                    'successes': 0
                }
            
            strategy_breakdown[strategy]['count'] += 1
            strategy_breakdown[strategy]['total_duration'] += metric.get('duration', 0)
            strategy_breakdown[strategy]['total_tokens'] += metric.get('tokens_used', 0)
            
            if metric.get('success', False):
                strategy_breakdown[strategy]['successes'] += 1
        
        # Calculate averages for each strategy
        for strategy_data in strategy_breakdown.values():
            if strategy_data['count'] > 0:
                strategy_data['avg_duration'] = strategy_data['total_duration'] / strategy_data['count']
                strategy_data['avg_tokens'] = strategy_data['total_tokens'] / strategy_data['count']
                strategy_data['success_rate'] = strategy_data['successes'] / strategy_data['count']
        
        return {
            'total_queries': len(self.metrics),
            'successful_queries': len(successful_metrics),
            'avg_duration': sum(m.get('duration', 0) for m in successful_metrics) / len(successful_metrics) if successful_metrics else 0,
            'success_rate': len(successful_metrics) / len(self.metrics) if self.metrics else 0,
            'avg_tokens': sum(m.get('tokens_used', 0) for m in successful_metrics) / len(successful_metrics) if successful_metrics else 0,
            'strategy_breakdown': strategy_breakdown
        }
    
    def export_traces(self) -> str:
        """Export all traces as JSON"""
        export_data = {
            'traces': self.traces,
            'metrics': self.metrics,
            'exported_at': datetime.now().isoformat()
        }
        
        return json.dumps(export_data, indent=2, default=str)
    
    def clear_data(self):
        """Clear all trace and metric data"""
        self.traces.clear()
        self.metrics.clear()
        
        if self.logfire_enabled:
            try:
                with logfire.span('observability_data_cleared') as span:
                    span.set_attribute('action', 'clear_data')
                    span.set_attribute('timestamp', datetime.now().isoformat())
            except Exception as e:
                st.warning(f"Logfire clear logging error: {str(e)}")
