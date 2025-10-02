import time
import statistics
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class EvaluationMetrics:
    """Evaluation and performance metrics for RAG system"""
    
    def __init__(self):
        self.query_history = []
        self.performance_data = []
    
    def log_query_performance(self, 
                             query: str, 
                             strategy: str, 
                             response_time: float, 
                             tokens_used: int, 
                             num_sources: int,
                             config: Dict[str, Any],
                             response_quality_score: Optional[float] = None):
        """Log performance data for a query"""
        
        performance_entry = {
            'timestamp': datetime.now(),
            'query': query,
            'strategy': strategy,
            'response_time': response_time,
            'tokens_used': tokens_used,
            'num_sources': num_sources,
            'config': config.copy(),
            'response_quality_score': response_quality_score
        }
        
        self.performance_data.append(performance_entry)
        
        # Keep only last 1000 entries to prevent memory issues
        if len(self.performance_data) > 1000:
            self.performance_data = self.performance_data[-1000:]
    
    def get_performance_summary(self, strategy: Optional[str] = None, 
                               time_range_hours: Optional[int] = None) -> Dict[str, Any]:
        """Get performance summary statistics"""
        
        # Filter data based on criteria
        filtered_data = self.performance_data.copy()
        
        if strategy:
            filtered_data = [d for d in filtered_data if d['strategy'] == strategy]
        
        if time_range_hours:
            cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
            filtered_data = [d for d in filtered_data if d['timestamp'] > cutoff_time]
        
        if not filtered_data:
            return self._empty_summary()
        
        # Calculate statistics
        response_times = [d['response_time'] for d in filtered_data]
        tokens_used = [d['tokens_used'] for d in filtered_data]
        num_sources = [d['num_sources'] for d in filtered_data]
        
        summary = {
            'total_queries': len(filtered_data),
            'avg_response_time': statistics.mean(response_times),
            'median_response_time': statistics.median(response_times),
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'std_response_time': statistics.stdev(response_times) if len(response_times) > 1 else 0,
            'avg_tokens_used': statistics.mean(tokens_used),
            'total_tokens_used': sum(tokens_used),
            'avg_sources_retrieved': statistics.mean(num_sources),
            'queries_per_hour': self._calculate_queries_per_hour(filtered_data),
            'strategy_breakdown': self._get_strategy_breakdown(filtered_data)
        }
        
        # Add percentiles
        if len(response_times) > 4:
            summary.update({
                'p95_response_time': np.percentile(response_times, 95),
                'p99_response_time': np.percentile(response_times, 99)
            })
        
        return summary
    
    def compare_strategies(self, strategies: List[str]) -> Dict[str, Dict[str, Any]]:
        """Compare performance between different RAG strategies"""
        
        comparison = {}
        
        for strategy in strategies:
            strategy_data = [d for d in self.performance_data if d['strategy'] == strategy]
            
            if strategy_data:
                response_times = [d['response_time'] for d in strategy_data]
                tokens_used = [d['tokens_used'] for d in strategy_data]
                
                comparison[strategy] = {
                    'query_count': len(strategy_data),
                    'avg_response_time': statistics.mean(response_times),
                    'median_response_time': statistics.median(response_times),
                    'avg_tokens_used': statistics.mean(tokens_used),
                    'success_rate': 1.0,  # Assuming all logged queries were successful
                    'p95_response_time': np.percentile(response_times, 95) if len(response_times) > 4 else max(response_times)
                }
            else:
                comparison[strategy] = self._empty_strategy_stats()
        
        return comparison
    
    def get_performance_trends(self, time_window_hours: int = 24) -> Dict[str, List[Any]]:
        """Get performance trends over time"""
        
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_data = [d for d in self.performance_data if d['timestamp'] > cutoff_time]
        
        if not recent_data:
            return {'timestamps': [], 'response_times': [], 'tokens_used': [], 'strategies': []}
        
        # Sort by timestamp
        recent_data.sort(key=lambda x: x['timestamp'])
        
        return {
            'timestamps': [d['timestamp'] for d in recent_data],
            'response_times': [d['response_time'] for d in recent_data],
            'tokens_used': [d['tokens_used'] for d in recent_data],
            'strategies': [d['strategy'] for d in recent_data],
            'queries': [d['query'] for d in recent_data]
        }
    
    def get_slowest_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the slowest queries for analysis"""
        
        if not self.performance_data:
            return []
        
        # Sort by response time in descending order
        sorted_data = sorted(self.performance_data, key=lambda x: x['response_time'], reverse=True)
        
        return sorted_data[:limit]
    
    def get_token_usage_analysis(self) -> Dict[str, Any]:
        """Analyze token usage patterns"""
        
        if not self.performance_data:
            return self._empty_token_analysis()
        
        tokens_by_strategy = {}
        for entry in self.performance_data:
            strategy = entry['strategy']
            if strategy not in tokens_by_strategy:
                tokens_by_strategy[strategy] = []
            tokens_by_strategy[strategy].append(entry['tokens_used'])
        
        analysis = {
            'total_tokens_used': sum(d['tokens_used'] for d in self.performance_data),
            'avg_tokens_per_query': statistics.mean([d['tokens_used'] for d in self.performance_data]),
            'strategy_token_usage': {}
        }
        
        for strategy, tokens in tokens_by_strategy.items():
            analysis['strategy_token_usage'][strategy] = {
                'avg_tokens': statistics.mean(tokens),
                'total_tokens': sum(tokens),
                'min_tokens': min(tokens),
                'max_tokens': max(tokens)
            }
        
        return analysis
    
    def evaluate_response_latency(self, target_latency_seconds: float = 3.0) -> Dict[str, Any]:
        """Evaluate response latency against target"""
        
        if not self.performance_data:
            return {'within_target': 0, 'total_queries': 0, 'percentage': 0}
        
        within_target = sum(1 for d in self.performance_data if d['response_time'] <= target_latency_seconds)
        total_queries = len(self.performance_data)
        
        return {
            'within_target': within_target,
            'total_queries': total_queries,
            'percentage': (within_target / total_queries) * 100,
            'target_latency': target_latency_seconds,
            'avg_latency': statistics.mean([d['response_time'] for d in self.performance_data])
        }
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report"""
        
        if not self.performance_data:
            return "No performance data available to generate report."
        
        summary = self.get_performance_summary()
        token_analysis = self.get_token_usage_analysis()
        latency_eval = self.evaluate_response_latency()
        
        report = f"""
# RAG System Performance Report

## Query Statistics
- Total Queries: {summary['total_queries']}
- Average Response Time: {summary['avg_response_time']:.2f}s
- Median Response Time: {summary['median_response_time']:.2f}s
- 95th Percentile Response Time: {summary.get('p95_response_time', 'N/A')}s

## Token Usage
- Total Tokens Used: {token_analysis['total_tokens_used']:,}
- Average Tokens per Query: {token_analysis['avg_tokens_per_query']:.1f}

## Latency Analysis
- Queries within 3s target: {latency_eval['within_target']}/{latency_eval['total_queries']} ({latency_eval['percentage']:.1f}%)

## Strategy Breakdown
"""
        
        for strategy, stats in summary['strategy_breakdown'].items():
            report += f"- {strategy}: {stats['count']} queries, avg {stats['avg_response_time']:.2f}s\n"
        
        return report
    
    def export_performance_data(self) -> pd.DataFrame:
        """Export performance data as pandas DataFrame"""
        
        if not self.performance_data:
            return pd.DataFrame()
        
        # Flatten the data for DataFrame
        flattened_data = []
        for entry in self.performance_data:
            flat_entry = {
                'timestamp': entry['timestamp'],
                'query': entry['query'],
                'strategy': entry['strategy'],
                'response_time': entry['response_time'],
                'tokens_used': entry['tokens_used'],
                'num_sources': entry['num_sources'],
                'chunk_size': entry['config'].get('chunk_size'),
                'chunk_overlap': entry['config'].get('chunk_overlap'),
                'top_k': entry['config'].get('top_k'),
                'retrieval_method': entry['config'].get('retrieval_method'),
                'response_quality_score': entry.get('response_quality_score')
            }
            flattened_data.append(flat_entry)
        
        return pd.DataFrame(flattened_data)
    
    def clear_performance_data(self):
        """Clear all performance data"""
        self.performance_data.clear()
        self.query_history.clear()
    
    def _empty_summary(self) -> Dict[str, Any]:
        """Return empty summary structure"""
        return {
            'total_queries': 0,
            'avg_response_time': 0,
            'median_response_time': 0,
            'min_response_time': 0,
            'max_response_time': 0,
            'std_response_time': 0,
            'avg_tokens_used': 0,
            'total_tokens_used': 0,
            'avg_sources_retrieved': 0,
            'queries_per_hour': 0,
            'strategy_breakdown': {}
        }
    
    def _empty_strategy_stats(self) -> Dict[str, Any]:
        """Return empty strategy statistics"""
        return {
            'query_count': 0,
            'avg_response_time': 0,
            'median_response_time': 0,
            'avg_tokens_used': 0,
            'success_rate': 0,
            'p95_response_time': 0
        }
    
    def _empty_token_analysis(self) -> Dict[str, Any]:
        """Return empty token analysis"""
        return {
            'total_tokens_used': 0,
            'avg_tokens_per_query': 0,
            'strategy_token_usage': {}
        }
    
    def _calculate_queries_per_hour(self, data: List[Dict[str, Any]]) -> float:
        """Calculate queries per hour from data"""
        if len(data) < 2:
            return 0
        
        sorted_data = sorted(data, key=lambda x: x['timestamp'])
        time_span = sorted_data[-1]['timestamp'] - sorted_data[0]['timestamp']
        
        if time_span.total_seconds() == 0:
            return 0
        
        hours = time_span.total_seconds() / 3600
        return len(data) / hours
    
    def _get_strategy_breakdown(self, data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Get breakdown by strategy"""
        breakdown = {}
        
        for entry in data:
            strategy = entry['strategy']
            if strategy not in breakdown:
                breakdown[strategy] = {
                    'count': 0,
                    'total_response_time': 0,
                    'total_tokens': 0
                }
            
            breakdown[strategy]['count'] += 1
            breakdown[strategy]['total_response_time'] += entry['response_time']
            breakdown[strategy]['total_tokens'] += entry['tokens_used']
        
        # Calculate averages
        for strategy_data in breakdown.values():
            if strategy_data['count'] > 0:
                strategy_data['avg_response_time'] = strategy_data['total_response_time'] / strategy_data['count']
                strategy_data['avg_tokens'] = strategy_data['total_tokens'] / strategy_data['count']
        
        return breakdown
