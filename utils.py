import re
import string
from typing import List, Dict, Any, Optional
import hashlib
import json
from datetime import datetime

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-\'""]', ' ', text)
    
    # Remove multiple consecutive punctuation
    text = re.sub(r'[.,!?;]+', lambda m: m.group()[0], text)
    
    # Strip and return
    return text.strip()

def extract_keywords(text: str, min_length: int = 3, max_keywords: int = 20) -> List[str]:
    """Extract keywords from text"""
    if not text:
        return []
    
    # Convert to lowercase and split
    words = text.lower().split()
    
    # Remove stopwords and short words
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'between', 'among', 'within', 'without', 'against', 'across',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
        'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
    }
    
    # Filter words
    keywords = []
    for word in words:
        # Remove punctuation
        word = word.strip(string.punctuation)
        
        # Check criteria
        if (len(word) >= min_length and 
            word not in stopwords and 
            word.isalpha() and
            not word.isdigit()):
            keywords.append(word)
    
    # Count frequency and return most common
    word_freq = {}
    for word in keywords:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    return [word for word, freq in sorted_keywords[:max_keywords]]

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity based on word overlap"""
    if not text1 or not text2:
        return 0.0
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Chunk text into overlapping segments"""
    if not text or chunk_size <= 0:
        return [text] if text else []
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 200 characters
            search_start = max(end - 200, start)
            sentence_end = -1
            
            for i in range(end, search_start, -1):
                if text[i] in '.!?':
                    sentence_end = i + 1
                    break
            
            if sentence_end > start:
                end = sentence_end
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = max(start + chunk_size - overlap, end)
        
        # Prevent infinite loop
        if start >= len(text):
            break
    
    return chunks

def generate_content_hash(content: str) -> str:
    """Generate hash for content deduplication"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    if len(filename) > 255:
        filename = filename[:255]
    
    return filename.strip()

def format_metrics(metrics: Dict[str, Any]) -> str:
    """Format metrics dictionary as readable string"""
    formatted_lines = []
    
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted_lines.append(f"{key}: {value:.3f}")
        elif isinstance(value, dict):
            formatted_lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, float):
                    formatted_lines.append(f"  {sub_key}: {sub_value:.3f}")
                else:
                    formatted_lines.append(f"  {sub_key}: {sub_value}")
        else:
            formatted_lines.append(f"{key}: {value}")
    
    return "\n".join(formatted_lines)

def parse_roi_metrics(text: str) -> List[Dict[str, Any]]:
    """Parse ROI metrics from text"""
    metrics = []
    
    # Patterns for different types of metrics
    patterns = [
        # Percentage improvements
        (r'(\d+(?:\.\d+)?)%\s*(?:improvement|increase|reduction|savings?|better)', 'percentage'),
        # Dollar amounts
        (r'\$?([\d,]+(?:\.\d{2})?)\s*(?:saved|savings?|reduced?|million|thousand|k)', 'currency'),
        # Time savings
        (r'(\d+(?:\.\d+)?)\s*(?:hours?|minutes?|days?|weeks?|months?)\s*(?:saved|reduced?)', 'time'),
        # Multiplier improvements
        (r'(\d+(?:\.\d+)?)x\s*(?:faster|improvement|increase|better)', 'multiplier'),
        # Efficiency metrics
        (r'(\d+(?:\.\d+)?)%\s*(?:more\s+efficient|efficiency)', 'efficiency')
    ]
    
    for pattern, metric_type in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                value = float(match.replace(',', ''))
                metrics.append({
                    'type': metric_type,
                    'value': value,
                    'text': match
                })
            except ValueError:
                continue
    
    return metrics

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize configuration"""
    validated = {}
    
    # Strategy validation
    valid_strategies = ['naive', 'parent_document', 'hybrid']
    validated['strategy'] = config.get('strategy', 'naive')
    if validated['strategy'] not in valid_strategies:
        validated['strategy'] = 'naive'
    
    # Chunk size validation
    chunk_size = config.get('chunk_size', 1000)
    validated['chunk_size'] = max(100, min(5000, int(chunk_size)))
    
    # Chunk overlap validation
    chunk_overlap = config.get('chunk_overlap', 200)
    validated['chunk_overlap'] = max(0, min(validated['chunk_size'] // 2, int(chunk_overlap)))
    
    # Top-k validation
    top_k = config.get('top_k', 5)
    validated['top_k'] = max(1, min(50, int(top_k)))
    
    # Retrieval method validation
    valid_methods = ['semantic', 'keyword', 'hybrid']
    validated['retrieval_method'] = config.get('retrieval_method', 'semantic')
    if validated['retrieval_method'] not in valid_methods:
        validated['retrieval_method'] = 'semantic'
    
    # Max tokens validation
    max_tokens = config.get('max_tokens', 2048)
    validated['max_tokens'] = max(100, min(8192, int(max_tokens)))
    
    return validated

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def safe_json_serialize(obj: Any) -> str:
    """Safely serialize object to JSON with datetime handling"""
    def json_serializer(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    try:
        return json.dumps(obj, default=json_serializer, indent=2)
    except Exception as e:
        return f"Serialization error: {str(e)}"

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length"""
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def extract_company_info(text: str) -> Dict[str, Any]:
    """Extract company information from text"""
    info = {
        'company_name': '',
        'industry': '',
        'size': '',
        'location': ''
    }
    
    # Simple patterns for company info extraction
    industry_keywords = {
        'construction': ['construction', 'building', 'contractor'],
        'logistics': ['logistics', 'transportation', 'shipping', 'delivery'],
        'education': ['education', 'school', 'university', 'college'],
        'manufacturing': ['manufacturing', 'factory', 'production'],
        'retail': ['retail', 'store', 'shop', 'commerce'],
        'healthcare': ['healthcare', 'hospital', 'medical', 'clinic'],
        'government': ['government', 'municipal', 'city', 'county', 'state'],
        'agriculture': ['agriculture', 'farming', 'farm', 'crop'],
        'energy': ['energy', 'oil', 'gas', 'power', 'utility']
    }
    
    text_lower = text.lower()
    
    # Detect industry
    for industry, keywords in industry_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            info['industry'] = industry
            break
    
    # Extract company size indicators
    if any(term in text_lower for term in ['fortune 500', 'large', 'enterprise']):
        info['size'] = 'large'
    elif any(term in text_lower for term in ['small', 'startup', 'local']):
        info['size'] = 'small'
    elif any(term in text_lower for term in ['medium', 'mid-size']):
        info['size'] = 'medium'
    
    return info
