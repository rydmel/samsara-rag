import requests
import os
from typing import List, Dict, Any
import json
import re
from bs4 import BeautifulSoup
import streamlit as st
import time

class SamsaraCustomerScraper:
    """Scraper for Samsara customer stories"""
    
    def __init__(self):
        self.base_url = "https://www.samsara.com/customers"
        self.customer_stories = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def scrape_customer_stories(self) -> List[Dict[str, Any]]:
        """Scrape customer stories from Samsara website"""
        try:
            return self._scrape_stories()
        except Exception as e:
            st.error(f"Error during scraping: {str(e)}")
            return []
    
    def _scrape_stories(self) -> List[Dict[str, Any]]:
        """Scrape customer stories using requests"""
        stories = []
        
        try:
            # First, scrape the main customers page
            st.info("Fetching main customers page...")
            response = self.session.get(self.base_url, timeout=30)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract customer story links and basic info
                customer_links = self._extract_customer_links(soup)
                
                st.info(f"Found {len(customer_links)} customer story links")
                
                # Process all customer stories
                for i, link_info in enumerate(customer_links, 1):
                    st.text(f"Processing customer story {i}/{len(customer_links)}")
                    
                    try:
                        time.sleep(0.5)  # Be polite to the server
                        story_response = self.session.get(link_info['url'], timeout=30)
                        
                        if story_response.status_code == 200:
                            story_data = self._extract_story_content(
                                story_response.content, 
                                link_info
                            )
                            if story_data:
                                stories.append(story_data)
                    
                    except Exception as e:
                        st.warning(f"Failed to scrape {link_info['url']}: {str(e)}")
                        continue
            
            else:
                st.error(f"Failed to scrape main customers page. Status code: {response.status_code}")
            
        except Exception as e:
            st.error(f"Scraping error: {str(e)}")
        
        return stories
    
    def _extract_customer_links(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract customer story links from the main page"""
        links = []
        
        # Look for customer story links - these patterns may need adjustment based on actual HTML structure
        customer_selectors = [
            'a[href*="/customers/"]',
            '.customer-story a',
            '.case-study a',
            '[data-testid*="customer"] a'
        ]
        
        for selector in customer_selectors:
            elements = soup.select(selector)
            for element in elements:
                href = element.get('href')
                if href and isinstance(href, str) and '/customers/' in href and href != '/customers':
                    # Convert relative URLs to absolute
                    if href.startswith('/'):
                        href = f"https://www.samsara.com{href}"
                    
                    title = element.get_text(strip=True) or element.get('title', '')
                    
                    if href not in [link['url'] for link in links]:
                        links.append({
                            'url': href,
                            'title': title
                        })
        
        # If no specific customer links found, try to find them in the page content
        if not links:
            # Look for any links that might be customer stories
            all_links = soup.find_all('a', href=True)
            for link in all_links:
                href = link['href']
                if '/customers/' in href and href != '/customers':
                    if href.startswith('/'):
                        href = f"https://www.samsara.com{href}"
                    
                    title = link.get_text(strip=True)
                    if title and len(title) > 5:  # Only meaningful titles
                        links.append({
                            'url': href,
                            'title': title
                        })
        
        # Remove duplicates
        unique_links = []
        seen_urls = set()
        for link in links:
            if link['url'] not in seen_urls:
                unique_links.append(link)
                seen_urls.add(link['url'])
        
        return unique_links
    
    def _extract_story_content(self, html_content: bytes, link_info: Dict[str, str]) -> Dict[str, Any]:
        """Extract content from individual customer story pages"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        story_data = {
            'url': link_info['url'],
            'title': link_info['title'],
            'company_name': self._extract_company_name(soup, link_info['title']),
            'industry': self._extract_industry(soup),
            'content': self._extract_main_content(soup),
            'highlights': self._extract_highlights(soup),
            'roi_metrics': self._extract_roi_metrics(soup),
            'challenges': self._extract_challenges(soup),
            'solutions': self._extract_solutions(soup),
            'competitor_info': self._extract_competitor_info(soup)
        }
        
        # Only return if we have substantial content
        if len(story_data['content']) > 200:
            return story_data
        
        return story_data
    
    def _extract_company_name(self, soup: BeautifulSoup, title: str) -> str:
        """Extract company name from the page"""
        # Try multiple selectors for company name
        selectors = [
            'h1',
            '.company-name',
            '.customer-name',
            '[data-testid*="company"]'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                name = element.get_text(strip=True)
                if name and len(name) < 100:  # Reasonable company name length
                    return name
        
        # Fallback to title
        return title if title else "Unknown Company"
    
    def _extract_industry(self, soup: BeautifulSoup) -> str:
        """Extract industry information"""
        industry_keywords = [
            'construction', 'logistics', 'education', 'transportation', 
            'manufacturing', 'retail', 'healthcare', 'government',
            'agriculture', 'energy', 'utilities', 'food', 'beverage'
        ]
        
        text_content = soup.get_text().lower()
        
        for keyword in industry_keywords:
            if keyword in text_content:
                return keyword.capitalize()
        
        return "Other"
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from the page"""
        content_selectors = [
            '.story-content',
            '.case-study-content',
            '.customer-story',
            'main',
            '.content',
            'article'
        ]
        
        content = ""
        
        # Try specific content selectors first
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                content = element.get_text(strip=True)
                if len(content) > 200:
                    break
        
        # If no specific content found, get all text from body
        if len(content) < 200:
            body = soup.find('body')
            if body:
                content = body.get_text(strip=True)
        
        # Clean up the content
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n+', '\n', content)
        
        return content
    
    def _extract_highlights(self, soup: BeautifulSoup) -> List[str]:
        """Extract key highlights or benefits"""
        highlights = []
        
        # Look for bullet points, highlights, or benefits sections
        highlight_selectors = [
            '.highlights li',
            '.benefits li',
            '.key-points li',
            'ul li',
            '.feature li'
        ]
        
        for selector in highlight_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(strip=True)
                if text and len(text) > 10 and len(text) < 200:
                    highlights.append(text)
        
        return highlights[:10]  # Limit to 10 highlights
    
    def _extract_roi_metrics(self, soup: BeautifulSoup) -> List[str]:
        """Extract ROI and performance metrics"""
        metrics = []
        text_content = soup.get_text()
        
        # Look for percentage improvements, cost savings, time reductions
        patterns = [
            r'(\d+%\s*(?:improvement|increase|reduction|savings?))',
            r'(\$[\d,]+\s*(?:saved|savings?|reduced?))',
            r'(\d+\s*hours?\s*(?:saved|reduced?))',
            r'(\d+x\s*(?:faster|improvement|increase))',
            r'(reduced?\s*(?:by\s*)?\d+%)',
            r'(increased?\s*(?:by\s*)?\d+%)',
            r'(saved?\s*\$[\d,]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text_content, re.IGNORECASE)
            for match in matches:
                if match not in metrics:
                    metrics.append(match)
        
        return metrics[:5]  # Limit to 5 metrics
    
    def _extract_challenges(self, soup: BeautifulSoup) -> List[str]:
        """Extract challenges mentioned in the story"""
        challenges = []
        text_content = soup.get_text().lower()
        
        # Look for sections about challenges or problems
        challenge_keywords = [
            'challenge', 'problem', 'issue', 'difficulty', 'struggle',
            'pain point', 'obstacle', 'barrier'
        ]
        
        sentences = text_content.split('.')
        for sentence in sentences:
            for keyword in challenge_keywords:
                if keyword in sentence and len(sentence) > 20:
                    challenges.append(sentence.strip())
                    break
        
        return challenges[:3]  # Limit to 3 challenges
    
    def _extract_solutions(self, soup: BeautifulSoup) -> List[str]:
        """Extract solutions or implementations"""
        solutions = []
        text_content = soup.get_text().lower()
        
        solution_keywords = [
            'solution', 'implemented', 'deployed', 'installed',
            'used samsara', 'adopted', 'leveraged'
        ]
        
        sentences = text_content.split('.')
        for sentence in sentences:
            for keyword in solution_keywords:
                if keyword in sentence and len(sentence) > 20:
                    solutions.append(sentence.strip())
                    break
        
        return solutions[:3]  # Limit to 3 solutions
    
    def _extract_competitor_info(self, soup: BeautifulSoup) -> str:
        """Extract information about competitors they switched from"""
        text_content = soup.get_text().lower()
        
        competitor_patterns = [
            r'switched from (\w+)',
            r'replaced (\w+)',
            r'moved from (\w+)',
            r'previously used (\w+)',
            r'migrated from (\w+)'
        ]
        
        for pattern in competitor_patterns:
            match = re.search(pattern, text_content)
            if match:
                return match.group(1)
        
        return ""
