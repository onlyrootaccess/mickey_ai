# Selenium-based browser control
"""
Mickey AI - Browser Automator
Automates browser actions like opening URLs, searching, form filling, etc.
"""

import logging
import time
import random
from typing import Dict, List, Any, Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

class BrowserAutomator:
    def __init__(self, headless: bool = False):
        self.logger = logging.getLogger(__name__)
        self.driver = None
        self.headless = headless
        self.current_url = None
        self.browser_state = "closed"
        
        # Browser configuration
        self.chrome_options = Options()
        if headless:
            self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        self.chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Mickey's browser personalities
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        ]
        
        self.logger.info("🌐 Browser Automator initialized - Ready to surf!")

    def initialize_browser(self) -> bool:
        """Initialize the Chrome browser instance"""
        try:
            self.logger.info("Starting Chrome browser...")
            
            # Add random user agent
            self.chrome_options.add_argument(f"--user-agent={random.choice(self.user_agents)}")
            
            # Initialize driver
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=self.chrome_options)
            
            # Execute script to remove webdriver property
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            self.browser_state = "ready"
            self.logger.info("✅ Browser initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize browser: {str(e)}")
            self.browser_state = "error"
            return False

    def open_url(self, url: str, wait_time: int = 5) -> Dict[str, Any]:
        """
        Open a URL in the browser
        
        Args:
            url: URL to open
            wait_time: Time to wait for page load
            
        Returns:
            Dictionary with result details
        """
        try:
            if not self.driver:
                if not self.initialize_browser():
                    return self._create_error_response("Browser not initialized")
            
            self.logger.info(f"Opening URL: {url}")
            
            # Ensure URL has protocol
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            self.driver.get(url)
            self.current_url = url
            
            # Wait for page to load
            time.sleep(wait_time)
            
            # Get page title
            page_title = self.driver.title
            
            self.logger.info(f"Successfully opened: {page_title}")
            
            return {
                'success': True,
                'url': url,
                'title': page_title,
                'message': f"Opened {page_title}",
                'mickey_response': random.choice([
                    "Website opened! Mickey's surfing the web! 🏄‍♂️",
                    "Voila! Page loaded successfully!",
                    "Mickey's browsing magic worked! ✨",
                    "Hot dog! We're on the web!"
                ])
            }
            
        except Exception as e:
            self.logger.error(f"Failed to open URL {url}: {str(e)}")
            return self._create_error_response(f"Failed to open URL: {str(e)}")

    def search(self, query: str, search_engine: str = "google") -> Dict[str, Any]:
        """
        Perform a web search
        
        Args:
            query: Search query
            search_engine: Search engine to use (google, bing, duckduckgo)
            
        Returns:
            Dictionary with search results
        """
        try:
            if not self.driver:
                if not self.initialize_browser():
                    return self._create_error_response("Browser not initialized")
            
            search_urls = {
                "google": "https://www.google.com",
                "bing": "https://www.bing.com",
                "duckduckgo": "https://duckduckgo.com"
            }
            
            if search_engine not in search_urls:
                return self._create_error_response(f"Unsupported search engine: {search_engine}")
            
            # Navigate to search engine
            self.open_url(search_urls[search_engine])
            
            # Find search box and enter query
            if search_engine == "google":
                search_box = self.driver.find_element(By.NAME, "q")
            elif search_engine == "bing":
                search_box = self.driver.find_element(By.NAME, "q")
            else:  # duckduckgo
                search_box = self.driver.find_element(By.NAME, "q")
            
            search_box.clear()
            search_box.send_keys(query)
            search_box.send_keys(Keys.RETURN)
            
            # Wait for results
            time.sleep(3)
            
            # Get first few results
            results = self._extract_search_results(search_engine)
            
            self.logger.info(f"Search completed for: {query}")
            
            return {
                'success': True,
                'query': query,
                'search_engine': search_engine,
                'results': results,
                'message': f"Found {len(results)} results for '{query}'",
                'mickey_response': random.choice([
                    f"Mickey found {len(results)} results for you! 🎯",
                    "Search complete! Time to explore!",
                    "Voila! Your search results are ready!",
                    "Mickey's magic search worked! ✨"
                ])
            }
            
        except Exception as e:
            self.logger.error(f"Search failed for '{query}': {str(e)}")
            return self._create_error_response(f"Search failed: {str(e)}")

    def _extract_search_results(self, search_engine: str) -> List[Dict[str, str]]:
        """Extract search results from the page"""
        results = []
        
        try:
            if search_engine == "google":
                # Google results
                result_elements = self.driver.find_elements(By.CSS_SELECTOR, "div.g")
                for i, element in enumerate(result_elements[:5]):  # First 5 results
                    try:
                        title_element = element.find_element(By.CSS_SELECTOR, "h3")
                        link_element = element.find_element(By.CSS_SELECTOR, "a")
                        
                        results.append({
                            'title': title_element.text,
                            'url': link_element.get_attribute("href"),
                            'description': element.text[:200] + "..." if len(element.text) > 200 else element.text
                        })
                    except:
                        continue
                        
            elif search_engine == "bing":
                # Bing results
                result_elements = self.driver.find_elements(By.CSS_SELECTOR, "li.b_algo")
                for i, element in enumerate(result_elements[:5]):
                    try:
                        title_element = element.find_element(By.CSS_SELECTOR, "h2")
                        link_element = element.find_element(By.CSS_SELECTOR, "a")
                        
                        results.append({
                            'title': title_element.text,
                            'url': link_element.get_attribute("href"),
                            'description': element.text[:200] + "..." if len(element.text) > 200 else element.text
                        })
                    except:
                        continue
        
        except Exception as e:
            self.logger.warning(f"Could not extract search results: {str(e)}")
        
        return results

    def fill_form(self, form_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Fill a web form with provided data
        
        Args:
            form_data: Dictionary with field names and values
            
        Returns:
            Dictionary with form filling result
        """
        try:
            if not self.driver:
                return self._create_error_response("Browser not open")
            
            filled_fields = 0
            errors = []
            
            for field_name, value in form_data.items():
                try:
                    # Try different selectors
                    selectors = [
                        f"input[name='{field_name}']",
                        f"textarea[name='{field_name}']",
                        f"input[placeholder*='{field_name}']",
                        f"#{field_name}",
                        f"input[id='{field_name}']"
                    ]
                    
                    field_found = False
                    for selector in selectors:
                        try:
                            element = self.driver.find_element(By.CSS_SELECTOR, selector)
                            element.clear()
                            element.send_keys(value)
                            filled_fields += 1
                            field_found = True
                            break
                        except:
                            continue
                    
                    if not field_found:
                        errors.append(f"Field '{field_name}' not found")
                        
                except Exception as e:
                    errors.append(f"Error filling '{field_name}': {str(e)}")
            
            self.logger.info(f"Filled {filled_fields} form fields")
            
            return {
                'success': filled_fields > 0,
                'filled_fields': filled_fields,
                'total_fields': len(form_data),
                'errors': errors,
                'message': f"Filled {filled_fields} out of {len(form_data)} fields",
                'mickey_response': random.choice([
                    f"Mickey filled {filled_fields} form fields for you! 📝",
                    "Form filling complete! Ready to submit?",
                    "Voila! Your form is filled with Mickey magic! ✨"
                ]) if filled_fields > 0 else "Oops! Mickey couldn't find the form fields! 😅"
            }
            
        except Exception as e:
            self.logger.error(f"Form filling failed: {str(e)}")
            return self._create_error_response(f"Form filling failed: {str(e)}")

    def click_element(self, selector: str, selector_type: str = "css") -> Dict[str, Any]:
        """
        Click on a web element
        
        Args:
            selector: CSS selector or XPath
            selector_type: 'css' or 'xpath'
            
        Returns:
            Dictionary with click result
        """
        try:
            if not self.driver:
                return self._create_error_response("Browser not open")
            
            if selector_type == "css":
                element = self.driver.find_element(By.CSS_SELECTOR, selector)
            else:  # xpath
                element = self.driver.find_element(By.XPATH, selector)
            
            element.click()
            
            self.logger.info(f"Clicked element: {selector}")
            
            return {
                'success': True,
                'selector': selector,
                'message': "Element clicked successfully",
                'mickey_response': random.choice([
                    "Click! Mickey pressed the button! 🔘",
                    "Element clicked! Magic happening! ✨",
                    "Voila! Mickey worked the click magic!",
                    "Hot dog! That element got clicked! 🌭"
                ])
            }
            
        except Exception as e:
            self.logger.error(f"Click failed for selector '{selector}': {str(e)}")
            return self._create_error_response(f"Click failed: {str(e)}")

    def take_screenshot(self, filename: str = None) -> Dict[str, Any]:
        """
        Take a screenshot of the current page
        
        Args:
            filename: Optional filename to save screenshot
            
        Returns:
            Dictionary with screenshot result
        """
        try:
            if not self.driver:
                return self._create_error_response("Browser not open")
            
            if not filename:
                timestamp = int(time.time())
                filename = f"screenshot_{timestamp}.png"
            
            # Ensure screenshots directory exists
            import os
            os.makedirs("screenshots", exist_ok=True)
            
            filepath = os.path.join("screenshots", filename)
            self.driver.save_screenshot(filepath)
            
            self.logger.info(f"Screenshot saved: {filepath}")
            
            return {
                'success': True,
                'filepath': filepath,
                'message': f"Screenshot saved as {filename}",
                'mickey_response': random.choice([
                    "Cheese! Mickey took a screenshot! 📸",
                    "Screenshot captured! Memories saved!",
                    "Voila! Page screenshot ready! 🖼️",
                    "Mickey's camera magic worked! ✨"
                ])
            }
            
        except Exception as e:
            self.logger.error(f"Screenshot failed: {str(e)}")
            return self._create_error_response(f"Screenshot failed: {str(e)}")

    def get_page_info(self) -> Dict[str, Any]:
        """
        Get information about the current page
        
        Returns:
            Dictionary with page information
        """
        try:
            if not self.driver:
                return self._create_error_response("Browser not open")
            
            title = self.driver.title
            current_url = self.driver.current_url
            source = self.driver.page_source[:500] + "..." if len(self.driver.page_source) > 500 else self.driver.page_source
            
            return {
                'success': True,
                'title': title,
                'url': current_url,
                'source_preview': source,
                'message': f"Page info for {title}",
                'mickey_response': f"Mickey's viewing {title}! 📄"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get page info: {str(e)}")
            return self._create_error_response(f"Failed to get page info: {str(e)}")

    def close_browser(self) -> Dict[str, Any]:
        """
        Close the browser
        
        Returns:
            Dictionary with close result
        """
        try:
            if self.driver:
                self.driver.quit()
                self.driver = None
                self.browser_state = "closed"
                self.current_url = None
                
                self.logger.info("Browser closed")
                
                return {
                    'success': True,
                    'message': "Browser closed successfully",
                    'mickey_response': random.choice([
                        "Browser closed! Mickey's taking a break! ☕",
                        "All done! Browser shut down!",
                        "Mickey signed off from browsing! 👋",
                        "Hot dog! That was a productive session! 🌭"
                    ])
                }
            else:
                return {
                    'success': True,
                    'message': "Browser was already closed",
                    'mickey_response': "Mickey says the browser was already closed! 😊"
                }
                
        except Exception as e:
            self.logger.error(f"Failed to close browser: {str(e)}")
            return self._create_error_response(f"Failed to close browser: {str(e)}")

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'success': False,
            'error': error_message,
            'mickey_response': random.choice([
                "Oops! Mickey slipped on a banana peel while browsing! 🍌",
                "Uh oh! Browser magic didn't work this time!",
                "Mickey's having trouble with the web! 😅",
                "Hot dog! Something went wrong with the browser! 🌭"
            ])
        }

    def get_browser_status(self) -> Dict[str, Any]:
        """Get current browser status"""
        return {
            'browser_state': self.browser_state,
            'current_url': self.current_url,
            'headless_mode': self.headless,
            'driver_initialized': self.driver is not None
        }

# Test function
def test_browser_automator():
    """Test the browser automator"""
    automator = BrowserAutomator(headless=True)  # Use headless for testing
    
    try:
        # Test URL opening
        result = automator.open_url("https://www.google.com")
        print("Open URL Result:", result)
        
        # Test page info
        info = automator.get_page_info()
        print("Page Info:", info)
        
        # Test search
        search_result = automator.search("Mickey Mouse")
        print("Search Result:", search_result)
        
        # Test screenshot
        screenshot_result = automator.take_screenshot("test_screenshot.png")
        print("Screenshot Result:", screenshot_result)
        
    finally:
        # Close browser
        close_result = automator.close_browser()
        print("Close Result:", close_result)

if __name__ == "__main__":
    test_browser_automator()