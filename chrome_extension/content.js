// content.js
// This script runs on every web page and can be used to inject additional functionality
// For now, it's mainly used to help extract content when requested by the popup

// Optional: Add a visual indicator when the page is being analyzed
function showAnalysisIndicator() {
  // Create a temporary indicator
  const indicator = document.createElement('div');
  indicator.id = 'privacy-analyzer-indicator';
  indicator.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 12px 20px;
    border-radius: 8px;
    z-index: 10000;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.2);
  `;
  indicator.innerHTML = '🔍 Analyzing privacy policy...';
  
  document.body.appendChild(indicator);
  
  // Remove after 3 seconds
  setTimeout(() => {
    const el = document.getElementById('privacy-analyzer-indicator');
    if (el) el.remove();
  }, 3000);
}

// Listen for messages from the popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'showIndicator') {
    showAnalysisIndicator();
    sendResponse({ success: true });
  }
});

// Optional: Auto-detect privacy policy pages and show a subtle indicator
function detectPrivacyPage() {
  const url = window.location.href.toLowerCase();
  const title = document.title.toLowerCase();
  const content = document.body.textContent.toLowerCase();
  
  const privacyKeywords = [
    'privacy policy', 'privacy notice', 'data policy', 
    'terms of service', 'terms and conditions', 'cookie policy'
  ];
  
  const isPrivacyPage = privacyKeywords.some(keyword => 
    url.includes(keyword.replace(' ', '-')) || 
    url.includes(keyword.replace(' ', '')) ||
    title.includes(keyword) ||
    content.includes(keyword)
  );
  
  if (isPrivacyPage) {
    // Add a subtle badge to indicate this page can be analyzed
    const badge = document.createElement('div');
    badge.id = 'privacy-analyzer-badge';
    badge.style.cssText = `
      position: fixed;
      bottom: 20px;
      right: 20px;
      background: rgba(102, 126, 234, 0.9);
      color: white;
      padding: 8px 12px;
      border-radius: 20px;
      z-index: 9999;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      font-size: 12px;
      cursor: pointer;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
      transition: all 0.3s ease;
    `;
    badge.innerHTML = '🔍 Privacy Policy Detected';
    badge.title = 'Click the extension icon to analyze this page';
    
    badge.addEventListener('mouseenter', () => {
      badge.style.transform = 'scale(1.05)';
    });
    
    badge.addEventListener('mouseleave', () => {
      badge.style.transform = 'scale(1)';
    });
    
    document.body.appendChild(badge);
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
      const el = document.getElementById('privacy-analyzer-badge');
      if (el) {
        el.style.opacity = '0';
        setTimeout(() => el.remove(), 300);
      }
    }, 5000);
  }
}

// Run detection when page loads
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', detectPrivacyPage);
} else {
  detectPrivacyPage();
}