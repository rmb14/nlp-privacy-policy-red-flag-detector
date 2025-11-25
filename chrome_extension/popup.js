// popup.js
document.addEventListener('DOMContentLoaded', function() {
  const analyzeBtn = document.getElementById('analyzeBtn');
  const clearBtn = document.getElementById('clearBtn');
  const status = document.getElementById('status');
  const results = document.getElementById('results');

  // API configuration
  const API_URL = 'http://localhost:8000/classify';

  analyzeBtn.addEventListener('click', analyzeCurrentPage);
  clearBtn.addEventListener('click', clearResults);

  async function analyzeCurrentPage() {
    try {
      setStatus('loading', 'Extracting page content...');
      analyzeBtn.disabled = true;

      // Get current tab
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      
      // Extract text content from the page
      const [result] = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        function: extractPageText
      });

      const pageText = result.result;
      
      if (!pageText || pageText.trim().length === 0) {
        setStatus('error', 'No text content found on this page');
        return;
      }

      setStatus('loading', 'Analyzing content with AI model...');

      // Send to API for analysis
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: pageText })
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status} ${response.statusText}`);
      }

      const analysisResults = await response.json();
      
      if (analysisResults.error) {
        throw new Error(analysisResults.error);
      }

      // Handle the new response format with results property
      const results = analysisResults.results || analysisResults;
      const summary = analysisResults.summary || {};
      
      displayResults(results, summary);
      
      if (results.length === 0) {
        setStatus('success', `No concerning clauses detected on this page`);
      } else {
        setStatus('success', `Found ${results.length} concerning clause(s)`);
      }

    } catch (error) {
      console.error('Analysis failed:', error);
      let errorMessage = 'Analysis failed';
      
      if (error.message.includes('Failed to fetch')) {
        errorMessage = 'Cannot connect to API. Make sure main.py is running on localhost:8000';
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      setStatus('error', errorMessage);
    } finally {
      analyzeBtn.disabled = false;
    }
  }

  function extractPageText() {
    // Remove script and style elements
    const scripts = document.querySelectorAll('script, style, noscript');
    scripts.forEach(el => el.remove());

    // Get text content, prioritizing main content areas
    const contentSelectors = [
      'main',
      'article', 
      '.content',
      '.privacy-policy',
      '.terms-of-service',
      '.legal',
      '#content',
      '#main',
      '.policy',
      'body'
    ];

    let text = '';
    
    for (const selector of contentSelectors) {
      const element = document.querySelector(selector);
      if (element) {
        text = element.innerText || element.textContent || '';
        if (text.trim().length > 500) { // Found substantial content
          break;
        }
      }
    }

    // Fallback to body if nothing substantial found
    if (!text || text.trim().length < 500) {
      text = document.body.innerText || document.body.textContent || '';
    }

    // Clean up the text
    return text
      .replace(/\\s+/g, ' ') // Replace multiple whitespace with single space
      .replace(/\\n+/g, '\\n') // Replace multiple newlines with single newline
      .trim();
  }

  function displayResults(analysisResults, summary = {}) {
    results.style.display = 'block';
    
    if (analysisResults.length === 0) {
      results.innerHTML = '<div class="no-results">No concerning clauses detected 🎉</div>';
      return;
    }

    // Create summary
    const summaryDiv = document.createElement('div');
    summaryDiv.className = 'summary';
    summaryDiv.innerHTML = `
      <strong>Analysis Summary:</strong><br>
      Found ${analysisResults.length} concerning clause(s) in the privacy policy
      ${summary.total_sentences ? `<br>Analyzed ${summary.total_sentences} total sentences` : ''}
      ${summary.filtered_out_false_positives ? `<br>Filtered out ${summary.filtered_out_false_positives} false positives` : ''}
      ${summary.threshold_used ? `<br>Confidence threshold: ${(summary.threshold_used * 100).toFixed(0)}%` : ''}
    `;

    // Create results HTML
    const resultItems = analysisResults.map(item => {
      const severity = item.severity || getSeverity(item.confidence);
      const labelDisplay = item.formatted_label || formatLabel(item.label);
      const confidencePercent = Math.round(item.confidence * 100);
      
      return `
        <div class="result-item ${severity}">
          <div class="sentence">${escapeHtml(item.sentence)}</div>
          <div class="metadata">
            <span class="label">${labelDisplay}</span>
            <span class="confidence">${confidencePercent}% confidence</span>
          </div>
        </div>
      `;
    }).join('');

    results.innerHTML = summaryDiv.outerHTML + resultItems;
  }

  function getSeverity(confidence) {
    if (confidence >= 0.9) return 'high';
    if (confidence >= 0.8) return 'medium';
    return 'low';
  }

  function formatLabel(label) {
    // Convert label to more readable format
    return label
      .replace(/_/g, ' ')
      .replace(/\\b\\w/g, l => l.toUpperCase());
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  function setStatus(type, message) {
    status.className = `status ${type}`;
    
    if (type === 'loading') {
      status.innerHTML = `<span class="loading-spinner"></span>${message}`;
    } else {
      status.textContent = message;
    }
    
    status.style.display = 'block';
    
    // Auto-hide success/error messages after 3 seconds
    if (type !== 'loading') {
      setTimeout(() => {
        status.style.display = 'none';
      }, 3000);
    }
  }

  function clearResults() {
    results.innerHTML = '';
    results.style.display = 'none';
    status.style.display = 'none';
  }
});