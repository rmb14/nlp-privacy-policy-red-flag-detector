// background.js
// Service worker for the Chrome extension

// Handle installation
chrome.runtime.onInstalled.addListener(() => {
  console.log('Privacy Policy Analyzer extension installed');
  
  // Create context menu item for right-click analysis
  try {
    chrome.contextMenus.create({
      id: 'analyzePage',
      title: 'Analyze Privacy Policy',
      contexts: ['page']
    }, () => {
      if (chrome.runtime.lastError) {
        console.log('Context menu creation error:', chrome.runtime.lastError);
      } else {
        console.log('Context menu created successfully');
      }
    });
  } catch (error) {
    console.log('Context menu not supported:', error);
  }
});

// Optional: Handle extension icon click
chrome.action.onClicked.addListener((tab) => {
  // This will open the popup, which is handled by the manifest
  console.log('Extension icon clicked on tab:', tab.url);
});

// Optional: Handle messages from content script or popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'analyze') {
    // Handle analysis request if needed
    console.log('Analysis requested for:', sender.tab?.url);
    sendResponse({ success: true });
  }
});

// Handle context menu clicks - with error checking
if (chrome.contextMenus && chrome.contextMenus.onClicked) {
  chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === 'analyzePage') {
      // Open the popup or trigger analysis
      try {
        chrome.action.openPopup();
      } catch (error) {
        console.log('Cannot open popup programmatically:', error);
        // Fallback: send message to content script
        chrome.tabs.sendMessage(tab.id, { action: 'showIndicator' });
      }
    }
  });
}