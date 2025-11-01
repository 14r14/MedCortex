import uuid
from dotenv import load_dotenv
import streamlit as st

from app.config import Settings
from app.rag.pipeline import IngestionPipeline, QueryPipeline


def inject_custom_css():
    """Inject custom CSS for medical research UI with IBM brand colors."""
    st.markdown(
        r"""
        <style>
        /* IBM Medical Research UI Color Palette */
        
        /* Light Mode Colors */
        :root {
            --ibm-blue: #0f62fe;
            --dark-slate: #394a59;
            --verification-green: #24a148;
            --light-gray: #f0f2f6;
            --dark-gray: #161616;
            --medium-gray: #525252;
        }
        
        /* Dark Mode Colors */
        @media (prefers-color-scheme: dark) {
            :root {
                --ibm-blue: #3d8bff;
                --dark-slate: #6f7d8a;
                --verification-green: #42be65;
                --light-gray: #1a1a1a;
                --dark-gray: #f4f4f4;
                --medium-gray: #a8a8a8;
            }
        }
        
        /* Override Streamlit default styles */
        .stApp {
            background-color: var(--light-gray);
        }
        
        /* Main container */
        .main .block-container {
            background-color: var(--light-gray);
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Title styling - VeriCite branding */
        h1 {
            color: var(--dark-gray) !important;
            font-weight: 700;
            font-size: 2.5rem;
            letter-spacing: -0.02em;
        }
        
        /* VeriCite logo/title accent - removed icon for professional look */
        
        /* Caption/subtitle */
        .stMarkdown p {
            color: var(--medium-gray);
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: var(--dark-slate);
        }
        
        [data-testid="stSidebar"] {
            background-color: var(--dark-slate);
        }
        
        /* Sidebar title - VeriCite */
        [data-testid="stSidebar"] h1 {
            color: #ffffff !important;
            font-weight: 700;
            font-size: 1.75rem;
            letter-spacing: -0.01em;
            margin-bottom: 0.5rem !important;
        }
        
        [data-testid="stSidebar"] .stMarkdown h2,
        [data-testid="stSidebar"] .stMarkdown h3 {
            color: #ffffff;
        }
        
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] p {
            color: #e5e5e5;
        }
        
        /* Sidebar dividers */
        [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.2) !important;
            margin: 1rem 0 !important;
        }
        
        /* Sidebar document items */
        [data-testid="stSidebar"] .stMarkdown strong {
            color: #ffffff !important;
            font-weight: 600;
        }
        
        [data-testid="stSidebar"] .stMarkdown .stCaption {
            color: rgba(255, 255, 255, 0.7) !important;
            font-size: 0.875rem;
        }
        
        /* Sidebar info box */
        [data-testid="stSidebar"] .stInfo {
            background-color: rgba(255, 255, 255, 0.1) !important;
            border-left: 3px solid var(--ibm-blue) !important;
        }
        
        /* User chat message - IBM Blue */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="user"]) [data-testid="stChatMessageContent"] {
            background-color: var(--ibm-blue);
            color: #ffffff;
            border-radius: 12px 12px 0 12px;
            padding: 12px 16px;
        }
        
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="user"]) [data-testid="stChatMessageContent"] p {
            color: #ffffff !important;
        }
        
        /* Assistant chat message - Dark Slate */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stChatMessageContent"] {
            background-color: var(--dark-slate);
            color: #ffffff !important;
            border-radius: 12px 12px 12px 0;
            padding: 12px 16px;
        }
        
        /* Ensure all text in assistant messages is white */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stChatMessageContent"] p,
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stChatMessageContent"] div,
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stChatMessageContent"] span,
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stChatMessageContent"] li,
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stChatMessageContent"] ul,
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stChatMessageContent"] ol,
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stChatMessageContent"] h1,
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stChatMessageContent"] h2,
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stChatMessageContent"] h3,
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stChatMessageContent"] h4,
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stChatMessageContent"] h5,
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stChatMessageContent"] h6 {
            color: #ffffff !important;
        }
        
        /* Bold/strong text in assistant messages */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stChatMessageContent"] strong,
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stChatMessageContent"] b {
            color: #ffffff !important;
        }
        
        /* Links in assistant messages - white for maximum contrast */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stChatMessageContent"] a {
            color: #ffffff !important;
            text-decoration: underline !important;
        }
        
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stChatMessageContent"] a:hover {
            color: #ffffff !important;
            opacity: 0.9 !important;
        }
        
        /* Buttons */
        .stButton > button {
            background-color: var(--ibm-blue);
            color: #ffffff;
            border: none;
            border-radius: 6px;
            font-weight: 500;
            transition: background-color 0.2s;
            white-space: nowrap !important;
        }
        
        /* Upload More Documents button - fixed width and padding */
        [data-testid="stButton"] > button {
            min-width: fit-content !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }
        
        .stButton > button:hover {
            background-color: #0050e6;
            color: #ffffff;
        }
        
        /* Headers with dividers - compact */
        h2 {
            color: var(--dark-gray) !important;
            font-weight: 700 !important;
            margin-top: 1rem !important;
            margin-bottom: 0.75rem !important;
            font-size: 1.5rem !important;
        }
        
        /* Horizontal dividers - compact */
        hr {
            border-color: var(--ibm-blue) !important;
            border-width: 2px !important;
            margin: 1rem 0 !important;
        }
        
        /* Info messages */
        .stInfo {
            background-color: rgba(15, 98, 254, 0.1) !important;
            border-left: 4px solid var(--ibm-blue) !important;
            padding: 1rem !important;
            border-radius: 6px !important;
        }
        
        /* File uploader - Compact styling */
        [data-testid="stFileUploader"] {
            border: 2px dashed var(--ibm-blue) !important;
            border-radius: 8px !important;
            background: rgba(15, 98, 254, 0.03) !important;
            padding: 1rem !important;
            transition: all 0.2s ease !important;
            min-height: 60px !important;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: var(--ibm-blue) !important;
            background: rgba(15, 98, 254, 0.05) !important;
            box-shadow: 0 2px 8px rgba(15, 98, 254, 0.1) !important;
        }
        
        /* Drag and drop overlay */
        .drag-drop-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(15, 98, 254, 0.1);
            backdrop-filter: blur(4px);
            z-index: 9999;
            display: none;
            align-items: center;
            justify-content: center;
            border: 4px dashed var(--ibm-blue);
            pointer-events: none;
        }
        
        .drag-drop-overlay.active {
            display: flex;
        }
        
        .drag-drop-overlay-content {
            background: #ffffff;
            padding: 3rem 4rem;
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            text-align: center;
            border: 2px solid var(--ibm-blue);
        }
        
        .drag-drop-overlay-content h2 {
            color: var(--ibm-blue);
            font-size: 2rem;
            margin: 0 0 1rem 0;
            font-weight: 700;
        }
        
        .drag-drop-overlay-content p {
            color: var(--medium-gray);
            font-size: 1.1rem;
            margin: 0;
        }
        
        .drag-drop-overlay-content .icon {
            font-size: 4rem;
            margin-bottom: 1rem;
        }
        
        /* Selected files preview */
        h3 {
            color: var(--dark-gray) !important;
            font-weight: 600 !important;
            font-size: 1.2rem !important;
            margin-top: 1.5rem !important;
            margin-bottom: 1rem !important;
        }
        
        /* File uploader label - compact */
        [data-testid="stFileUploader"] label {
            color: var(--dark-gray) !important;
            font-weight: 500 !important;
            font-size: 0.95rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* File name container - simple, clean styling */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] {
            background-color: #ffffff !important;
            border: 1px solid var(--medium-gray) !important;
            border-radius: 6px !important;
            padding: 0.5rem 0.75rem !important;
            margin: 0.5rem 0 !important;
            transition: all 0.15s ease !important;
            display: flex !important;
            align-items: center !important;
            justify-content: space-between !important;
            gap: 0.5rem !important;
            position: relative !important;
        }
        
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"]:hover {
            background-color: #f8f9fa !important;
            border-color: var(--ibm-blue) !important;
        }
        
        /* File name text - simple */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] > *:first-child,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] span:first-of-type,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] div:first-of-type {
            color: var(--dark-gray) !important;
            font-weight: 400 !important;
            font-size: 0.875rem !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            flex: 1 !important;
            min-width: 0 !important;
        }
        
        /* File size - simple, aligned to right */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFileSize"],
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] + [data-testid="stFileUploaderFileSize"],
        [data-testid="stFileUploader"] span[data-testid="stFileUploaderFileSize"],
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] ~ span,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] ~ div,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] > [data-testid="stFileUploaderFileSize"],
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] span:last-child,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] div:last-child:not(:first-child) {
            color: var(--medium-gray) !important;
            font-size: 0.8rem !important;
            font-weight: 400 !important;
            margin-left: auto !important;
            margin-right: 0.5rem !important;
            white-space: nowrap !important;
            flex-shrink: 0 !important;
        }
        
        /* Remove button - simple styling */
        [data-testid="stFileUploader"] button {
            background-color: transparent !important;
            border: none !important;
            color: var(--medium-gray) !important;
            font-size: 1rem !important;
            padding: 0.25rem 0.5rem !important;
            cursor: pointer !important;
            transition: all 0.15s ease !important;
            border-radius: 4px !important;
            opacity: 0.7 !important;
        }
        
        [data-testid="stFileUploader"] button:hover {
            background-color: rgba(218, 30, 40, 0.08) !important;
            color: #da1e28 !important;
            opacity: 1 !important;
        }
        
        /* Success messages */
        .stSuccess {
            background-color: var(--verification-green);
            color: #ffffff;
        }
        
        /* Input field */
        .stTextInput > div > div > input {
            border-color: var(--medium-gray);
        }
        
        .stTextInput > div > div > input:focus {
            border-color: var(--ibm-blue);
            box-shadow: 0 0 0 2px rgba(15, 98, 254, 0.1);
        }
        
        /* Chat input */
        .stChatInputContainer {
            background-color: #ffffff;
            border-top: 1px solid var(--medium-gray);
        }
        
        /* Citations/References - in main markdown */
        .stMarkdown strong {
            color: var(--dark-gray);
        }
        
        /* References heading in assistant messages - keep white, not green */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) 
        [data-testid="stChatMessageContent"] strong {
            color: #ffffff !important;
        }
        
        /* Links in citations */
        .stMarkdown a {
            color: var(--ibm-blue);
        }
        
        .stMarkdown a:hover {
            color: #0050e6;
        }
        
        /* Reference list items - simple, clean styling */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) 
        [data-testid="stChatMessageContent"] ul {
            border-left: 2px solid rgba(255, 255, 255, 0.3);
            padding-left: 16px;
        }
        
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) 
        [data-testid="stChatMessageContent"] ol {
            padding-left: 16px;
        }
        
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) 
        [data-testid="stChatMessageContent"] li {
            color: #ffffff !important;
            margin-bottom: 0.25rem;
        }
        
        /* Spinner */
        .stSpinner > div {
            border-color: var(--ibm-blue);
        }
        
        /* Dark mode specific overrides */
        @media (prefers-color-scheme: dark) {
            .stApp {
                background-color: var(--light-gray);
            }
            
            .main .block-container {
                background-color: var(--light-gray);
            }
            
            h1 {
                color: var(--dark-gray) !important;
            }
            
            .stMarkdown p {
                color: var(--dark-gray);
            }
            
            [data-testid="stSidebar"] {
                background-color: #2d3a47;
            }
            
            h2 {
                color: var(--dark-gray) !important;
            }
            
            .stInfo {
                background-color: rgba(61, 139, 255, 0.15) !important;
                border-left-color: var(--ibm-blue) !important;
            }
            
            [data-testid="stFileUploader"] {
                background-color: #2a2a2a;
                border-color: var(--medium-gray);
            }
            
            .drag-drop-overlay {
                background: rgba(61, 139, 255, 0.15);
                border-color: var(--ibm-blue);
            }
            
            .drag-drop-overlay-content {
                background: #2a2a2a;
                border-color: var(--ibm-blue);
            }
            
            .drag-drop-overlay-content h2 {
                color: var(--ibm-blue);
            }
            
            .drag-drop-overlay-content p {
                color: var(--medium-gray);
            }
            
            [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] {
                background-color: #1a1a1a !important;
                border-color: var(--ibm-blue) !important;
            }
            
            [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"]:hover {
                background-color: #2a2a2a !important;
                border-color: var(--ibm-blue) !important;
            }
            
            [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] span,
            [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] div {
                color: var(--dark-gray) !important;
            }
            
            .stChatInputContainer {
                background-color: #2a2a2a;
                border-top: 1px solid var(--medium-gray);
            }
            
            /* Dark mode chat messages */
            [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="user"]) [data-testid="stChatMessageContent"] {
                background-color: var(--ibm-blue);
            }
            
            [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stChatMessageContent"] {
                background-color: #4a5a6a;
            }
        }

        /* Hide the Streamlit header and footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        </style>
        
        <!-- Drag and Drop Overlay -->
        <div class="drag-drop-overlay" id="dragDropOverlay">
            <div class="drag-drop-overlay-content">
                <div class="icon">📄</div>
                <h2>Drop files to upload</h2>
                <p>Release to add files to VeriCite</p>
            </div>
        </div>
        
        <script>
        (function() {
            // Wait for Streamlit to be ready
            function waitForStreamlit(callback) {
                if (window.parent.streamlitLoaded) {
                    callback();
                } else {
                    window.addEventListener('streamlit:loaded', callback);
                }
            }
            
            // Find the file uploader input element with retry
            function findFileUploader() {
                // Try multiple selectors for Streamlit's file uploader
                const selectors = [
                    '[data-testid="stFileUploader"] input[type="file"]',
                    '.stFileUploader input[type="file"]',
                    'input[type="file"][accept*="pdf"]',
                    'input[type="file"]'
                ];
                
                for (const selector of selectors) {
                    const uploader = document.querySelector(selector);
                    if (uploader) {
                        return uploader;
                    }
                }
                return null;
            }
            
            const overlay = document.getElementById('dragDropOverlay');
            let dragCounter = 0;
            let isDragging = false;
            
            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }
            
            // Handle drag enter
            function handleDragEnter(e) {
                // Only handle file drags
                if (e.dataTransfer.items) {
                    for (let i = 0; i < e.dataTransfer.items.length; i++) {
                        if (e.dataTransfer.items[i].kind === 'file') {
                            isDragging = true;
                            dragCounter++;
                            overlay.classList.add('active');
                            break;
                        }
                    }
                }
            }
            
            // Handle drag over - critical for drop to work
            function handleDragOver(e) {
                if (isDragging) {
                    preventDefaults(e);
                    e.dataTransfer.dropEffect = 'copy';
                }
            }
            
            // Handle drag leave
            function handleDragLeave(e) {
                // Only decrease counter if leaving the window
                if (!e.relatedTarget || !document.contains(e.relatedTarget)) {
                    dragCounter--;
                    if (dragCounter <= 0) {
                        dragCounter = 0;
                        isDragging = false;
                        overlay.classList.remove('active');
                    }
                }
            }
            
            // Handle drop
            function handleDrop(e) {
                preventDefaults(e);
                dragCounter = 0;
                isDragging = false;
                overlay.classList.remove('active');
                
                const dt = e.dataTransfer;
                const files = dt.files;
                
                if (files && files.length > 0) {
                    // Find uploader with retry mechanism
                    let uploader = findFileUploader();
                    
                    // If not found, wait a bit and retry (Streamlit might be re-rendering)
                    if (!uploader) {
                        setTimeout(() => {
                            uploader = findFileUploader();
                            if (uploader) {
                                setFiles(uploader, files);
                            }
                        }, 100);
                    } else {
                        setFiles(uploader, files);
                    }
                }
            }
            
            function setFiles(uploader, files) {
                try {
                    // Create a new FileList-like object using DataTransfer API
                    const dataTransfer = new DataTransfer();
                    
                    // Add existing files first (if any)
                    if (uploader.files) {
                        for (let i = 0; i < uploader.files.length; i++) {
                            dataTransfer.items.add(uploader.files[i]);
                        }
                    }
                    
                    // Add new files
                    for (let i = 0; i < files.length; i++) {
                        // Only add PDF files
                        if (files[i].type === 'application/pdf' || files[i].name.toLowerCase().endsWith('.pdf')) {
                            dataTransfer.items.add(files[i]);
                        }
                    }
                    
                    uploader.files = dataTransfer.files;
                    
                    // Trigger multiple events to ensure Streamlit picks it up
                    ['change', 'input'].forEach(eventType => {
                        const event = new Event(eventType, { bubbles: true, cancelable: true });
                        uploader.dispatchEvent(event);
                    });
                    
                    // Also try native input event
                    const inputEvent = new InputEvent('input', { bubbles: true });
                    uploader.dispatchEvent(inputEvent);
                    
                    // Visual feedback - success flash
                    overlay.style.display = 'flex';
                    overlay.style.background = 'rgba(36, 161, 72, 0.25)';
                    overlay.querySelector('h2').textContent = '✓ Files added!';
                    overlay.querySelector('p').textContent = 'Check the upload area in the sidebar';
                    setTimeout(() => {
                        overlay.classList.remove('active');
                        overlay.style.background = '';
                        overlay.querySelector('h2').textContent = 'Drop files to upload';
                        overlay.querySelector('p').textContent = 'Release to add files to VeriCite';
                    }, 2000);
                } catch (error) {
                    console.error('Error setting files:', error);
                    // Fallback: show error feedback
                    overlay.style.display = 'flex';
                    overlay.style.background = 'rgba(218, 30, 40, 0.2)';
                    overlay.querySelector('h2').textContent = 'Upload failed';
                    overlay.querySelector('p').textContent = 'Please use the file uploader in the sidebar';
                    setTimeout(() => {
                        overlay.classList.remove('active');
                        overlay.style.background = '';
                        overlay.querySelector('h2').textContent = 'Drop files to upload';
                        overlay.querySelector('p').textContent = 'Release to add files to VeriCite';
                    }, 2000);
                }
            }
            
            // Initialize when DOM is ready
            function initDragDrop() {
                // Prevent default drag behaviors on document
                ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                    document.addEventListener(eventName, preventDefaults, false);
                    document.body.addEventListener(eventName, preventDefaults, false);
                });
                
                // Add specific handlers
                document.addEventListener('dragenter', handleDragEnter);
                document.addEventListener('dragover', handleDragOver);
                document.addEventListener('dragleave', handleDragLeave);
                document.addEventListener('drop', handleDrop);
                
                // Also on window level for when dragging from outside
                window.addEventListener('dragenter', handleDragEnter);
                window.addEventListener('dragover', handleDragOver);
                window.addEventListener('dragleave', handleDragLeave);
                window.addEventListener('drop', handleDrop);
            }
            
            // Wait for Streamlit to be ready, then initialize
            waitForStreamlit(initDragDrop);
            
            // Also try initializing immediately (in case Streamlit is already loaded)
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', initDragDrop);
            } else {
                initDragDrop();
            }
        })();
        
        // Fix file size alignment - move to right side
        function fixFileSizeAlignment() {
            const fileUploaders = document.querySelectorAll('[data-testid="stFileUploader"]');
            fileUploaders.forEach(uploader => {
                const fileNames = uploader.querySelectorAll('[data-testid="stFileUploaderFileName"]');
                fileNames.forEach(fileName => {
                    // Ensure flexbox layout - simple
                    if (fileName.style.display !== 'flex') {
                        fileName.style.display = 'flex';
                        fileName.style.alignItems = 'center';
                        fileName.style.justifyContent = 'space-between';
                        fileName.style.gap = '0.5rem';
                    }
                    
                    // Find file size element - could be sibling or child
                    let fileSize = fileName.querySelector('[data-testid="stFileUploaderFileSize"]');
                    if (!fileSize) {
                        fileSize = fileName.nextElementSibling?.querySelector('[data-testid="stFileUploaderFileSize"]');
                    }
                    if (!fileSize) {
                        // Try to find by text content (MB, KB, etc.)
                        const allElements = Array.from(fileName.parentElement.querySelectorAll('*'));
                        const sizePattern = /\d+\.?\d*\s*(MB|KB|GB|bytes?)/i;
                        fileSize = allElements.find(el => 
                            el.textContent && 
                            (el.textContent.match(sizePattern) || 
                             (el.classList.contains('fileSize') || el.textContent.includes('MB')))
                        );
                    }
                    
                    // If file size is found as a child, move it or style it
                    if (fileSize && fileSize.parentElement === fileName) {
                        fileSize.style.marginLeft = 'auto';
                        fileSize.style.marginRight = '0.5rem';
                        fileSize.style.flexShrink = '0';
                        fileSize.style.whiteSpace = 'nowrap';
                    }
                    
                    // Make filename take available space
                    const filenameText = fileName.firstElementChild || fileName.childNodes[0];
                    if (filenameText && filenameText !== fileSize) {
                        if (filenameText.nodeType === 1) { // Element node
                            filenameText.style.flex = '1';
                            filenameText.style.minWidth = '0';
                            filenameText.style.overflow = 'hidden';
                            filenameText.style.textOverflow = 'ellipsis';
                            filenameText.style.whiteSpace = 'nowrap';
                        }
                    }
                });
            });
        }
        
        // Run on load and after mutations
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', fixFileSizeAlignment);
        } else {
            fixFileSizeAlignment();
        }
        
        // Watch for Streamlit updates
        const observer = new MutationObserver(fixFileSizeAlignment);
        observer.observe(document.body, { childList: true, subtree: true });
        
        // Also run periodically to catch Streamlit renders
        setInterval(fixFileSizeAlignment, 500);
        </script>
        
        """,
        unsafe_allow_html=True,
    )


def init_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "ingested_docs" not in st.session_state:
        st.session_state["ingested_docs"] = []
    if "show_upload_ui" not in st.session_state:
        # Hide upload UI by default if documents exist, show if no documents
        has_docs = len(st.session_state.get("ingested_docs", [])) > 0
        st.session_state["show_upload_ui"] = not has_docs


def upload_section(ingestion: IngestionPipeline):
    """File upload section in main area."""
    has_documents = len(st.session_state.get("ingested_docs", [])) > 0
    
    # If documents exist and upload UI is hidden, show toggle button
    if has_documents and not st.session_state.get("show_upload_ui", True):
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Upload More Documents", type="secondary", use_container_width=False):
                st.session_state["show_upload_ui"] = True
                st.rerun()
        return None
    
    # Show upload UI
    st.header("Upload Documents", divider="blue")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Drag PDF files here or click to browse", 
            type=["pdf"], 
            accept_multiple_files=True,
            help="Upload one or more PDF documents for research. Files will be processed and indexed.",
            label_visibility="visible"
        )
    
    with col2:
        if uploaded_files:
            st.markdown("")  # Spacing
            ingest_button = st.button(
                f"Ingest {len(uploaded_files)} file(s)", 
                type="primary",
                use_container_width=True
            )
        else:
            ingest_button = False
    
    # Hide button if documents exist
    if has_documents:
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("Hide Upload", type="secondary", use_container_width=True):
                st.session_state["show_upload_ui"] = False
                st.rerun()
    
    if uploaded_files and ingest_button:
        with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
            for f in uploaded_files:
                doc_id = str(uuid.uuid4())
                with st.spinner(f"Uploading and ingesting {f.name}..."):
                    source_uri = ingestion.upload_to_cos(doc_id, f.name, f)
                    count = ingestion.ingest_pdf(doc_id, f.name, source_uri)
                st.session_state["ingested_docs"].append((doc_id, f.name, source_uri, count))
        st.success(f"Successfully ingested {len(uploaded_files)} document(s)")
        # Hide upload UI after successful ingestion
        st.session_state["show_upload_ui"] = False
        st.rerun()  # Refresh to show new documents
    
    return uploaded_files


def sidebar_documents():
    """Compact sidebar showing only ingested documents."""
    st.sidebar.markdown("### Ingested Documents")
    
    if st.session_state["ingested_docs"]:
        st.sidebar.caption(f"{len(st.session_state['ingested_docs'])} document(s)")
        for doc_id, name, uri, count in st.session_state["ingested_docs"]:
            # Truncate long names
            display_name = name if len(name) <= 35 else name[:32] + "..."
            st.sidebar.markdown(f"**{display_name}**")
            st.sidebar.caption(f"{count} chunks")
    else:
        st.sidebar.info("Upload documents to get started")


def chat_ui(query_pipeline: QueryPipeline):
    """Main chat interface."""
    st.header("Research Assistant", divider="blue")
    
    if not st.session_state["ingested_docs"]:
        st.info("Upload documents above to start asking questions about your research materials.")
        # Show placeholder chat interface even without documents
        return
    
    # Display chat history
    for role, content in st.session_state["messages"]:
        with st.chat_message(role):
            st.markdown(content)

    # Chat input
    user_input = st.chat_input("Ask a research question about your documents…")
    if user_input:
        st.session_state["messages"].append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                answer, sources = query_pipeline.answer(user_input)
                if sources:
                    citations = "\n\n**References:**\n" + "\n".join([f"- {s}" for s in sources])
                    full_response = answer + citations
                else:
                    full_response = answer
                st.markdown(full_response)
                st.session_state["messages"].append(("assistant", full_response))


def main():
    # Set page config for VeriCite branding
    st.set_page_config(
        page_title="VeriCite - Research Assistant",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load local .env for development
    load_dotenv()
    settings = Settings.from_env()
    init_state()
    
    # Inject custom CSS for medical research UI
    inject_custom_css()

    ingestion = IngestionPipeline(settings)
    query_pipeline = QueryPipeline(settings)

    # Main layout: Title, Upload, Chat
    st.title("VeriCite")
    st.caption("Research Assistant • Powered by watsonx.ai • Hybrid Search • Granite-3-8B-Instruct")
    
    # Upload section in main area (can be hidden)
    upload_result = upload_section(ingestion)
    
    # Only show divider if upload UI was shown
    if st.session_state.get("show_upload_ui", True) and upload_result is not None:
        st.markdown("---")
    
    # Chat interface
    chat_ui(query_pipeline)
    
    # Sidebar: Just documents list
    sidebar_documents()


if __name__ == "__main__":
    main()


