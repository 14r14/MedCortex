import uuid
from dotenv import load_dotenv
import streamlit as st

from app.config import Settings
from app.rag.pipeline import IngestionPipeline, QueryPipeline


@st.cache_resource
def get_css_content():
    """Return CSS content - cached to avoid re-injecting on every rerun."""
    return r"""
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
            color: #f0f0f0 !important;
            font-weight: 700;
            font-size: 1.75rem;
            letter-spacing: -0.01em;
            margin-bottom: 0.5rem !important;
        }
        
        [data-testid="stSidebar"] .stMarkdown h2,
        [data-testid="stSidebar"] .stMarkdown h3 {
            color: #f0f0f0 !important;
        }
        
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] p {
            color: #e0e0e0 !important;
        }
        
        /* Sidebar dividers */
        [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.2) !important;
            margin: 1rem 0 !important;
        }
        
        /* Sidebar document items */
        [data-testid="stSidebar"] .stMarkdown strong {
            color: #f0f0f0 !important;
            font-weight: 600;
        }
        
        [data-testid="stSidebar"] .stMarkdown .stCaption {
            color: rgba(240, 240, 240, 0.8) !important;
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
            color: #f0f0f0 !important;
            border-radius: 12px 12px 12px 0;
            padding: 12px 16px;
        }
        
        /* Ensure all text in assistant messages is soft white */
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
            color: #f0f0f0 !important;
        }
        
        /* Bold/strong text in assistant messages */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stChatMessageContent"] strong,
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stChatMessageContent"] b {
            color: #f0f0f0 !important;
        }
        
        /* Links in assistant messages - soft white */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stChatMessageContent"] a {
            color: #f0f0f0 !important;
            text-decoration: underline !important;
        }
        
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stChatMessageContent"] a:hover {
            color: #f0f0f0 !important;
            opacity: 0.9 !important;
        }
        
        /* ============ CURSOR-STYLE BUTTONS ============ */
        /* Base Cursor-style button - applies to all buttons by default */
        .stButton > button,
        [data-testid="stButton"] > button,
        .stDownloadButton > button,
        button[data-testid="baseButton-secondary"],
        button[data-testid="baseButton-primary"] {
            /* Cursor-style base appearance */
            background-color: rgba(255, 255, 255, 0.9) !important;
            color: var(--dark-gray) !important;
            border: 1px solid rgba(0, 0, 0, 0.12) !important;
            border-radius: 6px !important;
            padding: 0.375rem 0.75rem !important;
            font-size: 0.8125rem !important;
            font-weight: 400 !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
            cursor: pointer !important;
            transition: all 0.15s ease !important;
            box-shadow: none !important;
            outline: none !important;
            min-height: auto !important;
            height: auto !important;
            white-space: nowrap !important;
        }
        
        /* Hover state */
        .stButton > button:hover,
        [data-testid="stButton"] > button:hover,
        .stDownloadButton > button:hover,
        button[data-testid="baseButton-secondary"]:hover,
        button[data-testid="baseButton-primary"]:hover {
            background-color: rgba(255, 255, 255, 1) !important;
            border-color: rgba(0, 0, 0, 0.2) !important;
        }
        
        /* Active/pressed state - no special styling */
        .stButton > button:active,
        [data-testid="stButton"] > button:active,
        .stDownloadButton > button:active,
        button[data-testid="baseButton-secondary"]:active,
        button[data-testid="baseButton-primary"]:active {
            /* No active styles - keep same as default */
        }
        
        /* Focus state - no special styling */
        .stButton > button:focus,
        [data-testid="stButton"] > button:focus,
        .stDownloadButton > button:focus,
        button[data-testid="baseButton-secondary"]:focus,
        button[data-testid="baseButton-primary"]:focus,
        .stButton > button:focus-visible,
        [data-testid="stButton"] > button:focus-visible,
        .stDownloadButton > button:focus-visible,
        button[data-testid="baseButton-secondary"]:focus-visible,
        button[data-testid="baseButton-primary"]:focus-visible {
            /* No focus styles - keep same as default */
            outline: none !important;
        }
        
        /* Disabled state */
        .stButton > button:disabled,
        [data-testid="stButton"] > button:disabled,
        .stDownloadButton > button:disabled,
        button[data-testid="baseButton-secondary"]:disabled,
        button[data-testid="baseButton-primary"]:disabled {
            opacity: 0.5 !important;
            cursor: not-allowed !important;
        }
        
        /* Dark mode support */
        @media (prefers-color-scheme: dark) {
            .stButton > button,
            [data-testid="stButton"] > button,
            .stDownloadButton > button,
            button[data-testid="baseButton-secondary"],
            button[data-testid="baseButton-primary"] {
                background-color: rgba(255, 255, 255, 0.1) !important;
                color: #f0f0f0 !important;
                border-color: rgba(255, 255, 255, 0.2) !important;
            }
            
            .stButton > button:hover,
            [data-testid="stButton"] > button:hover,
            .stDownloadButton > button:hover,
            button[data-testid="baseButton-secondary"]:hover,
            button[data-testid="baseButton-primary"]:hover {
                background-color: rgba(255, 255, 255, 0.15) !important;
                border-color: rgba(255, 255, 255, 0.3) !important;
            }
            
            .stButton > button:active,
            [data-testid="stButton"] > button:active,
            .stDownloadButton > button:active,
            button[data-testid="baseButton-secondary"]:active,
            button[data-testid="baseButton-primary"]:active {
                /* No active styles - keep same as default */
            }
        }
        
        /* Context-specific button styles - different styles for different locations */
        
        /* Main content buttons */
        .main .stButton > button {
            padding: 0.5rem 1rem !important;
            font-size: 0.875rem !important;
        }
        
        /* Chat area buttons (Add to Report) - smaller */
        [data-testid="stChatMessage"] ~ .stButton > button,
        .main [data-testid="stChatMessage"] ~ .stButton > button {
            padding: 0.375rem 0.75rem !important;
            font-size: 0.75rem !important;
        }
        
        /* Download buttons - same as base */
        .stDownloadButton > button {
            /* Uses base Cursor style */
        }
        
        /* ============ NAVIGATION SECTION (Sidebar) ============ */
        /* Reduce spacing after Navigation heading */
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] .stMarkdown h3 {
            margin-bottom: 0.1rem !important;
            margin-top: 0 !important;
            padding-bottom: 0 !important;
            padding-top: 0 !important;
            line-height: 1.2 !important;
        }
        
        /* Reduce spacing between Navigation heading and buttons */
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(h3) + [data-testid="stVerticalBlock"],
        [data-testid="stSidebar"] .stMarkdown:has(h3) ~ [data-testid="stVerticalBlock"] {
            margin-top: 0.1rem !important;
            padding-top: 0 !important;
        }
        
        /* Reduce spacing on button containers that follow Navigation heading */
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(h3) ~ [data-testid="stVerticalBlock"] .stButton,
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(h3) + [data-testid="stVerticalBlock"] .stButton {
            margin-top: 0 !important;
            padding-top: 0 !important;
            margin-bottom: 0.25rem !important;
        }
        
        /* Target the first button after Navigation more directly */
        [data-testid="stSidebar"] h3 ~ [data-testid="stVerticalBlock"] .stButton,
        [data-testid="stSidebar"] h3 + [data-testid="stVerticalBlock"] .stButton,
        [data-testid="stSidebar"] .stMarkdown:has(h3) + [data-testid="stVerticalBlock"] .stButton {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        
        /* Navigation buttons in sidebar - high contrast for light mode */
        [data-testid="stSidebar"] .stButton > button,
        [data-testid="stSidebar"] button[data-testid="baseButton-secondary"] {
            background-color: rgba(255, 255, 255, 0.15) !important;
            color: #f0f0f0 !important;
            border: 1px solid rgba(255, 255, 255, 0.25) !important;
            font-weight: 500 !important;
        }
        
        [data-testid="stSidebar"] .stButton > button:hover,
        [data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:hover {
            background-color: rgba(255, 255, 255, 0.25) !important;
            border-color: rgba(255, 255, 255, 0.4) !important;
            color: #ffffff !important;
        }
        
        [data-testid="stSidebar"] .stButton > button:disabled,
        [data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:disabled {
            background-color: rgba(255, 255, 255, 0.08) !important;
            color: rgba(240, 240, 240, 0.5) !important;
            border-color: rgba(255, 255, 255, 0.15) !important;
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
        
        /* Status update text with pulsing animation */
        .status-update-text {
            color: var(--dark-gray) !important;
            font-size: 0.95rem !important;
            padding: 0 !important;
            margin: 0 !important;
            animation: pulse 2s ease-in-out infinite;
            line-height: 1.5 !important;
        }
        
        /* Remove paragraph default margins for status text */
        [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] p.status-update-text {
            margin: 0 !important;
            padding: 0 !important;
        }
        
        /* Align status container with chat message avatar - assistant messages soft white */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stChatMessageContent"] .status-update-text {
            color: #f0f0f0 !important;
            display: block !important;
            line-height: 1.5 !important;
        }
        
        /* Ensure status container has no extra padding */
        [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] [data-testid="stVerticalBlock"]:has(.status-update-text),
        [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] div:has(.status-update-text) {
            padding: 0 !important;
            margin: 0 !important;
        }
        
        @keyframes pulse {
            0%, 100% {
                opacity: 1;
            }
            50% {
                opacity: 0.6;
            }
        }
        
        /* Minimalistic expander styling - match status text style */
        [data-testid="stExpander"] {
            border: none !important;
            box-shadow: none !important;
            background: transparent !important;
            margin: 0.5rem 0 !important;
            outline: none !important;
        }
        
        [data-testid="stExpander"] > div {
            border: none !important;
            background: transparent !important;
            outline: none !important;
        }
        
        /* Expander header (button) - minimalistic, match status text */
        [data-testid="stExpander"] summary {
            border: none !important;
            outline: none !important;
            background: transparent !important;
            padding: 0 !important;
            margin: 0 !important;
            font-weight: 400 !important;
            font-size: 0.95rem !important;
            color: var(--dark-gray) !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            border-radius: 8px !important;
            display: flex !important;
            align-items: center !important;
            list-style: none !important;
        }
        
        /* Expander in assistant chat messages - soft white text */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stExpander"] summary {
            color: #f0f0f0 !important;
        }
        
        /* Curved hover effect - subtle background with rounded corners */
        [data-testid="stExpander"] summary:hover {
            background: rgba(0, 0, 0, 0.05) !important;
            border-radius: 8px !important;
        }
        
        /* Expander hover in assistant messages */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stExpander"] summary:hover {
            background: rgba(240, 240, 240, 0.1) !important;
        }
        
        /* Expander content area - distinguished with subtle background and indentation */
        [data-testid="stExpander"] [data-testid="stExpanderContent"] {
            border: none !important;
            outline: none !important;
            background: rgba(0, 0, 0, 0.02) !important;
            padding: 0.75rem 1rem !important;
            margin-top: 0.25rem !important;
            margin-left: 0.5rem !important;
            border-radius: 8px !important;
            border-left: 2px solid rgba(0, 0, 0, 0.1) !important;
        }
        
        /* For assistant chat messages, use lighter background and soft white border */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stExpander"] [data-testid="stExpanderContent"] {
            background: rgba(240, 240, 240, 0.05) !important;
            border-left: 2px solid rgba(240, 240, 240, 0.2) !important;
        }
        
        [data-testid="stExpander"][open] [data-testid="stExpanderContent"] {
            background: rgba(0, 0, 0, 0.03) !important;
            border-left: 2px solid rgba(0, 0, 0, 0.15) !important;
        }
        
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stExpander"][open] [data-testid="stExpanderContent"] {
            background: rgba(240, 240, 240, 0.08) !important;
            border-left: 2px solid rgba(240, 240, 240, 0.3) !important;
        }
        
        /* Ensure no borders on expander container */
        [data-testid="stExpander"] details {
            border: none !important;
            outline: none !important;
            background: transparent !important;
        }
        
        /* File uploader - Compact styling */
        [data-testid="stFileUploader"] {
            border: 2px dashed var(--ibm-blue) !important;
            border-radius: 8px !important;
            background: rgba(15, 98, 254, 0.03) !important;
            padding: 1rem !important;
            transition: all 0.2s ease !important;
            min-height: 60px !important;
            width: 100% !important;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: var(--ibm-blue) !important;
            background: rgba(15, 98, 254, 0.05) !important;
            box-shadow: 0 2px 8px rgba(15, 98, 254, 0.1) !important;
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
        
        /* File uploader browse button - IBM blue instead of red */
        [data-testid="stFileUploader"] button {
            background-color: transparent !important;
            border: none !important;
            color: var(--ibm-blue) !important;
            font-size: 1rem !important;
            padding: 0.25rem 0.5rem !important;
            cursor: pointer !important;
            transition: all 0.15s ease !important;
            border-radius: 4px !important;
        }
        
        [data-testid="stFileUploader"] button:hover {
            background-color: rgba(15, 98, 254, 0.1) !important;
            color: var(--ibm-blue) !important;
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
        
        /* Textarea styling - Cursor-style minimal design */
        .stTextArea > div > div > textarea {
            background-color: #ffffff !important;
            border: 1px solid rgba(0, 0, 0, 0.12) !important;
            border-radius: 6px !important;
            padding: 0.75rem 0.875rem !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
            font-size: 0.875rem !important;
            line-height: 1.6 !important;
            color: var(--dark-gray) !important;
            box-shadow: none !important;
            transition: all 0.15s ease !important;
            resize: vertical !important;
            outline: none !important;
        }
        
        .stTextArea > div > div > textarea:focus {
            border-color: var(--ibm-blue) !important;
            box-shadow: none !important;
            outline: none !important;
            outline-width: 0 !important;
            outline-style: none !important;
            outline-color: transparent !important;
            background-color: #ffffff !important;
        }
        
        .stTextArea > div > div > textarea:focus-visible {
            outline: none !important;
            outline-width: 0 !important;
            outline-style: none !important;
            outline-color: transparent !important;
        }
        
        .stTextArea > div > div > textarea:hover {
            border-color: rgba(0, 0, 0, 0.2) !important;
        }
        
        /* Textarea container */
        .stTextArea > div {
            background-color: transparent !important;
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
        
        /* References heading in assistant messages - keep soft white, not green */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) 
        [data-testid="stChatMessageContent"] strong {
            color: #f0f0f0 !important;
        }
        
        /* Verification badges - pill-shaped indicators */
        .verification-badge {
            display: inline-block;
            padding: 3px 10px;
            margin-left: 6px;
            margin-right: 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            vertical-align: middle;
            line-height: 1.4;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            min-width: 110px;
            width: auto;
            text-align: center;
            box-sizing: border-box;
            white-space: nowrap;
        }
        .verification-badge.verified {
            background-color: #24a148;
            color: white;
        }
        .verification-badge.unverified {
            background-color: #ff832b;
            color: white;
            position: relative;
            cursor: help;
        }
        .verification-badge.refuted {
            background-color: #da1e28;
            color: white;
        }
        
        /* Tooltip for unverified/extrapolated badges */
        .verification-badge.unverified:hover::after {
            content: "This statement is part of the synthesized summary but could not be directly verified from the cited text. Please review the source for full context.";
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            margin-bottom: 8px;
            padding: 0.75rem 1rem;
            background-color: var(--dark-gray);
            color: #ffffff;
            font-size: 0.875rem;
            font-weight: 400;
            white-space: normal;
            width: 280px;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            z-index: 1000;
            pointer-events: none;
            text-transform: none;
            line-height: 1.5;
            text-align: left;
        }
        
        /* Tooltip arrow */
        .verification-badge.unverified:hover::before {
            content: "";
            position: absolute;
            bottom: 92%;
            left: 50%;
            transform: translateX(-50%);
            border: 6px solid transparent;
            border-top-color: var(--dark-gray);
            z-index: 1001;
            pointer-events: none;
        }
        
        /* Badge in claim list - no left margin */
        .claim-item-badge {
            margin-left: 0 !important;
        }
        
        /* Dark mode support for badges */
        @media (prefers-color-scheme: dark) {
            .verification-badge.verified {
                background-color: #42be65;
            }
            .verification-badge.unverified {
                background-color: #ff832b;
            }
            .verification-badge.refuted {
                background-color: #fa4d56;
            }
            
            /* Dark mode tooltip - use light background */
            .verification-badge.unverified:hover::after {
                background-color: #2a2a2a;
                color: #f0f0f0;
            }
            
            .verification-badge.unverified:hover::before {
                border-top-color: #2a2a2a;
            }
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
            color: #f0f0f0 !important;
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
            
            /* Status text - dark mode */
            .status-update-text {
                color: var(--dark-gray) !important;
            }
            
            /* Expander - dark mode */
            [data-testid="stExpander"] summary {
                color: var(--dark-gray) !important;
            }
            
            [data-testid="stExpander"] summary:hover {
                background: rgba(255, 255, 255, 0.05) !important;
            }
            
            [data-testid="stExpander"] [data-testid="stExpanderContent"] {
                background: rgba(255, 255, 255, 0.03) !important;
                border-left: 2px solid rgba(255, 255, 255, 0.15) !important;
            }
            
            [data-testid="stExpander"][open] [data-testid="stExpanderContent"] {
                background: rgba(255, 255, 255, 0.05) !important;
                border-left: 2px solid rgba(255, 255, 255, 0.2) !important;
            }
            
            /* Expander in assistant messages still uses soft white text in dark mode */
            [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stExpander"] summary {
                color: #f0f0f0 !important;
            }
            
            [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stExpander"] summary:hover {
                background: rgba(240, 240, 240, 0.1) !important;
            }
            
            [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stExpander"] [data-testid="stExpanderContent"] {
                background: rgba(240, 240, 240, 0.05) !important;
                border-left: 2px solid rgba(240, 240, 240, 0.2) !important;
            }
            
            [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"] [data-icon="assistant"]) [data-testid="stExpander"][open] [data-testid="stExpanderContent"] {
                background: rgba(240, 240, 240, 0.08) !important;
                border-left: 2px solid rgba(240, 240, 240, 0.3) !important;
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
            
            /* Textarea dark mode styling - Cursor-style */
            .stTextArea > div > div > textarea {
                background-color: #1a1a1a !important;
                border: 1px solid rgba(255, 255, 255, 0.2) !important;
                color: #f0f0f0 !important;
                box-shadow: none !important;
                outline: none !important;
            }
            
            .stTextArea > div > div > textarea:focus {
                border-color: var(--ibm-blue) !important;
                box-shadow: none !important;
                outline: none !important;
                outline-width: 0 !important;
                outline-style: none !important;
                outline-color: transparent !important;
                background-color: #1a1a1a !important;
            }
            
            .stTextArea > div > div > textarea:focus-visible {
                outline: none !important;
                outline-width: 0 !important;
                outline-style: none !important;
                outline-color: transparent !important;
            }
            
            .stTextArea > div > div > textarea:hover {
                border-color: rgba(255, 255, 255, 0.3) !important;
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
        
        /* Ensure sidebar toggle button remains visible and functional */
        [data-testid="stSidebarCollapseButton"],
        button[data-testid="stSidebarCollapseButton"],
        [aria-label*="sidebar"][aria-label*="toggle"],
        button[aria-label*="Open sidebar"],
        button[aria-label*="Close sidebar"] {
            visibility: hidden !important;
            display: none !important;
        }
        
        /* Ensure sidebar toggle button icon is visible */
        [data-testid="stSidebarCollapseButton"] svg,
        button[data-testid="stSidebarCollapseButton"] svg,
        [data-testid="stSidebarCollapseButton"]::before,
        [data-testid="stSidebarCollapseButton"]::after {
            display: block !important;
            visibility: visible !important;
        }
        
        /* Custom sidebar toggle icon styling */
        [data-testid="stSidebarCollapseButton"] .custom-sidebar-icon {
            width: 20px !important;
            height: 20px !important;
            fill: currentColor !important;
            stroke: currentColor !important;
            color: var(--dark-gray) !important;
        }
        
        /* Hide text/aria-label and show icon */
        [data-testid="stSidebarCollapseButton"] .sr-only,
        [data-testid="stSidebarCollapseButton"] [class*="sr-only"],
        button[aria-label*="sidebar"] .sr-only {
            position: absolute !important;
            width: 1px !important;
            height: 1px !important;
            padding: 0 !important;
            margin: -1px !important;
            overflow: hidden !important;
            clip: rect(0, 0, 0, 0) !important;
            white-space: nowrap !important;
            border: 0 !important;
        }
        
        </style>
        """


def inject_custom_css():
    """Inject custom CSS for medical research UI with IBM brand colors - optimized with caching."""
    css_content = get_css_content()
    # Use a key to ensure CSS is only injected once per session
    if "_css_injected" not in st.session_state:
        st.markdown(css_content, unsafe_allow_html=True)
        st.session_state["_css_injected"] = True
    else:
        # Still inject CSS but Streamlit will optimize it
        st.markdown(css_content, unsafe_allow_html=True)


def init_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "ingested_docs" not in st.session_state:
        st.session_state["ingested_docs"] = []
    if "show_upload_ui" not in st.session_state:
        # Hide upload UI by default if documents exist, show if no documents
        has_docs = len(st.session_state.get("ingested_docs", [])) > 0
        st.session_state["show_upload_ui"] = not has_docs
    if "report_text" not in st.session_state:
        st.session_state["report_text"] = ""


def upload_section(ingestion: IngestionPipeline):
    """File upload section in main area."""
    has_documents = len(st.session_state.get("ingested_docs", [])) > 0
    
    # Check if answer generation is in progress (status callback indicates generation)
    is_generating = "_status_callback" in st.session_state
    
    # If documents exist and upload UI is hidden, show toggle button
    if has_documents and not st.session_state.get("show_upload_ui", True):
        if st.button("Upload More Documents", type="secondary", use_container_width=False, disabled=is_generating):
            # Don't allow state change during generation
            if not is_generating:
                st.session_state["show_upload_ui"] = True
                # Use rerun for navigation state changes
                st.rerun()
        if is_generating:
            st.caption("⚠️ Answer generation in progress - please wait...")
        return None
    
    # Show upload UI
    st.header("Upload Documents")
    
    # Hide button if documents exist - positioned on the left
    if has_documents:
        if st.button("Hide Upload", type="secondary", use_container_width=False, disabled=is_generating):
            # Don't allow state change during generation
            if not is_generating:
                st.session_state["show_upload_ui"] = False
                # Use rerun for navigation state changes
                st.rerun()
        if is_generating:
            st.caption("⚠️ Answer generation in progress - please wait...")
        st.markdown("")  # Spacing after button
    
    # Upload area - full width
    uploaded_files = st.file_uploader(
        "Drag PDF files here or click to browse", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Upload one or more PDF documents for research. Files will be processed and indexed.",
        label_visibility="visible"
    )
    
    # Check if answer generation is in progress
    is_generating = "_status_callback" in st.session_state
    
    # Ingest button below uploader
    if uploaded_files:
        ingest_button = st.button(
            f"Ingest {len(uploaded_files)} file(s)", 
            type="primary",
            use_container_width=False,
            disabled=is_generating
        )
        if is_generating:
            st.caption("⚠️ Answer generation in progress - please wait before ingesting files...")
    else:
        ingest_button = False
    
    # Don't process during generation to avoid interruption
    if uploaded_files and ingest_button and not is_generating:
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
        # Rerun needed to show new documents and update UI
        st.rerun()
    
    return uploaded_files


def add_to_report(answer: str, sources: list[str], query: str = "") -> None:
    """Add a verified answer to the report."""
    if "report_text" not in st.session_state:
        st.session_state["report_text"] = ""
    
    # Format the entry (only answer and references, no query)
    # Add spacing only if report is not empty
    report_empty = not st.session_state["report_text"].strip()
    entry = "" if report_empty else "\n\n"
    # Strip leading/trailing whitespace from answer
    answer_clean = answer.strip()
    entry += f"{answer_clean}\n\n"
    
    if sources:
        entry += "**References:**\n"
        for i, source in enumerate(sources, 1):
            entry += f"{i}. {source}\n"
    
    st.session_state["report_text"] += entry


def generate_bibliography(sources: list[str]) -> str:
    """Generate a formatted bibliography from source URIs."""
    import re
    
    if not sources:
        return ""
    
    bibliography = "\n\n## Bibliography\n\n"
    
    # Extract unique documents from sources
    unique_docs = {}
    for source in sources:
        # Find document metadata
        for doc_info in st.session_state["ingested_docs"]:
            if len(doc_info) >= 3 and doc_info[2] == source:
                title = doc_info[4] if len(doc_info) >= 6 and doc_info[4] else None
                author = doc_info[5] if len(doc_info) >= 6 and doc_info[5] else None
                filename = doc_info[1] if len(doc_info) >= 2 else None
                
                doc_key = source
                if doc_key not in unique_docs:
                    unique_docs[doc_key] = {
                        "title": title,
                        "author": author,
                        "filename": filename,
                        "source_uri": source
                    }
                break
    
    # Format in APA style (simplified)
    for i, (doc_key, doc_info) in enumerate(unique_docs.items(), 1):
        title = doc_info.get("title") or doc_info.get("filename", "").replace(".pdf", "").replace(".PDF", "")
        author = doc_info.get("author", "Unknown Author")
        source_uri = doc_info.get("source_uri", "")
        
        # Format as APA citation
        bibliography += f"{i}. {author}. {title}. {source_uri}\n"
    
    return bibliography


def export_report(format: str = "docx") -> bytes:
    """Export the research report in the specified format."""
    from datetime import datetime
    import re
    
    report_text = st.session_state.get("report_text", "")
    if not report_text:
        return b""
    
    # Extract sources from report text
    source_pattern = r's3://[^\s\n]+'
    all_sources = re.findall(source_pattern, report_text)
    unique_sources = list(dict.fromkeys(all_sources))
    
    if format == "docx":
        try:
            from docx import Document
            from docx.shared import Pt
            
            doc = Document()
            
            # Title
            doc.add_heading('Research Report', 0)
            doc.add_paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}")
            doc.add_paragraph("")  # Blank line
            
            # Parse report text and convert to DOCX
            lines = report_text.split('\n')
            current_para = []
            in_refs_section = False
            
            for line in lines:
                line_stripped = line.strip()
                if not line_stripped:
                    if current_para:
                        para = doc.add_paragraph(" ".join(current_para))
                        current_para = []
                elif line_stripped.startswith("**References:**"):
                    if current_para:
                        para = doc.add_paragraph(" ".join(current_para))
                        current_para = []
                    doc.add_heading("References", level=2)
                    in_refs_section = True
                elif in_refs_section and re.match(r'^\d+\.', line_stripped):
                    # Reference list item
                    ref_text = line_stripped
                    doc.add_paragraph(ref_text, style='List Number')
                else:
                    # Regular paragraph text
                    if line_stripped.startswith("**") and line_stripped.endswith("**"):
                        # Bold text
                        current_para.append(line_stripped.replace("**", ""))
                    else:
                        current_para.append(line_stripped)
            
            if current_para:
                doc.add_paragraph(" ".join(current_para))
            
            # Add bibliography
            bibliography = generate_bibliography(unique_sources)
            if bibliography:
                doc.add_page_break()
                for line in bibliography.split('\n'):
                    if line.strip():
                        if line.startswith("##"):
                            doc.add_heading(line.replace("##", "").strip(), level=2)
                        elif re.match(r'^\d+\.', line.strip()):
                            doc.add_paragraph(line.strip(), style='List Number')
                        else:
                            doc.add_paragraph(line.strip())
            
            # Save to bytes
            from io import BytesIO
            buffer = BytesIO()
            doc.save(buffer)
            return buffer.getvalue()
        except ImportError:
            return b""
    
    elif format == "md":
        md_content = report_text
        bibliography = generate_bibliography(unique_sources)
        if bibliography:
            md_content += bibliography
        
        return md_content.encode('utf-8')
    
    return b""


def report_page():
    """Display and manage the research report page."""
    from datetime import datetime
    
    st.header("Research Report")
    st.caption("Curate your verified findings into a formatted research report")
    
    # Export buttons at the top
    try:
        from docx import Document
        DOCX_AVAILABLE = True
    except ImportError:
        DOCX_AVAILABLE = False
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if DOCX_AVAILABLE:
            docx_data = export_report("docx")
            st.download_button(
                label="Export as DOCX",
                data=docx_data if docx_data else b"",
                file_name=f"research_report_{datetime.now().strftime('%Y%m%d')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                disabled=not st.session_state.get("report_text", ""),
                use_container_width=True
            )
        else:
            st.info("⚠️ python-docx not available. Install it to export as DOCX.")
    
    with col2:
        md_data = export_report("md")
        st.download_button(
            label="Export as Markdown",
            data=md_data if md_data else b"",
            file_name=f"research_report_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
            disabled=not st.session_state.get("report_text", ""),
            use_container_width=True
        )
    
    st.divider()
    
    # Report text area (editable)
    if not st.session_state.get("report_text", ""):
        st.info("Your report is empty. Add verified findings from the Research Assistant page to build your report.")
        st.session_state["report_text"] = ""
    
    # Editable text area
    report_content = st.text_area(
        "Report Content",
        value=st.session_state.get("report_text", ""),
        height=500,
        help="Edit your report here. Content added via 'Add to Report' buttons will appear here.",
        label_visibility="collapsed"
    )
    
    # Update session state if user edits
    if report_content != st.session_state.get("report_text", ""):
        st.session_state["report_text"] = report_content
    
    # Clear button
    if st.button("Clear Report", use_container_width=True):
        st.session_state["report_text"] = ""
        # Rerun needed to clear the text area
        st.rerun()


def sidebar_documents():
    """Compact sidebar showing only ingested documents - optimized."""
    st.sidebar.markdown("### Ingested Documents")
    
    ingested_docs = st.session_state.get("ingested_docs", [])
    if ingested_docs:
        st.sidebar.caption(f"{len(ingested_docs)} document(s)")
        # Updated to handle new format: (doc_id, filename, source_uri, count, title, author)
        for doc_info in ingested_docs:
            if len(doc_info) >= 4:
                doc_id, name, uri, count = doc_info[0], doc_info[1], doc_info[2], doc_info[3]
                # Truncate long names
                display_name = name if len(name) <= 35 else name[:32] + "..."
                st.sidebar.markdown(f"**{display_name}**")
                st.sidebar.caption(f"{count} chunks")
    else:
        st.sidebar.info("Upload documents to get started")


def display_agent_trajectory(query: str, trajectory: list[dict]) -> None:
    """Display the agent's step-by-step reasoning process."""
    with st.expander("Analysis Breakdown", expanded=False):
        st.caption(f"Showing step-by-step reasoning for: \"{query[:30]}...{query[-30:] if len(query) > 60 else query}\"")
        
        with st.container():
            for step_info in trajectory:
                step_type = step_info.get("type", "")
                title = step_info.get("title", "")
                content = step_info.get("content", "")
                details = step_info.get("details", "")
                
                # Style based on step type
                if step_type == "planning":
                    with st.expander(f"{title}", expanded=False):
                        st.info(content)
                        if details:
                            st.caption(details)
                elif step_type == "decomposition":
                    with st.expander(f"{title}", expanded=False):
                        st.markdown(content)
                        if details:
                            st.caption(details)
                elif step_type == "retrieval":
                    with st.expander(f"{title}", expanded=False):
                        st.markdown(f"**Query:** {content}")
                        if details:
                            st.caption(details)
                elif step_type == "intermediate_answer":
                    with st.expander(f"{title}", expanded=False):
                        st.markdown(content)
                        if details:
                            st.caption(details)
                        # Show full answer in expander
                        full_answer = step_info.get("full_answer", "")
                        if full_answer and len(full_answer) > len(content):
                            with st.expander("View full intermediate answer", expanded=False):
                                st.markdown(full_answer)
                        # Show sources if available
                        sources = step_info.get("sources", [])
                        if sources:
                            st.caption(f"Sources: {len(sources)}")
                elif step_type == "synthesis":
                    with st.expander(f"{title}", expanded=False):
                        st.markdown(content)
                        if details:
                            st.caption(details)
                elif step_type == "verification":
                    with st.expander(f"{title}", expanded=False):
                        st.markdown(content)
                        if details:
                            st.caption(details)
                elif step_type == "verification_result":
                    with st.expander(f"{title}", expanded=False):
                        st.markdown(content)
                        if details:
                            st.caption(details)
                        # Show verification details if available
                        verification_data = step_info.get("verification_results", [])
                        if verification_data:
                            supports = [r for r in verification_data if r.get("status") == "Supports"]
                            refutes = [r for r in verification_data if r.get("status") == "Refutes"]
                            not_mentioned = [r for r in verification_data if r.get("status") == "Not Mentioned"]
                            
                            if supports:
                                st.success(f"{len(supports)} claim(s) verified against sources")
                            if refutes:
                                st.error(f"{len(refutes)} claim(s) contradicted by sources")
                            if not_mentioned:
                                st.warning(f"{len(not_mentioned)} claim(s) extrapolated from sources")
                elif step_type == "final_answer":
                    st.success(f"{title}")
                    if details:
                        st.caption(details)


def display_answer_with_verification(answer_text: str, verification_results: list[dict]) -> None:
    """Display answer with verification status indicators."""
    import re
    
    if not verification_results:
        st.markdown(answer_text)
        return
    
    # Create a mapping of claims to status
    claim_to_status = {r.get("claim", ""): r.get("status", "Not Mentioned") for r in verification_results}
    
    # Track which claims have been matched to avoid duplicate badges
    matched_claims = set()
    
    # Clean answer text (remove citations section for processing)
    answer_only = answer_text.split("**References:**")[0].strip() if "**References:**" in answer_text else answer_text
    
    # Split answer into sentences and add verification markers
    sentences = re.split(r'([.;]\s+|\n+)', answer_only)
    
    displayed_text = ""
    for sentence in sentences:
        sentence_clean = sentence.strip()
        if not sentence_clean or len(sentence_clean) < 10:
            displayed_text += sentence
            continue
        
        # Check if any claim matches this sentence (fuzzy match)
        # Only match claims that haven't been matched yet
        status = None
        best_match = None
        best_match_score = 0
        
        for claim, claim_status in claim_to_status.items():
            if claim and len(claim) > 15 and claim not in matched_claims:
                # Check if claim text appears in sentence (first 50 chars for matching)
                claim_preview = claim[:50].lower().strip()
                sentence_preview = sentence_clean[:100].lower().strip()
                
                match_score = 0
                # Check for substring match (exact match is best)
                if claim_preview in sentence_preview:
                    match_score = 100  # High score for substring match
                    status = claim_status
                    best_match = claim
                    best_match_score = match_score
                    break  # Exact match, stop searching
                # Check word overlap for longer claims
                claim_words = set(w for w in claim_preview.split() if len(w) > 3)
                sentence_words = set(w for w in sentence_preview.split() if len(w) > 3)
                if claim_words and len(claim_words & sentence_words) >= 2:
                    # Calculate overlap score (more overlap = higher score)
                    overlap_score = len(claim_words & sentence_words) / len(claim_words) if claim_words else 0
                    match_score = overlap_score * 50  # Medium score for word overlap
                    if match_score > best_match_score:
                        status = claim_status
                        best_match = claim
                        best_match_score = match_score
        
        # Only add badge if we found a match and it hasn't been used yet
        if best_match and best_match not in matched_claims:
            matched_claims.add(best_match)
            if status == "Supports":
                # Add verified badge
                displayed_text += f'{sentence}<span class="verification-badge verified">Verified</span>'
            elif status == "Refutes":
                # Add refuted badge
                displayed_text += f'{sentence}<span class="verification-badge refuted">Refuted</span>'
            elif status == "Not Mentioned":
                # Add unverified badge
                displayed_text += f'{sentence}<span class="verification-badge unverified">Extrapolated</span>'
            else:
                displayed_text += sentence
        else:
            displayed_text += sentence
    
    st.markdown(displayed_text, unsafe_allow_html=True)
    
    # Show verification summary in expander
    with st.expander("Verification Details", expanded=False):
        supports = [r for r in verification_results if r.get("status") == "Supports"]
        refutes = [r for r in verification_results if r.get("status") == "Refutes"]
        not_mentioned = [r for r in verification_results if r.get("status") == "Not Mentioned"]
        
        # Create styled summary badges
        summary_html = "<div style='display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px;'>"
        
        if supports:
            summary_html += f'<span class="verification-badge verified">{len(supports)} Verified</span>'
        if refutes:
            summary_html += f'<span class="verification-badge refuted">{len(refutes)} Refuted</span>'
        if not_mentioned:
            summary_html += f'<span class="verification-badge unverified">{len(not_mentioned)} Extrapolated</span>'
        
        summary_html += "</div>"
        st.markdown(summary_html, unsafe_allow_html=True)
        
        if supports:
            st.info(f"**{len(supports)} claim(s) verified** against source documents")
        if refutes:
            st.error(f"**{len(refutes)} claim(s) contradicted** by source documents")
        if not_mentioned:
            st.warning(f"**{len(not_mentioned)} claim(s) extrapolated** from source documents")
        
        # Show individual claim details
        if verification_results:
            st.divider()
            st.caption("Individual Claims:")
            for i, result in enumerate(verification_results, 1):
                claim = result.get("claim", "")  # Show full claim, no truncation
                status = result.get("status", "Not Mentioned")
                
                if status == "Supports":
                    badge_class = "verification-badge verified claim-item-badge"
                elif status == "Refutes":
                    badge_class = "verification-badge refuted claim-item-badge"
                else:
                    badge_class = "verification-badge unverified claim-item-badge"
                
                badge_html = f'<span class="{badge_class}">{"Verified" if status == "Supports" else "Refuted" if status == "Refutes" else "Extrapolated"}</span>'
                
                # Use flexbox layout to ensure wrapped text aligns with "Claim X:" label
                claim_html = f"""
                <div style="display: flex; align-items: flex-start; margin-bottom: 0.75rem; gap: 8px;">
                    <div style="flex-shrink: 0;">
                        {badge_html}
                    </div>
                    <div style="flex: 1; min-width: 0;">
                        <strong>Claim {i}:</strong> {claim}
                    </div>
                </div>
                """
                st.markdown(claim_html, unsafe_allow_html=True)


def chat_ui(query_pipeline: QueryPipeline):
    """Main chat interface."""
    st.header("Research Assistant")
    
    if not st.session_state["ingested_docs"]:
        st.info("Upload documents above to start asking questions about your research materials.")
        # Show placeholder chat interface even without documents
        return
    
    # Get document IDs for current session to filter search results
    # Updated to handle new format: (doc_id, filename, source_uri, count, title, author)
    session_doc_ids = []
    for doc_info in st.session_state["ingested_docs"]:
        if len(doc_info) >= 4:
            session_doc_ids.append(doc_info[0])  # doc_id is first element
    
    # Display chat history with verification status and trajectory
    # Use cached rendering to avoid unnecessary recomputation
    messages = st.session_state.get("messages", [])
    for idx, (role, content) in enumerate(messages):
        with st.chat_message(role):
            if role == "assistant":
                # Extract answer and sources from content first
                answer_only = content.split("**References:**")[0].strip() if "**References:**" in content else content
                
                # Extract sources from content
                sources = []
                if "**References:**" in content:
                    refs_text = content.split("**References:**", 1)[1].strip()
                    # Extract sources from reference list (handles both bullet points and numbered lists)
                    import re
                    # Match s3:// URIs, handling both "- s3://..." and "1. s3://..." formats
                    sources = re.findall(r's3://[^\s\n]+', refs_text)
                    sources = [s.rstrip('-').strip().lstrip('0123456789. ').strip() for s in sources]
                    sources = [s for s in sources if s.startswith('s3://')]  # Filter to only valid URIs
                
                # Check for trajectory first
                trajectory_shown = False
                if "agent_trajectory" in st.session_state and st.session_state["agent_trajectory"]:
                    # Find trajectory for this message (match by answer content)
                    for traj_entry in st.session_state["agent_trajectory"]:
                        # Try to match by checking if the answer is in the content
                        if traj_entry.get("answer") and traj_entry["answer"] in content:
                            trajectory_data = traj_entry.get("trajectory")
                            if trajectory_data:
                                display_agent_trajectory(traj_entry.get("query", ""), trajectory_data)
                                trajectory_shown = True
                                break
                
                # Check if verification results are available
                verification_results = []
                if "verification_results" in st.session_state:
                    verification_data = st.session_state["verification_results"]
                    for verif in verification_data:
                        if verif.get("answer") == answer_only:
                            verification_results = verif.get("verification", [])
                            # Use sources from verification if available, otherwise keep extracted sources
                            sources = verif.get("sources", sources)
                            break
                
                if verification_results:
                    display_answer_with_verification(answer_only, verification_results)
                    # Show references separately if they exist
                    if "**References:**" in content:
                        references = content.split("**References:**", 1)[1].strip()
                        st.markdown(f"\n\n**References:**\n{references}")
                else:
                    # No verification found, just display the content
                    if not trajectory_shown:
                        st.markdown(content)
                
                # Add "Add to Report" button inside chat message context
                # Use sources extracted above
                sources_final = sources
                
                # Extract query from previous user message
                query_text = ""
                if idx > 0 and messages[idx - 1][0] == "user":
                    query_text = messages[idx - 1][1]
                
                # Place button inside the chat message, after content
                button_key = f"add_report_{idx}"
                # Use a stable key based on answer content hash
                answer_hash = hash((answer_only, str(sources_final)))
                report_key = f"_report_item_{answer_hash}"
                
                # Check if answer is already in the report (check both flag and text)
                report_text = st.session_state.get("report_text", "")
                # Check if the answer content is actually present in the report
                # Use first 100 chars of answer to check if it's in report (more reliable than full text)
                answer_snippet = answer_only[:100].strip() if len(answer_only) > 100 else answer_only.strip()
                is_in_text = answer_snippet in report_text if answer_snippet else False
                is_flagged = st.session_state.get(report_key, False)
                is_already_added = is_in_text or is_flagged
                
                if is_already_added:
                    # Show disabled button if already added
                    st.button("Already in Report", key=f"report_status_{idx}", disabled=True, use_container_width=False)
                else:
                    # Show active button
                    if st.button("Add to Report", key=button_key, use_container_width=False):
                        # Add to report immediately
                        add_to_report(answer_only, sources_final, query_text)
                        # Set flag to prevent duplicate additions
                        st.session_state[report_key] = True
                        # Rerun to update button state
                        st.rerun()
            else:
                st.markdown(content)

    # Chat input
    user_input = st.chat_input("Ask a research question about your documents…")
    if user_input:
        st.session_state["messages"].append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # Check if this query used the orchestrator (has trajectory)
            trajectory_data = None
            
            # Create dynamic status display using empty container
            status_container = st.empty()
            
            def update_status(status_text: str):
                """Update the status text dynamically"""
                # Use status container to update text with pulsing animation (no background, no emoji)
                with status_container.container():
                    st.markdown(f'<p class="status-update-text">{status_text}</p>', unsafe_allow_html=True)
            
            # Store status callback in session state so orchestrator/pipeline can use it
            st.session_state["_status_callback"] = update_status
            
            # Start with initial status
            update_status("Searching documents and generating answer...")
            
            # Filter search to only use documents from current session
            answer, sources = query_pipeline.answer(user_input, allowed_doc_ids=session_doc_ids)
            
            # Clear status and callback after processing
            status_container.empty()
            if "_status_callback" in st.session_state:
                del st.session_state["_status_callback"]
                
                # Get trajectory if available (check after answer is generated)
                if "agent_trajectory" in st.session_state and st.session_state["agent_trajectory"]:
                    # Get the latest trajectory
                    latest_traj = st.session_state["agent_trajectory"][-1]
                    if latest_traj.get("query") == user_input:
                        trajectory_data = latest_traj.get("trajectory")
            
            # Show trajectory if available (display before answer)
            if trajectory_data:
                display_agent_trajectory(user_input, trajectory_data)
            
            # Get verification results if available
            verification_results = []
            if "verification_results" in st.session_state and st.session_state["verification_results"]:
                # Get the latest verification result
                latest_verif = st.session_state["verification_results"][-1]
                if latest_verif.get("answer") == answer:
                    verification_results = latest_verif.get("verification", [])
            
            # Display answer with verification
            display_answer_with_verification(answer, verification_results)
            
            # Append citations separately
            if sources:
                citations = "\n\n**References:**\n" + "\n".join([f"- {s}" for s in sources])
                st.markdown(citations)
                full_response = answer + citations
            else:
                full_response = answer
            
            # Add "Add to Report" button inside chat message context
            # Use a stable key based on answer content hash
            answer_hash = hash((answer, str(sources)))
            report_key = f"_report_new_{answer_hash}"
            
            # Check if answer is already in the report (check both flag and text)
            report_text = st.session_state.get("report_text", "")
            # Check if the answer content is actually present in the report
            # Use first 100 chars of answer to check if it's in report (more reliable than full text)
            answer_snippet = answer[:100].strip() if len(answer) > 100 else answer.strip()
            is_in_text = answer_snippet in report_text if answer_snippet else False
            is_flagged = st.session_state.get(report_key, False)
            is_already_added = is_in_text or is_flagged
            
            if is_already_added:
                # Show disabled button if already added
                st.button("Already in Report", key="report_status_new", disabled=True, use_container_width=False)
            else:
                # Show active button
                if st.button("Add to Report", key="add_report_new", use_container_width=False):
                    # Add to report immediately
                    add_to_report(answer, sources, user_input)
                    # Set flag to prevent duplicate additions
                    st.session_state[report_key] = True
                    # Rerun to update button state
                    st.rerun()
            
            st.session_state["messages"].append(("assistant", full_response))


def research_assistant_page(ingestion: IngestionPipeline, query_pipeline: QueryPipeline):
    """Research Assistant page - separate route."""
    st.title("VeriCite")
    st.caption("Research Assistant • Powered by watsonx.ai • Hybrid Search • Granite-3-8B-Instruct")
    
    # Upload section in main area (can be hidden)
    upload_section(ingestion)
    
    # Chat interface
    chat_ui(query_pipeline)


def research_report_page():
    """Research Report page - separate route."""
    st.title("VeriCite")
    st.caption("Research Report • Curate and Export Your Findings")
    
    report_page()


@st.cache_resource
def get_settings():
    """Cache settings loading to avoid reloading on every rerun."""
    load_dotenv()
    return Settings.from_env()


@st.cache_resource
def get_pipelines(settings):
    """Cache pipeline initialization to avoid recreating on every rerun."""
    return IngestionPipeline(settings), QueryPipeline(settings)


def main():
    # Set page config for VeriCite branding
    st.set_page_config(
        page_title="VeriCite - Research Assistant",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load settings (cached)
    settings = get_settings()
    init_state()
    
    # Inject custom CSS for medical research UI (optimized)
    inject_custom_css()

    # Get pipelines (cached)
    ingestion, query_pipeline = get_pipelines(settings)

    # Initialize navigation state
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "assistant"
    
    # Navigation in sidebar - separate pages
    st.sidebar.markdown("### Navigation")
    
    # Get current page from query params or session state
    try:
        query_params = st.query_params
        if "page" in query_params:
            current_page = query_params["page"]
            st.session_state["current_page"] = current_page
        else:
            current_page = st.session_state.get("current_page", "assistant")
    except:
        # Fallback if query_params not available
        current_page = st.session_state.get("current_page", "assistant")
    
    # Check if answer generation is in progress
    is_generating = "_status_callback" in st.session_state
    
    # Navigation links - optimized to reduce reruns
    # Disable navigation during generation to prevent interruption
    nav_assistant = st.sidebar.button("Research Assistant", key="nav_assistant", use_container_width=False, type="secondary", disabled=is_generating)
    nav_report = st.sidebar.button("Research Report", key="nav_report", use_container_width=False, type="secondary", disabled=is_generating)
    
    if is_generating:
        st.sidebar.caption("⚠️ Answer generation in progress...")
    
    # Handle navigation - only rerun if page actually changes and not during generation
    if nav_assistant and not is_generating:
        if st.session_state.get("current_page") != "assistant":
            st.session_state["current_page"] = "assistant"
            try:
                st.query_params.page = "assistant"
            except:
                pass
            st.rerun()
    
    if nav_report and not is_generating:
        if st.session_state.get("current_page") != "report":
            st.session_state["current_page"] = "report"
            try:
                st.query_params.page = "report"
            except:
                pass
            st.rerun()
    
    # Display content based on selected page - completely separate pages
    if current_page == "assistant":
        research_assistant_page(ingestion, query_pipeline)
    elif current_page == "report":
        research_report_page()
    
    # Sidebar: Documents list below navigation
    st.sidebar.divider()
    sidebar_documents()


if __name__ == "__main__":
    main()


