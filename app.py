# enhanced_code_audit_app.py
# Enhanced version with improved security, error handling, performance, and features

import streamlit as st
import openai
import os
from transformers import pipeline
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from pathlib import Path

# === Configuration ===
@dataclass
class AuditConfig:
    """Configuration settings for the audit application"""
    max_code_length: int = 50000  # Maximum characters in code input
    max_history_items: int = 20   # Maximum audit history to keep
    supported_extensions: List[str] = None
    openai_model: str = "gpt-4"   # Updated to stable model
    hf_model: str = "bigcode/starcoderbase"
    
    def __post_init__(self):
        if self.supported_extensions is None:
            self.supported_extensions = ["py", "js", "java", "cpp", "ts", "jsx", "tsx", "c", "h", "cs", "php", "rb", "go", "rs", "kt"]

CONFIG = AuditConfig()

# === Categories with descriptions ===
CATEGORIES_WITH_DESC = {
    "Security": "Vulnerability assessment and security best practices",
    "Logic Correctness": "Algorithmic correctness and edge case handling", 
    "Performance": "Efficiency and optimization opportunities",
    "Scalability": "Ability to handle increased load and data volume",
    "Reusability / Modularity": "Code organization and reusability",
    "Readability / Maintainability": "Code clarity and ease of maintenance",
    "Testability": "How easily the code can be tested",
    "Best Practices / Modernity": "Adherence to modern coding standards",
    "AI-Generated Code Smells": "Patterns indicating potential AI-generated issues"
}

CATEGORIES = list(CATEGORIES_WITH_DESC.keys())

# === Logging Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Enhanced API Setup ===
def setup_openai():
    """Setup OpenAI API with proper error handling"""
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ OpenAI API key not found. Please set OPENAI_API_KEY in secrets or environment.")
        return False
    
    try:
        openai.api_key = api_key
        # Test the API key with a minimal request
        openai.Model.list()
        return True
    except Exception as e:
        st.error(f"⚠️ OpenAI API key validation failed: {str(e)}")
        return False

# === Streamlit App Config ===
st.set_page_config(
    page_title="Code Quality & Risk Evaluator", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Enhanced Session Management ===
@dataclass
class AuditResult:
    """Data class for audit results"""
    timestamp: datetime
    code_hash: str
    model_used: str
    scores: Dict[str, int]
    response: str
    code_snippet: str  # First 200 chars for preview
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "code_hash": self.code_hash,
            "model_used": self.model_used,
            "scores": self.scores,
            "response": self.response,
            "code_snippet": self.code_snippet
        }

def init_session():
    """Initialize session state with enhanced structure"""
    defaults = {
        "audit_history": [],
        "openai_available": None,
        "last_audit_result": None,
        "code_input": "",
        "analysis_cache": {}  # Cache for repeated analyses
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# === Enhanced Caching ===
@st.cache_resource(show_spinner=False)
def load_hf_pipeline():
    """Load Hugging Face pipeline with error handling"""
    try:
        return pipeline(
            "text-generation", 
            model=CONFIG.hf_model,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        logger.error(f"Failed to load HF pipeline: {e}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_analysis(code_hash: str, model: str, options: str) -> Optional[str]:
    """Check if we have a cached analysis for this code"""
    cache_key = f"{code_hash}_{model}_{options}"
    return st.session_state.analysis_cache.get(cache_key)

def cache_analysis(code_hash: str, model: str, options: str, result: str):
    """Cache analysis result"""
    cache_key = f"{code_hash}_{model}_{options}"
    st.session_state.analysis_cache[cache_key] = result
    
    # Limit cache size
    if len(st.session_state.analysis_cache) > 100:
        # Remove oldest entries
        keys = list(st.session_state.analysis_cache.keys())
        for key in keys[:20]:
            del st.session_state.analysis_cache[key]

# === Enhanced Prompt Building ===
def build_prompt(code: str, simple_mode: bool, suggest_fixes: bool, focus_areas: List[str] = None) -> str:
    """Build enhanced prompt with better structure"""
    
    base_prompt = f"""
You are a senior software architect and security-focused code auditor with 15+ years of experience.

TASK: Analyze the provided code and rate it from 0-10 on each category below.

RATING SCALE:
- 0-2: Critical issues, major flaws
- 3-4: Significant problems requiring attention  
- 5-6: Average quality with room for improvement
- 7-8: Good quality with minor issues
- 9-10: Excellent, production-ready code

CATEGORIES TO EVALUATE:
"""
    
    # Add category descriptions
    for i, (category, description) in enumerate(CATEGORIES_WITH_DESC.items(), 1):
        focus_indicator = " **[FOCUS AREA]**" if focus_areas and category in focus_areas else ""
        base_prompt += f"{i}. {category}{focus_indicator}: {description}\n"
    
    base_prompt += """
FORMAT YOUR RESPONSE EXACTLY AS:
**Category Name**: X/10 - Brief explanation (1-2 sentences)

Then provide:
**Overall Code Health Score**: X/10
**Summary**: 2-3 sentence overall assessment
"""
    
    # Add mode-specific instructions
    if simple_mode:
        base_prompt += "\n**IMPORTANT**: Explain all technical terms and concepts in beginner-friendly language."
    
    if suggest_fixes:
        base_prompt += "\n**IMPROVEMENTS**: After your analysis, provide 3-5 specific, actionable improvement suggestions with code examples where helpful."
    
    if focus_areas:
        base_prompt += f"\n**SPECIAL FOCUS**: Pay extra attention to: {', '.join(focus_areas)}"
    
    return f"{base_prompt}\n\n**CODE TO ANALYZE**:\n```\n{code}\n```"

# === Enhanced Score Extraction ===
def extract_scores(text: str) -> Tuple[pd.DataFrame, int]:
    """Extract scores with better regex and overall score"""
    scores = []
    overall_score = 0
    
    for category in CATEGORIES:
        # Multiple regex patterns for robustness
        patterns = [
            rf"\*\*{re.escape(category)}\*\*:?\s*(\d+(?:\.\d+)?)",
            rf"{re.escape(category)}:?\s*(\d+(?:\.\d+)?)/10",
            rf"{re.escape(category)}.*?(\d+(?:\.\d+)?)\s*/\s*10",
        ]
        
        score = 0
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                score = min(10, max(0, float(match.group(1))))  # Clamp between 0-10
                break
        
        scores.append(score)
    
    # Extract overall score
    overall_patterns = [
        r"Overall.*?Score.*?(\d+(?:\.\d+)?)",
        r"Total.*?Score.*?(\d+(?:\.\d+)?)",
        r"Final.*?Score.*?(\d+(?:\.\d+)?)"
    ]
    
    for pattern in overall_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            overall_score = min(10, max(0, float(match.group(1))))
            break
    
    df = pd.DataFrame({
        "Category": CATEGORIES, 
        "Score": scores,
        "Description": [CATEGORIES_WITH_DESC[cat] for cat in CATEGORIES]
    })
    
    return df, overall_score

# === Enhanced Visualization ===
def generate_enhanced_chart(df: pd.DataFrame, overall_score: int):
    """Generate enhanced visualization with multiple chart types"""
    
    # Color scheme based on scores
    colors = ['#d73027' if score < 4 else '#fee08b' if score < 7 else '#1a9850' for score in df['Score']]
    
    # Horizontal bar chart
    fig1 = px.bar(
        df, 
        x="Score", 
        y="Category", 
        orientation='h',
        title="📊 Code Quality Audit Scores",
        range_x=[0, 10],
        color="Score",
        color_continuous_scale="RdYlGn",
        hover_data={"Description": True}
    )
    
    fig1.update_layout(
        height=500,
        showlegend=False,
        title_font_size=16,
        xaxis_title="Score (0-10)",
        yaxis_title=""
    )
    
    # Add score labels on bars
    fig1.update_traces(texttemplate='%{x}', textposition='outside')
    
    # Radar chart for alternative view
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatterpolar(
        r=df['Score'].tolist() + [df['Score'].iloc[0]],  # Close the polygon
        theta=df['Category'].tolist() + [df['Category'].iloc[0]],
        fill='toself',
        name='Code Quality',
        line_color='rgb(0,100,200)'
    ))
    
    fig2.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        title="🎯 Code Quality Radar",
        height=400
    )
    
    return fig1, fig2

# === Enhanced AI Response ===
def get_model_response(prompt: str, model_choice: str) -> str:
    """Get AI response with enhanced error handling and retries"""
    
    try:
        if model_choice == "OpenAI (GPT-4)":
            if not st.session_state.openai_available:
                return "⚠️ OpenAI API not available. Please check your API key configuration."
            
            response = openai.ChatCompletion.create(
                model=CONFIG.openai_model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert code auditor. Provide detailed, accurate assessments following the exact format requested."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Lower temperature for more consistent scoring
                max_tokens=2000,
                timeout=30
            )
            return response.choices[0].message.content
            
        else:  # Hugging Face
            hf_pipeline = load_hf_pipeline()
            if not hf_pipeline:
                return "⚠️ Hugging Face model not available. Please try OpenAI or check your configuration."
            
            result = hf_pipeline(
                prompt, 
                max_length=min(len(prompt) + 800, 2048),
                do_sample=True, 
                temperature=0.7,
                pad_token_id=hf_pipeline.tokenizer.eos_token_id,
                return_full_text=False
            )[0]
            
            return result['generated_text']
            
    except openai.error.RateLimitError:
        return "⚠️ Rate limit exceeded. Please wait a moment and try again."
    except openai.error.InvalidRequestError as e:
        return f"⚠️ Invalid request: {str(e)}"
    except Exception as e:
        logger.error(f"Model response error: {str(e)}")
        return f"⚠️ Error getting response: {str(e)}. Please try again."

# === Code Validation ===
def validate_code_input(code: str) -> Tuple[bool, str]:
    """Validate code input"""
    if not code.strip():
        return False, "Please provide code to analyze."
    
    if len(code) > CONFIG.max_code_length:
        return False, f"Code too long. Maximum {CONFIG.max_code_length} characters allowed."
    
    # Check for potential security issues in input
    suspicious_patterns = [
        r'__import__\s*\(\s*["\']os["\']',
        r'eval\s*\(',
        r'exec\s*\(',
        r'subprocess\.',
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            return False, "Code contains potentially unsafe patterns."
    
    return True, ""

# === Main Audit Logic ===
def run_audit(code: str, model_choice: str, simple_mode: bool, suggest_fixes: bool, focus_areas: List[str] = None):
    """Enhanced audit logic with caching and better error handling"""
    
    # Validate input
    is_valid, error_msg = validate_code_input(code)
    if not is_valid:
        st.error(error_msg)
        return
    
    # Generate hash for caching
    code_hash = hashlib.md5(code.encode()).hexdigest()
    options_hash = f"{simple_mode}_{suggest_fixes}_{','.join(focus_areas or [])}"
    
    # Check cache first
    cached_result = get_cached_analysis(code_hash, model_choice, options_hash)
    if cached_result:
        st.info("🚀 Using cached result for faster response!")
        response = cached_result
    else:
        # Generate new analysis
        prompt = build_prompt(code, simple_mode, suggest_fixes, focus_areas)
        
        with st.spinner("🔍 Analyzing code... This may take a moment."):
            response = get_model_response(prompt, model_choice)
        
        # Cache the result
        cache_analysis(code_hash, model_choice, options_hash, response)
    
    # Display results
    st.markdown("## 📝 Detailed Audit Report")
    
    # Parse scores
    df, overall_score = extract_scores(response)
    
    # Create audit result
    audit_result = AuditResult(
        timestamp=datetime.now(),
        code_hash=code_hash,
        model_used=model_choice,
        scores={cat: score for cat, score in zip(CATEGORIES, df['Score'])},
        response=response,
        code_snippet=code[:200] + "..." if len(code) > 200 else code
    )
    
    # Add to history (limit size)
    st.session_state.audit_history.append(audit_result)
    if len(st.session_state.audit_history) > CONFIG.max_history_items:
        st.session_state.audit_history.pop(0)
    
    st.session_state.last_audit_result = audit_result
    
    # Display overall score prominently
    if overall_score > 0:
        score_color = "🟢" if overall_score >= 7 else "🟡" if overall_score >= 4 else "🔴"
        st.markdown(f"### {score_color} Overall Code Health: {overall_score}/10")
    
    # Display the full response
    st.markdown(response)
    
    # Generate and display charts
    fig1, fig2 = generate_enhanced_chart(df, overall_score)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)
    
    # Score summary table
    st.markdown("### 📋 Score Summary")
    st.dataframe(df[['Category', 'Score']], use_container_width=True)
    
    # Download options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            "📄 Download Report (Markdown)",
            response,
            file_name=f"code_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
    
    with col2:
        # JSON export
        json_data = json.dumps(audit_result.to_dict(), indent=2)
        st.download_button(
            "📊 Download Data (JSON)",
            json_data,
            file_name=f"audit_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col3:
        # CSV export for scores
        csv_data = df.to_csv(index=False)
        st.download_button(
            "📈 Download Scores (CSV)",
            csv_data,
            file_name=f"audit_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# === Main UI ===
def main():
    """Main application UI"""
    
    # Initialize session
    init_session()
    
    # Check OpenAI availability
    if st.session_state.openai_available is None:
        st.session_state.openai_available = setup_openai()
    
    # Header
    st.title("🔍 Enhanced Code Quality & Risk Evaluator")
    st.markdown("""
    **Advanced code analysis** with detailed scoring, caching, and comprehensive reporting.
    Upload files or paste code to get started.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_options = ["OpenAI (GPT-4)", "Hugging Face (StarCoder)"]
        if not st.session_state.openai_available:
            st.warning("⚠️ OpenAI unavailable")
            model_options = [opt for opt in model_options if "OpenAI" not in opt]
        
        model_choice = st.selectbox("🧠 AI Model", model_options)
        
        # Analysis options
        st.subheader("📋 Analysis Options")
        simple_mode = st.toggle("🧑‍🎓 Beginner-friendly explanations")
        suggest_fixes = st.toggle("🛠️ Include improvement suggestions")
        
        # Focus areas
        st.subheader("🎯 Focus Areas (Optional)")
        focus_areas = st.multiselect(
            "Select specific areas to emphasize:",
            CATEGORIES,
            help="Choose categories to give extra attention to"
        )
        
        # Statistics
        if st.session_state.audit_history:
            st.subheader("📈 Statistics")
            st.metric("Total Audits", len(st.session_state.audit_history))
            
            if st.session_state.last_audit_result:
                avg_score = sum(st.session_state.last_audit_result.scores.values()) / len(st.session_state.last_audit_result.scores)
                st.metric("Last Average Score", f"{avg_score:.1f}/10")
    
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Upload Code File",
        type=CONFIG.supported_extensions,
        help=f"Supported formats: {', '.join(CONFIG.supported_extensions)}"
    )
    
    # Code input handling
    code_input = ""
    
    if uploaded_file:
        try:
            code_input = uploaded_file.read().decode("utf-8")
            st.success(f"✅ Loaded {uploaded_file.name} ({len(code_input)} characters)")
            
            # Show file preview
            with st.expander("👀 File Preview"):
                st.code(code_input[:1000] + ("..." if len(code_input) > 1000 else ""))
                
        except UnicodeDecodeError:
            st.error("❌ Unable to decode file. Please ensure it's a UTF-8 encoded text file.")
    else:
        code_input = st.text_area(
            "✏️ Or paste your code here:",
            height=300,
            placeholder="def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            help=f"Maximum {CONFIG.max_code_length:,} characters"
        )
    
    # Character count
    if code_input:
        char_count = len(code_input)
        color = "red" if char_count > CONFIG.max_code_length else "green"
        st.caption(f":{color}[Characters: {char_count:,}/{CONFIG.max_code_length:,}]")
    
    # Run audit button
    if st.button("🚀 Run Code Audit", type="primary", disabled=not code_input):
        if code_input:
            run_audit(code_input, model_choice, simple_mode, suggest_fixes, focus_areas)
        else:
            st.warning("Please provide code to analyze.")
    
    # Audit history
    if st.session_state.audit_history and st.checkbox("📚 Show Audit History"):
        st.markdown("### 📋 Previous Audits")
        
        for idx, audit in enumerate(reversed(st.session_state.audit_history[-10:]), 1):
            with st.expander(f"Audit #{idx} - {audit.timestamp.strftime('%Y-%m-%d %H:%M')} ({audit.model_used})"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Code Preview:**")
                    st.code(audit.code_snippet)
                
                with col2:
                    st.markdown("**Scores:**")
                    for category, score in audit.scores.items():
                        color = "🟢" if score >= 7 else "🟡" if score >= 4 else "🔴"
                        st.write(f"{color} {category}: {score}/10")
                
                if st.button(f"Re-run Analysis #{idx}", key=f"rerun_{idx}"):
                    # This would need the original code, which we'd need to store
                    st.info("Feature coming soon: Re-run with current settings")

if __name__ == "__main__":
    main()
