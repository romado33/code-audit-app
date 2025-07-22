# code_audit_app.py
# A Streamlit prototype for the "Code Quality and Risk Evaluator" app
# This app lets users paste code and evaluates it across multiple dimensions

import streamlit as st
import openai

# Set your OpenAI key securely (you can also use Streamlit secrets)
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Title
st.title("🔍 Code Quality & Risk Evaluator")
st.markdown("Paste any code snippet below. The AI will analyze its security, performance, maintainability, and more.")

# Text input
code_input = st.text_area("Paste your code here:", height=300, placeholder="e.g., def add(a, b): return a + b")

# Audit button
if st.button("Run Audit") and code_input:
    # Construct the audit prompt
    prompt = f"""
You are a senior software architect and security expert. Analyze the following code and provide an audit report with scores from 0 to 10 for each of these categories:

1. Security
2. Logic Correctness
3. Performance
4. Scalability
5. Reusability / Modularity
6. Readability / Maintainability
7. Testability
8. Best Practices / Modernity
9. AI-Generated Code Smells

For each, include a 1-2 sentence explanation.

Then give an Overall Code Health Score and a short summary.

Code to audit:
{code_input}
"""

    # Send request to OpenAI
    with st.spinner("Auditing code..."):
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a code quality auditor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        audit_output = response.choices[0].message.content
        
    # Show result
    st.markdown("## 📝 Audit Report")
    st.markdown(audit_output)

elif not code_input:
    st.info("Paste some code to begin.")
