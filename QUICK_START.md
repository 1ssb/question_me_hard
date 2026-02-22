# Quick Reference: question_me_hard

## ⚡ Quick Start (RECOMMENDED)

```bash
# Setup: Create api.txt with your Groq API key
echo "gsk_YOUR_KEY_HERE" > api.txt

# Run the interactive clarification session
python clarify.py
```

The script will:
1. Ask what you want to build
2. Ask clarifying questions one-by-one
3. Assess clarity of your answers
4. Generate a machine-readable spec contract
5. Save to `contract.json`

## 🔄 What Happens (Simplified)

```
"Create a function that processes data efficiently"
                    ↓
        [LLM asks 7 clarifying questions]
            - What are input types?
            - What's the output format?
            - What are constraints?
            - How handle edge cases?
            - What environment (Python version)?
            - Deterministic?
            - Error handling?
                    ↓
            [User/System answers each]
                    ↓
        [Spec converges: All fields KNOWN]
                    ↓
    [LLM synthesizes unambiguous contract]
    {
      "function_signature": "def process_data(...)",
      "docstring": "Process data efficiently...",
      "examples": "process_data([1,2,3])",
      "test_cases": ["empty list", "single item", ...],
      "error_handling": "Return error dict, never raise"
    }
                    ↓
        [Coding agent can now implement perfectly]
```

---

## 📖 More Info

- **Core docs:** `README.md`
- **Setup:** `SETUP.md`
