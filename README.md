# question_me_hard

LLM agent asks you clarifying questions until your vague prompt becomes a complete specification.

## Quick Start

### 1. Set Up Your API Key

```bash
# Create api.txt with your Groq API key
echo 'gsk_YOUR_KEY_HERE' > api.txt
```

Get a free API key: https://console.groq.com/keys

**Important**: `api.txt` is in `.gitignore` - never committed to git

### 2. Run Interactive Clarification

```bash
python clarify.py
```

Type your vague idea. Answer one question at a time. That's it!

## How It Works

1. **You enter**: Vague prompt (e.g., "Build a cache system")
2. **LLM asks**: One focused question at a time
3. **You answer**: Be specific and honest
4. **LLM assesses**: Is this answer clear enough?
5. **Repeat**: Until the spec is complete
6. **Output**: `contract.json` - ready for implementation

## Features

- ✅ One question at a time (not overwhelming)
- ✅ LLM decides if answer is clear
- ✅ Simple, focused conversation flow
- ✅ Shows your final specification
- ✅ Generates machine-readable contract

## Output

`contract.json` - Contains:
- Function signature
- Docstring with args/returns/raises
- Usage examples
- Test cases
- Error handling rules

Ready to pass to a coding agent for implementation.

## Documentation

- [SETUP.md](SETUP.md) - Detailed setup guide

