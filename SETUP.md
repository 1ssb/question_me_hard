# Setup Guide

## Getting Started

### 1. Obtain Your Groq API Key

1. Go to [Groq Console](https://console.groq.com/keys)
2. Sign up or log in
3. Generate a new API key
4. Copy the key (starts with `gsk_`)

### 2. Create api.txt

Create a file named `api.txt` in the project root with your API key:

```bash
# Option 1: Echo to file
echo 'gsk_YOUR_KEY_HERE' > api.txt

# Option 2: Copy and edit the template
cp api.txt.example api.txt
# Then edit api.txt and replace with your actual key
```

**Important**: `api.txt` is in `.gitignore` and will NOT be committed to git. This keeps your API key safe.

### 3. Run the Clarification System

```bash
# Start interactive clarification (waits for your answers)
python clarify.py
```

## Clarification Levels

For `smart_clarify.py`:
- `quick` - Minimal clarification (60% clarity threshold)
- `normal` - Standard depth (75% clarity threshold) - **default**
- `thorough` - Deep exploration (95% clarity threshold)

## Output Files

After each session, you'll get:
- `contract.json` - Machine-readable specification ready for coding agents

## Troubleshooting

**Error: "api.txt not found"**
- Create the file: `echo 'gsk_YOUR_KEY' > api.txt`

**Error: "api.txt is empty"**
- Add your actual API key to the file (starts with `gsk_`)

**Getting slow responses**
- Groq free tier has rate limits (9,000 requests/day)
- Consider using `example_mock.py` for testing

**Want to keep API key secret**
- `api.txt` is already in `.gitignore` ✓
- Never commit it to git
- Don't share it publicly
