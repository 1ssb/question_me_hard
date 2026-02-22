#!/usr/bin/env python3
"""
Simple Interactive Clarification Session
One question at a time - user-driven conversation
"""

import json
import os
import sys
import signal
from question_me_hard import ClarificationSubagent, Memory
from question_me_hard.spec import FieldStatus, Spec
from question_me_hard.groq_llm import create_groq_llm


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Input timeout")


def get_input_with_fallback(prompt: str, timeout_secs: int = 60) -> str:
    """Get user input with timeout fallback"""
    try:
        # Set alarm for timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_secs)
        
        answer = input(prompt).strip()
        
        signal.alarm(0)  # Cancel alarm
        return answer
        
    except TimeoutException:
        print("\n⏱ Input timeout - skipping this question")
        return ""
    except (EOFError, KeyboardInterrupt):
        return ""
    except Exception as e:
        return ""


def load_api_key() -> str:
    """Load API key from api.txt"""
    try:
        with open("api.txt", "r") as f:
            key = f.read().strip()
            if not key:
                raise ValueError("api.txt is empty")
            return key
    except FileNotFoundError:
        print("❌ api.txt not found")
        print("\nSetup: Create api.txt with your Groq API key")
        print("  echo 'gsk_YOUR_KEY_HERE' > api.txt")
        print("\nGet free API key: https://console.groq.com/keys")
        sys.exit(1)


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_section(title: str):
    print(f"\n{title}")
    print("-" * 80)


def get_initial_prompt() -> str:
    """Get the initial vague prompt from user"""
    print_header("question_me_hard: Interactive Clarification")
    
    print("What do you want to build? (Be vague - we'll clarify it)")
    print("Examples: 'Build a cache system', 'Create a REST API', 'Sort an array'")
    print()
    
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "":
                if lines:
                    break
            else:
                lines.append(line)
        except EOFError:
            break
    
    prompt = "\n".join(lines).strip()
    if not prompt:
        prompt = "Build a function that processes input"
    
    return prompt


def assess_clarity(question: str, answer: str, llm_fn) -> tuple:
    """
    Ask LLM: Is this clear enough?
    Returns: (is_clear, explanation)
    """
    query = f"""Question: {question}

Answer: {answer}

Is the user's answer clear enough to understand what they want?
Or is there ambiguity/vagueness that needs more drilling?

Respond ONLY with:
CLEAR: [brief explanation of what you understood]
OR
UNCLEAR: [what's still vague - suggest ONE follow-up question]"""

    response = llm_fn(query).strip()
    
    if response.startswith("CLEAR:"):
        explanation = response.replace("CLEAR:", "").strip()
        return (True, explanation)
    else:
        # Extract the follow-up question
        explanation = response.replace("UNCLEAR:", "").strip()
        return (False, explanation)


def ask_one_question(spec_field: str, prompt: str, subagent: ClarificationSubagent, llm_fn) -> bool:
    """
    Ask ONE question about a spec field.
    Returns: True if user wants to continue to next field, False to ask more about this one
    """
    # Generate question
    query = f"""The user wants: "{prompt}"

Generate ONE focused question about the {spec_field} aspect.
Make it specific and actionable. Just the question, no explanation."""

    try:
        question = llm_fn(query).strip()
    except Exception as e:
        print(f"⚠ Error generating question: {e}")
        return True
    
    print(f"\n❓ {question}\n")
    
    # Get answer with timeout
    answer = get_input_with_fallback("→ Your answer: ")
    
    if not answer:
        print("⚠ Skipping this question (no input)")
        return False  # Ask again, but will reach max_attempts
    
    # Assess clarity
    try:
        is_clear, explanation = assess_clarity(question, answer, llm_fn)
    except Exception as e:
        print(f"⚠ Error assessing clarity: {e}")
        is_clear = False
        explanation = str(e)
    
    if is_clear:
        print(f"\n✓ Got it: {explanation}\n")
        try:
            subagent.spec.update(spec_field, explanation)
        except Exception as e:
            print(f"⚠ Error updating spec: {e}")
        return True
    else:
        print(f"\n[Need more clarity]")
        print(f"{explanation}\n")
        return False


def run_interactive_session():
    """Run the interactive clarification session"""
    
    # Load API key
    api_key = load_api_key()
    
    try:
        llm_fn = create_groq_llm(api_key)
        print("✓ Connected to Groq API\n")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        sys.exit(1)
    
    # Get initial prompt
    prompt = get_initial_prompt()
    
    print_section("Your Prompt")
    print(f'"{prompt}"\n')
    
    # Create subagent
    subagent = ClarificationSubagent(memory=Memory(), max_questions=50)
    
    # Ask about each field one at a time
    print_section("Clarification Session")
    print("Press Ctrl+C at any time to stop\n")
    
    fields_to_ask = ["inputs", "outputs", "constraints", "edge_cases", "failure_modes"]
    
    for field in fields_to_ask:
        label = subagent.spec.field_label(field)
        print(f"\n📌 Clarifying: {label}\n")
        
        # Keep asking about this field until satisfied
        satisfied = False
        attempts = 0
        max_attempts = 3
        
        while not satisfied and attempts < max_attempts:
            attempts += 1
            satisfied = ask_one_question(field, prompt, subagent, llm_fn)
        
        if not satisfied:
            print(f"⊙ Moving on (max questions reached for {field})")
        
        # Ask if user wants to continue or skip remaining
        if field != fields_to_ask[-1]:
            try:
                continue_input = input("\nContinue to next aspect? (y/n): ").strip().lower()
                if continue_input == 'n':
                    print("\nSkipping remaining aspects...")
                    break
            except EOFError:
                break
    
    # Fill unknowns
    print_section("Finalizing Specification")
    print("Filling in reasonable assumptions for remaining aspects...\n")
    
    for field in Spec.FIELD_NAMES:
        if subagent.spec.get_entry(field).status == FieldStatus.UNKNOWN:
            subagent._fill_with_assumptions(prompt, llm_fn)
            break
    
    # Show final spec
    print_section("Your Specification")
    for field in Spec.FIELD_NAMES:
        entry = subagent.spec.get_entry(field)
        icon = {
            FieldStatus.KNOWN: "✓",
            FieldStatus.ASSUMED: "⊙",
            FieldStatus.CONFLICTING: "⚠",
            FieldStatus.UNKNOWN: "?",
        }.get(entry.status, "?")
        
        value = (entry.value or "")[:60]
        print(f"  {icon}  {field:<18} {value}")
    
    # Generate contract
    print_section("Generating Implementation Contract")
    print("Converting spec to machine-readable contract...\n")
    
    try:
        contract = subagent.to_contract(prompt, llm_fn)
    except Exception as e:
        print(f"⚠ Contract generation error: {e}")
        contract = {}
    
    # Show contract
    print_section("Generated Contract")
    
    if not contract or all(not v for v in contract.values()):
        print("⚠ Contract is incomplete (not enough clarification)")
        print("\nHere's what we know so far:")
        for field in Spec.FIELD_NAMES:
            entry = subagent.spec.get_entry(field)
            print(f"  • {field}: {entry.value or '(unknown)'}")
    else:
        for section, content in contract.items():
            title = section.upper().replace("_", " ")
            print(f"\n▶ {title}")
            if content:
                lines = content.split("\n")
                for line in lines[:8]:
                    print(f"  {line}")
                if len(lines) > 8:
                    print(f"  ... [{len(lines) - 8} more lines]")
            else:
                print("  (empty)")
    
    # Save
    print_section("Session Complete")
    
    contract_file = "contract.json"
    try:
        with open(contract_file, "w") as f:
            json.dump(contract, f, indent=2)
        print(f"✓ Contract saved to: {contract_file}\n")
    except Exception as e:
        print(f"⚠ Could not save contract: {e}\n")
    
    # Show summary
    try:
        print_section("Summary")
        print(f"Your Prompt:\n  \"{prompt}\"")
        print("\nClarified Specification:")
        
        for field in Spec.FIELD_NAMES:
            try:
                entry = subagent.spec.get_entry(field)
                if entry.value:
                    value = entry.value[:70].replace("\n", " ")
                    status = entry.status.value
                    print(f"  [{status}] {field}: {value}")
            except Exception as e:
                print(f"  [error] {field}: {str(e)}")
        
        print("\n✓ Done! Ready to implement.")
    except Exception as e:
        print(f"⚠ Error displaying summary: {e}")
        print("\n✓ Session complete.")


if __name__ == "__main__":
    try:
        run_interactive_session()
    except KeyboardInterrupt:
        print("\n\n❌ Session interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
