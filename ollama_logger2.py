import requests
import json
import re
from datetime import datetime

# ==========================
# Konfiguration
# ==========================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:4b"
LOGFILE = "ollama_log.jsonl"   # JSON Lines-format (1 rad = 1 interaktion)

# ==========================
# Heuristisk koddetektion
# ==========================
def looks_like_code(text: str) -> bool:
    if not text:
        return False

    patterns = [
        r"```",                    # Markdown code block
        r"\bdef\s+\w+\(",          # Python-funktion
        r"\bclass\s+\w+",          # Klass
        r"#include\s+<",           # C/C++
        r"\{|\}",                  # Klammrar
        r";\s*$",                  # Semikolon på radslut
        r"^\s{2,}\S+",             # Indenterade kodrader
        r"\b(import|from)\s+\w+",  # Python-import
    ]

    return any(re.search(p, text, re.MULTILINE) for p in patterns)

# ==========================
# Ollama-anrop
# ==========================
def query_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()

    data = response.json()
    return data.get("response", "")

# ==========================
# Loggning
# ==========================
def log_interaction(prompt: str, response: str):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": MODEL,
        "input": {
            "content": prompt,
            "is_code": looks_like_code(prompt)
        },
        "output": {
            "content": response,
            "is_code": looks_like_code(response)
        }
    }

    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ==========================
# Main-loop (REPL)
# ==========================
def main():
    print(f"Ollama logger started (model={MODEL})")
    print("Skriv 'exit' eller 'quit' för att avsluta.\n")

    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nAvslutar.")
            break

        if prompt.strip().lower() in {"exit", "quit"}:
            break

        response = query_ollama(prompt)
        print(response)
        log_interaction(prompt, response)

# ==========================
# Entrypoint
# ==========================
if __name__ == "__main__":
    main()
