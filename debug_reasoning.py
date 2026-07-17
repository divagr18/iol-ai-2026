import json
import urllib.request

payload = {
    "model": "local",
    "messages": [
        {"role": "system", "content": "You are an expert linguist solving International Linguistics Olympiad problems. Answer every numbered item. Put each answer on its own line, with NO numbering and NO extra text. NEVER show your reasoning, thinking, or analysis. Output ONLY the final answers, nothing else. Output only the missing form for each blank."},
        {"role": "user", "content": "Given these forms:\n1. ɨnetkʼa = 'I saw you'\n2. ɨkɨrʼo = 'I see him'\n\nFill in the blanks:\n1. ______ = 'I saw him'\n2. ______ = 'I see you'"}
    ],
    "max_tokens": 512,
    "temperature": 0.0,
    "stream": False,
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(
    "http://127.0.0.1:8080/v1/chat/completions",
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=60) as resp:
    result = json.loads(resp.read().decode("utf-8"))

print("=== FULL RESPONSE ===")
print(json.dumps(result, indent=2, ensure_ascii=False))
print()

content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
print("=== CONTENT ===")
print(repr(content))
print()
print("=== PRETTY ===")
print(content)
