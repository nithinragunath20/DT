
import requests

URL = "http://localhost:11434/api/generate"
CUSTOM_MODEL = "cybersecurity-slm"
ENGLISH_MODEL = "qwen3:0.6b"


def generate(model, prompt):
    print(f"\nRunning: {model}")

    response = requests.post(
        URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": 0,
            "options": {
                "temperature": 0.1,
                "num_ctx": 1024,
                "num_predict": 160
            }
        },
        timeout=900
    )

    if not response.ok:
        raise RuntimeError(
            f"{model} failed: {response.status_code}\n"
            f"{response.text}"
        )

    return response.json().get("response", "").strip()


print("CYBERSECURITY SLM - ENGLISH INTERFACE")
print("Type exit to quit.")

while True:
    question = input("\nYou: ").strip()

    if question.lower() == "exit":
        break

    if not question:
        continue

    try:
        telemetry = generate(CUSTOM_MODEL, question)

        print("\n========== ORIGINAL SLM OUTPUT ==========")
        print(telemetry or "(No output)")
        print("=========================================")

        if not telemetry:
            print("\nAssistant: The custom model returned no data.")
            continue

        prompt = f"""
You are a cybersecurity telemetry interpreter.

User request:
{question}

Generated telemetry:
{telemetry}

Explain ONLY what is supported by the generated telemetry.

Rules:
- Never invent missing fields or values.
- Do not treat generated telemetry as real system activity.
- If the record is incomplete, clearly mention this.
- Do not assume an event is malicious.
- If the telemetry does not answer the user's question,
  state that the available information is insufficient.
- Respond in simple, concise English.
"""

        explanation = generate(ENGLISH_MODEL, prompt)

        print("\n========== ENGLISH EXPLANATION ==========")
        print(explanation or "No explanation generated.")
        print("=========================================")

    except requests.exceptions.ConnectionError:
        print("\nError: Ollama server is not running.")

    except Exception as error:
        print(f"\nError: {error}")
