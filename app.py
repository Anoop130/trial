from flask import Flask, request, jsonify
import requests
import json
import os
from datetime import datetime

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
JOURNAL_FILE = "journal.jsonl"  # Each line is a JSON entry

@app.route("/respond", methods=["POST"])
def respond():
    user_input = request.json.get("text", "").strip()

    if not user_input:
        return jsonify({"error": "No input provided."}), 400

    prompt = f"""You are a supportive mental wellness companion.
The user said: "{user_input}"
Respond in one calm, caring paragraph with encouragement and reflection."""

    try:
        # Call Gemma 3n via Ollama
        response = requests.post(OLLAMA_URL, json={
            "model": "gemma3n:e2b",
            "prompt": prompt,
            "stream": False
        })
        response.raise_for_status()
        ai_reply = response.json().get("response", "").strip()

        # Log the conversation
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "input": user_input,
            "response": ai_reply
        }

        with open(JOURNAL_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return jsonify({"reply": ai_reply})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Model call failed: {e}"}), 500

if __name__ == "__main__":
    app.run(port=5000)
