"""
Interactive medical symptom chatbot — terminal version.

Usage:
    python /scratch/budayaku/SML/src/chatbot_demo.py

Requires the same environment that has: xgboost, transformers, torch,
and the trained model artifacts in /scratch/budayaku/SML/model/.
"""

import sys
import os

# Make sure we can import extractor from the same folder
sys.path.insert(0, "/scratch/budayaku/SML/src")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import extractor

# ---------------------------------------------------------------------------
# Load nanochat
# ---------------------------------------------------------------------------
print("Loading nanochat model (may take ~30 seconds)...")
MODEL_ID = "nanochat-students/d20-chat-transformers"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nanochat_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
).to(device)
nanochat_model.eval()
print(f"✓ nanochat-d20 loaded on {device}")


def chat(user_message: str, max_new_tokens: int = 150, temperature: float = 0.7) -> str:
    """Send a single-turn message to nanochat and get back a reply."""
    conversation = [{"role": "user", "content": user_message}]
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)
    with torch.no_grad():
        outputs = nanochat_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    reply = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return reply.strip()


# ---------------------------------------------------------------------------
# Conversation orchestrator
# ---------------------------------------------------------------------------
EMERGENCY_KEYWORDS = [
    "crushing chest pain", "chest pain and left arm",
    "cant move", "cannot move", "face drooping", "slurring my words",
    "suicidal", "kill myself", "want to die", "end my life",
    "severe bleeding", "unconscious", "not breathing",
    "overdose", "poisoning",
]

EMERGENCY_REPLY = (
    "⚠️  What you're describing could be a medical emergency.\n"
    "    Please call emergency services immediately — 911 (US), 108 (India),\n"
    "    or your local emergency number. Do not wait."
)

MIN_SYMPTOMS_TO_PREDICT = 3
MIN_CONFIDENCE_TO_PREDICT = 0.20
MAX_TURNS_BEFORE_FORCE_PREDICT = 5


def check_emergency(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in EMERGENCY_KEYWORDS)


def new_session() -> dict:
    return {"symptoms": [], "turn": 0, "user_history": []}


def generate_followup(current_symptoms: list) -> str:
    if current_symptoms:
        sym_str = ", ".join(current_symptoms)
        prompt = (
            f"You are a caring medical assistant helping gather symptoms. "
            f"The patient has mentioned: {sym_str}. "
            f"Ask them ONE short follow-up question to learn about any other symptoms "
            f"they might have. Be warm and brief."
        )
    else:
        prompt = (
            "You are a caring medical assistant. Ask the patient ONE short question "
            "about what they're feeling. Be warm and brief."
        )
    return chat(prompt, max_new_tokens=60, temperature=0.7)


def generate_diagnosis_reply(symptoms: list, predictions: list) -> str:
    top3 = predictions[:3]
    sym_str = ", ".join(symptoms)
    pred_str = "; ".join(f"{name} ({prob*100:.0f}% match)" for name, prob in top3)
    prompt = (
        f"You are a caring medical assistant. Based on the symptoms ({sym_str}), "
        f"the most likely conditions are: {pred_str}. "
        f"Tell the patient this in a warm, clear way in 2-3 sentences. "
        f"End by strongly recommending they see a real doctor for diagnosis."
    )
    return chat(prompt, max_new_tokens=180, temperature=0.6)


def chatbot_turn(user_msg: str, state: dict) -> tuple:
    state["turn"] += 1
    state["user_history"].append(user_msg)

    if check_emergency(user_msg):
        return EMERGENCY_REPLY, state

    new_symptoms = extractor.extract_symptoms(user_msg)
    for s in new_symptoms:
        if s not in state["symptoms"]:
            state["symptoms"].append(s)

    should_predict = False
    predictions = []
    if len(state["symptoms"]) >= MIN_SYMPTOMS_TO_PREDICT:
        predictions = extractor.predict_from_symptom_list(state["symptoms"], k=5)
        if predictions and predictions[0][1] >= MIN_CONFIDENCE_TO_PREDICT:
            should_predict = True
    if state["turn"] >= MAX_TURNS_BEFORE_FORCE_PREDICT and state["symptoms"]:
        predictions = extractor.predict_from_symptom_list(state["symptoms"], k=5)
        should_predict = True

    if should_predict:
        reply = generate_diagnosis_reply(state["symptoms"], predictions)
        reply += f"\n\n(Symptoms I noted: {', '.join(state['symptoms'])})"
    else:
        reply = generate_followup(state["symptoms"])

    return reply, state


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------
def main():
    session = new_session()

    print()
    print("=" * 70)
    print("🩺  MEDICAL SYMPTOM ASSISTANT  —  Research Prototype")
    print("=" * 70)
    print("⚠️  NOT medical advice. Consult a real doctor for health concerns.")
    print("    In emergencies, call 911 (US) / 108 (India) immediately.")
    print()
    print("Describe your symptoms in plain English. Type 'quit' to exit,")
    print("'reset' to start a new conversation.")
    print("=" * 70)
    print()

    while True:
        try:
            user_msg = input("👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! Take care of yourself. 🩺")
            break

        if not user_msg:
            continue

        if user_msg.lower() in ("quit", "exit", "stop", "bye"):
            print("\nGoodbye! Take care of yourself. 🩺")
            break

        if user_msg.lower() == "reset":
            session = new_session()
            print("\n[🔄 New conversation started]\n")
            continue

        try:
            reply, session = chatbot_turn(user_msg, session)
        except Exception as e:
            print(f"\n⚠️  Error: {e}\n")
            continue

        print()
        print(f"🤖 Bot: {reply}")
        print(f"   [symptoms so far: {session['symptoms']}]")
        print()


if __name__ == "__main__":
    main()