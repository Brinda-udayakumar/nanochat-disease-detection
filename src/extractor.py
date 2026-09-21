import json
import re
import numpy as np
import xgboost as xgb

# ---------------------------------------------------------------------------
# Paths — change MODEL_DIR if running elsewhere
# ---------------------------------------------------------------------------
MODEL_DIR = "/scratch/budayaku/SML/model"
clf = xgb.XGBClassifier()
clf.load_model(f"{MODEL_DIR}/xgb_disease_model.json")

with open(f"{MODEL_DIR}/symptoms.json") as f:
    SYMPTOM_NAMES = json.load(f)

with open(f"{MODEL_DIR}/diseases.json") as f:
    _raw = json.load(f)
    DISEASES = {int(k): v for k, v in _raw.items()}

SYMPTOM_INDEX = {name: i for i, name in enumerate(SYMPTOM_NAMES)}

# ---------------------------------------------------------------------------
# Synonym dictionary
# Maps canonical symptom name (must exist in symptoms.json) -> list of phrases
# Every key has been validated against SYMPTOM_NAMES.
# ---------------------------------------------------------------------------
SYNONYMS = {
    # --- Mental / emotional ---
    "anxiety and nervousness": ["anxiety", "anxious", "nervous", "nervousness", "panicky", "on edge"],
    "depression": ["depression", "depressed", "hopeless", "feeling down", "sad all the time"],
    "insomnia": ["insomnia", "cant sleep", "cannot sleep", "trouble sleeping", "sleepless"],

    # --- Chest / cardiac ---
    "sharp chest pain": [
        "sharp chest pain", "stabbing chest pain", "stabbing pain in chest",
        "chest stabbing", "chest pain", "chest hurts", "chest is hurting",
        "pain in chest", "my chest hurts", "chest is hurting sharply",
    ],
    "shortness of breath": [
        "shortness of breath", "short of breath", "cannot breathe",
        "hard to breathe", "breathless", "trouble breathing", "difficulty breathing",
    ],
    "palpitations": ["palpitations", "heart racing", "heart pounding", "fast heartbeat", "racing heart"],
    "irregular heartbeat": ["irregular heartbeat", "skipped beats", "heart skipping"],

    # --- Head / neuro ---
    "headache": ["headache", "head pain", "head hurts", "migraine", "pounding head", "head aching"],
    "dizziness": ["dizziness", "dizzy", "lightheaded", "light headed", "room spinning", "vertigo", "spinning"],
    "fainting": ["fainting", "fainted", "passed out", "blacking out"],
    "seizures": ["seizures", "seizure", "convulsions", "fit"],
    "disturbance of memory": ["memory loss", "forgetting things", "memory problems", "disturbance of memory"],
    "slurring words": ["slurred speech", "slurring words", "speech slurred"],

    # --- General / systemic ---
    "fever": ["fever", "feverish", "high temperature", "running a temperature", "temperature"],
    "chills": ["chills", "shivering", "shivers", "chilled"],
    "sweating": ["sweating", "sweaty", "perspiring", "sweats"],
    "fatigue": ["fatigue", "tired", "exhausted", "no energy", "worn out", "drained", "weary"],
    "feeling ill": ["feeling ill", "feel sick", "unwell", "not feeling well"],
    "feeling hot": ["feeling hot", "burning up", "hot"],
    "feeling cold": ["feeling cold", "always cold", "cold intolerance"],
    "feeling hot and cold": ["hot and cold", "alternating hot and cold"],
    "hot flashes": ["hot flashes", "hot flushes", "sudden heat"],
    "recent weight loss": ["weight loss", "losing weight", "lost weight", "dropping weight", "recent weight loss"],
    "weight gain": ["weight gain", "gaining weight", "gained weight"],
    "underweight": ["underweight", "too thin", "very thin"],

    # --- GI / abdominal ---
    "lower abdominal pain": [
        "stomach pain", "stomach ache", "tummy pain", "belly pain",
        "abdominal pain", "pain in abdomen", "pain in stomach",
        "lower abdominal pain", "lower belly pain",
    ],
    "upper abdominal pain": ["upper abdominal pain", "upper stomach pain", "pain in upper abdomen"],
    "sharp abdominal pain": ["sharp abdominal pain", "stabbing stomach pain", "sharp stomach pain"],
    "burning abdominal pain": ["burning abdominal pain", "burning stomach pain"],
    "nausea": ["nausea", "nauseous", "queasy", "sick to my stomach"],
    "vomiting": ["vomiting", "throwing up", "puking", "threw up", "vomit"],
    "diarrhea": ["diarrhea", "loose stool", "loose stools", "runny stool", "watery stool"],
    "constipation": ["constipation", "constipated", "hard stool"],
    "blood in stool": ["blood in stool", "bloody stool", "rectal bleeding", "blood when pooping"],
    "difficulty in swallowing": [
        "difficulty swallowing", "difficulty in swallowing", "hard to swallow",
        "trouble swallowing", "cannot swallow",
    ],
    "heartburn": ["heartburn", "acid reflux", "burning in chest", "burning after eating"],
    "stomach bloating": ["bloating", "bloated", "stomach bloated", "swollen belly", "stomach bloating"],
    "abdominal distention": ["abdominal distention", "distended stomach", "swollen abdomen"],
    "flatulence": ["gas", "gassy", "passing gas", "flatulence"],
    "decreased appetite": [
        "loss of appetite", "no appetite", "not hungry",
        "cannot eat", "decreased appetite",
    ],
    "excessive appetite": ["excessive appetite", "always hungry", "increased appetite"],
    "difficulty eating": ["difficulty eating", "trouble eating"],

    # --- Respiratory / ENT ---
    "cough": ["cough", "coughing", "hacking", "dry cough", "wet cough"],
    "sore throat": ["sore throat", "throat pain", "throat hurts", "scratchy throat"],
    "nasal congestion": [
        "stuffy nose", "blocked nose", "congested", "nasal congestion",
        "runny nose", "nose running", "nasal discharge",
    ],
    "nosebleed": ["nosebleed", "bloody nose", "nose bleeding"],
    "ear pain": ["ear pain", "earache", "ear hurts"],
    "diminished hearing": ["hearing loss", "trouble hearing", "diminished hearing"],
    "pus draining from ear": ["pus from ear", "ear discharge", "pus draining from ear"],

    # --- Eyes / vision ---
    "diminished vision": ["blurred vision", "blurry vision", "vision blurry", "diminished vision"],
    "double vision": ["double vision", "seeing double"],
    "spots or clouds in vision": ["spots in vision", "floaters", "cloudy vision", "spots or clouds in vision"],
    "eye redness": ["red eyes", "bloodshot eyes", "eye redness"],

    # --- Skin ---
    "skin rash": ["rash", "skin rash", "red patches", "skin irritation"],
    "itching of skin": ["itchy", "itchiness", "itching", "skin itching", "scratchy skin", "itching of skin"],
    "abnormal appearing skin": ["abnormal skin", "strange skin", "abnormal appearing skin"],
    "too little hair": ["hair loss", "losing hair", "hair falling out", "balding", "too little hair"],
    "unwanted hair": ["unwanted hair", "excess hair"],
    "irregular appearing nails": ["brittle nails", "nails breaking", "abnormal nails", "irregular appearing nails"],
    "mouth dryness": ["dry mouth", "mouth is dry", "cotton mouth", "mouth dryness"],

    # --- Urinary ---
    "painful urination": [
        "painful urination", "burning when peeing", "pain when urinating",
        "burns to pee", "burning when i pee", "burning when i urinate",
        "it burns when i pee", "hurts to pee", "hurts when i pee",
        "burning pee", "stinging when i pee",
    ],
    "frequent urination": ["frequent urination", "peeing a lot", "constantly peeing", "pee often"],
    "blood in urine": ["blood in urine", "bloody urine", "bleeding when peeing"],

    # --- Musculoskeletal ---
    "back pain": ["back pain", "back hurts", "aching back", "sore back"],
    "neck pain": ["neck pain", "neck hurts", "stiff neck", "sore neck"],
    "joint pain": ["joint pain", "joints hurt", "aching joints", "sore joints"],
    "muscle pain": ["muscle pain", "muscle ache", "muscles hurt", "sore muscles", "body ache", "body aches"],
    "leg pain": ["leg pain", "legs hurt", "aching legs", "sore legs"],
    "leg swelling": ["swollen legs", "leg swelling", "puffy legs"],
    "leg cramps or spasms": ["leg cramps", "leg spasms", "leg cramps or spasms"],
    "knee pain": ["knee pain", "knee hurts", "sore knee"],
    "ankle pain": ["ankle pain", "ankle hurts", "sore ankle"],
    "ankle swelling": ["swollen ankles", "ankle swelling", "puffy ankles"],
    "foot or toe pain": ["foot pain", "feet hurt", "sore feet", "toe pain", "foot or toe pain"],
    "foot or toe swelling": ["swollen feet", "foot swelling", "foot or toe swelling"],
    "arm pain": ["arm pain", "arm hurts", "sore arm"],
    "elbow pain": ["elbow pain", "elbow hurts", "sore elbow"],
    "elbow swelling": ["elbow swelling", "swollen elbow", "elbow is swollen", "my elbow is swollen"],
    "wrist pain": ["wrist pain", "wrist hurts", "sore wrist"],
    "hand or finger pain": ["hand pain", "finger pain", "hand hurts", "fingers hurt", "hand or finger pain"],
    "shoulder pain": ["shoulder pain", "shoulder hurts", "sore shoulder"],
    "hip pain": ["hip pain", "hip hurts", "sore hip"],
    "weakness": ["weakness", "weak", "feel weak"],

    # --- Lymph / swelling ---
    "swollen lymph nodes": ["swollen lymph nodes", "swollen glands", "lumps in neck"],
    "neck swelling": ["neck swelling", "swollen neck"],
    "facial pain": ["face pain", "facial pain"],

    # --- Reproductive ---
    "vaginal discharge": ["vaginal discharge", "discharge"],
    "impotence": ["erectile dysfunction", "ed", "impotence"],
    "pelvic pain": ["pelvic pain", "pelvis hurts", "pain in pelvis"],
}

# Filter out any keys that aren't in symptoms.json (safety net)
SYNONYMS = {k: v for k, v in SYNONYMS.items() if k in SYMPTOM_INDEX}

# ---------------------------------------------------------------------------
# Normalization: expand contractions, lowercase, strip punctuation
# ---------------------------------------------------------------------------
CONTRACTIONS = {
    "it's": "it is", "he's": "he is", "she's": "she is", "i'm": "i am",
    "i've": "i have", "can't": "cannot", "won't": "will not", "don't": "do not",
    "doesn't": "does not", "didn't": "did not", "isn't": "is not", "aren't": "are not",
    "wasn't": "was not", "weren't": "were not", "haven't": "have not", "hasn't": "has not",
    "hadn't": "had not", "wouldn't": "would not", "couldn't": "could not", "shouldn't": "should not",
    "i'll": "i will", "you're": "you are", "we're": "we are", "they're": "they are",
}

def expand_contractions(text: str) -> str:
    for c, e in CONTRACTIONS.items():
        text = text.replace(c, e)
    return text

def normalize(text: str) -> str:
    text = text.lower()
    text = expand_contractions(text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Pre-build phrase lookup sorted by length desc so longest phrases win
_PHRASE_LOOKUP = []
for canonical, phrases in SYNONYMS.items():
    for phrase in phrases:
        _PHRASE_LOOKUP.append((normalize(phrase), canonical))
_PHRASE_LOOKUP.sort(key=lambda x: -len(x[0]))

# ---------------------------------------------------------------------------
# Contextual linker: handle detached descriptors like "elbow hurts and it's swollen"
# ---------------------------------------------------------------------------
BODY_PARTS = [
    "elbow", "knee", "ankle", "wrist", "shoulder", "hip", "neck",
    "back", "leg", "arm", "hand", "foot", "finger", "toe",
]

_DESCRIPTOR_TEMPLATES = {
    "swollen": "{part} swelling",
    "swelling": "{part} swelling",
    "puffy": "{part} swelling",
    "stiff": "{part} stiffness or tightness",
    "tight": "{part} stiffness or tightness",
    "weak": "{part} weakness",
    "cramping": "{part} cramps or spasms",
    "cramps": "{part} cramps or spasms",
    "lump": "{part} lump or mass",
    "mass": "{part} lump or mass",
}

CONTEXTUAL_RULES = {}
for part in BODY_PARTS:
    for descriptor, template in _DESCRIPTOR_TEMPLATES.items():
        canonical = template.format(part=part)
        # foot/toe and hand/finger share combined column names
        if part in ("foot", "toe"):
            alt = template.format(part="foot or toe")
            if alt in SYMPTOM_INDEX:
                CONTEXTUAL_RULES[(part, descriptor)] = alt
                continue
        if part in ("hand", "finger"):
            alt = template.format(part="hand or finger")
            if alt in SYMPTOM_INDEX:
                CONTEXTUAL_RULES[(part, descriptor)] = alt
                continue
        if canonical in SYMPTOM_INDEX:
            CONTEXTUAL_RULES[(part, descriptor)] = canonical

def _apply_contextual_rules(normalized_text: str, already_found: list) -> list:
    already = set(already_found)
    new_finds = []
    sentences = re.split(r"[.!?;]", normalized_text)
    for sent in sentences:
        padded = f" {sent} "
        parts = [p for p in BODY_PARTS if f" {p} " in padded]
        for part in parts:
            for (rule_part, desc), canonical in CONTEXTUAL_RULES.items():
                if rule_part != part:
                    continue
                if f" {desc} " in padded and canonical not in already:
                    new_finds.append(canonical)
                    already.add(canonical)
    return new_finds

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_symptoms(user_text: str) -> list:
    """Turn free-form user text into a list of canonical symptom names."""
    text = " " + normalize(user_text) + " "
    found = []
    seen = set()
    for phrase, canonical in _PHRASE_LOOKUP:
        if f" {phrase} " in text and canonical not in seen:
            found.append(canonical)
            seen.add(canonical)
    found.extend(_apply_contextual_rules(text, found))
    return found

def symptoms_to_vector(symptom_list: list) -> np.ndarray:
    """Build a (1, 377) int8 vector from canonical symptom names."""
    vec = np.zeros(len(SYMPTOM_NAMES), dtype=np.int8)
    for s in symptom_list:
        if s in SYMPTOM_INDEX:
            vec[SYMPTOM_INDEX[s]] = 1
    return vec

def predict_from_text(user_text: str, k: int = 5) -> dict:
    """End-to-end: text -> top-k disease predictions."""
    symptoms = extract_symptoms(user_text)
    if not symptoms:
        return {"symptoms": [], "predictions": [], "message": "No recognized symptoms"}
    vec = symptoms_to_vector(symptoms).reshape(1, -1)
    proba = clf.predict_proba(vec)[0]
    top_idx = np.argsort(proba)[::-1][:k]
    preds = [(DISEASES[int(i)], float(proba[i])) for i in top_idx]
    return {"symptoms": symptoms, "predictions": preds}

def predict_from_symptom_list(symptoms: list, k: int = 5) -> list:
    """For the Day 4 chatbot: take an accumulated symptom list, return top-k."""
    if not symptoms:
        return []
    vec = symptoms_to_vector(symptoms).reshape(1, -1)
    proba = clf.predict_proba(vec)[0]
    top_idx = np.argsort(proba)[::-1][:k]
    return [(DISEASES[int(i)], float(proba[i])) for i in top_idx]

if __name__ == "__main__":
    # Quick smoke test
    samples = [
        "I have a headache and feel dizzy",
        "My elbow hurts and it's swollen",
        "burning when I pee and peeing a lot",
    ]
    for s in samples:
        r = predict_from_text(s, k=3)
        print(f"\n> {s}")
        print(f"  symptoms: {r['symptoms']}")
        for name, p in r["predictions"]:
            print(f"  {name}: {p:.3f}")


