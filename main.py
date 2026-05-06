from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with your GitHub Pages URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model...")
classifier = pipeline("text-classification", model="Minej/bert-base-personality")
print("Label format:", classifier.model.config.id2label)

class QuizAnswers(BaseModel):
    q1: str
    q2: str
    q3: str
    q4: str
    q5: str
    q6: str
    q7: str
    q8: str

def build_text(answers: QuizAnswers) -> str:
    style_map = {
        "Visual": "seeing images and diagrams",
        "Auditory": "listening to explanations",
        "Story": "hearing stories and real life examples",
    }
    return f"""
    I understand things best by {style_map.get(answers.q1, answers.q1)}.
    I remember topics through {style_map.get(answers.q2, answers.q2)}.
    I learn faster with {style_map.get(answers.q3, answers.q3)}.
    I prefer notes that {style_map.get(answers.q4, answers.q4)}.
    I am {'extraverted and energized by people' if answers.q5 == 'E' else 'introverted and energized by reflection'}.
    I focus on {'real facts and details' if answers.q6 == 'S' else 'big ideas and possibilities'}.
    I make decisions with {'logic' if answers.q7 == 'T' else 'feelings and values'}.
    I like to {'plan and stay organized' if answers.q8 == 'J' else 'stay flexible and spontaneous'}.
    """

def get_learning_style(answers: QuizAnswers):
    visual = auditory = story = 0
    for val in [answers.q1, answers.q2, answers.q3, answers.q4]:
        if val == "Visual": visual += 1
        elif val == "Auditory": auditory += 1
        else: story += 1
    if visual >= auditory and visual >= story:
        return "Visual", visual, auditory, story
    elif auditory >= visual and auditory >= story:
        return "Auditory", visual, auditory, story
    return "Story-based", visual, auditory, story

def parse_mbti_label(raw_label: str) -> str:
    """Handles both 'INTJ' and 'label_0' style outputs."""
    mbti_types = [
        "INTJ","INTP","INFJ","INFP","ISTJ","ISTP","ISFJ","ISFP",
        "ENTJ","ENTP","ENFJ","ENFP","ESTJ","ESTP","ESFJ","ESFP"
    ]
    upper = raw_label.upper()
    if upper in mbti_types:
        return upper
    # If model returns label_0, label_1 etc., map via id2label
    id2label = classifier.model.config.id2label
    if raw_label in id2label:
        return id2label[raw_label].upper()
    return "INTJ"  # fallback

@app.post("/predict")
def predict(answers: QuizAnswers):
    text = build_text(answers)
    result = classifier(text)
    raw_label = result[0]["label"]
    mbti = parse_mbti_label(raw_label)
    learning_style, visual, auditory, story = get_learning_style(answers)
    return {
        "mbti": mbti,
        "learningStyle": learning_style,
        "visual": visual,
        "auditory": auditory,
        "story": story,
    }
