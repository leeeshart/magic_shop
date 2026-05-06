from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuizAnswers(BaseModel):
    q1: str
    q2: str
    q3: str
    q4: str
    q5: str
    q6: str
    q7: str
    q8: str


@app.post("/predict")
def predict(answers: QuizAnswers):
    visual = 0
    auditory = 0
    story = 0

    for answer in [answers.q1, answers.q2, answers.q3, answers.q4]:
        if answer == "Visual":
            visual += 1
        elif answer == "Auditory":
            auditory += 1
        else:
            story += 1

    mbti = answers.q5 + answers.q6 + answers.q7 + answers.q8

    if visual >= auditory and visual >= story:
        learning_style = "Visual"
    elif auditory >= visual and auditory >= story:
        learning_style = "Auditory"
    else:
        learning_style = "Story-based"

    return {
        "learningStyle": learning_style,
        "mbti": mbti,
        "visual": visual,
        "auditory": auditory,
        "story": story,
    }
