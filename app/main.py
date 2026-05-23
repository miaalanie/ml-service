from fastapi import FastAPI

from app.schemas import MatchRequest
from app.matcher import MatcherService


app = FastAPI()

matcher_service = MatcherService()


@app.get('/')
def home():
    return {
        'message': 'ML Matching Service Running'
    }


@app.post('/match')
def match(payload: MatchRequest):

    results = matcher_service.match(payload)

    return {
        'success': True,
        'recommendations': results
    }