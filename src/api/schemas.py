from pydantic import BaseModel, Field

class QARequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=512,
                          example="ગુજરાતની રાજધાની કઈ છે?")
    context: str = Field(..., min_length=20, max_length=2000,
                         example="ગુજરાત ભારતનું એક રાજ્ય છે. ગાંધીનગર ગુજરાતની રાજધાની છે.")

class QAResponse(BaseModel):
    answer: str
    confidence: float
    question: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool