from pydantic import BaseModel
from typing import List, Optional


class LoanApplication(BaseModel):
    gender: str
    married: str
    dependents: Optional[str] = None
    education: str
    self_employed: Optional[str] = None
    applicant_income: float
    coapplicant_income: float
    loan_amount: float
    loan_amount_term: float
    credit_history: float
    property_area: str


class LoanBatch(BaseModel):
    applications: List[LoanApplication]


class LoanPredictionResponse(BaseModel):
    prediction: List[int]
    prediction_name: List[str]
    proba: List[List[float]]
