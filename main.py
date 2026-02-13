from groq import Groq
import os
from dotenv import load_dotenv
import instructor
from groq import Groq
from dotenv import load_dotenv
from typing import Optional, List
from enum import Enum
from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, Field
import uvicorn

load_dotenv()


# ==============================================================================
# 1. DOMAIN MODELS & DTOs (Data Transfer Objects)
# ==============================================================================
# Aceste clase definesc structura datelor. (S din SOLID)


class AuthorInfo(BaseModel):
    """Informații despre autorul recenziei"""
    name: Optional[str] = Field(None, description="Numele autorului, dacă este menționat")
    verified_buyer: bool = Field(False,
        description="true = cumpărător verificat, false = nu e specificat sau nu e verificat"
    )
    experience_level: Optional[str] = Field("nespecificat", description="Nivel de experiență al reviewărului: începător, mediu, avansat sau nespecificat.")


class RecommendationStatus(Enum):
    """Statusul recomandării."""
    recommended = "recommended"
    not_recommended = "not_recommended"
    neutral = "neutral"
    not_specified = "not_specified"


class ReviewAnalysis(BaseModel):
    """Analiza structurată a unei recenzii (Output Model)"""
    product_name: str = Field(description="Numele produsului recenzat")
    rating: int = Field(ge=1, le=5, description="Rating numeric: 1-5")
    author: AuthorInfo = Field(description="Informații despre autor")
    pros: List[str] = Field(default_factory=list, description="Lista avantajelor")
    cons: List[str] = Field(default_factory=list, description="Lista dezavantajelor")
    is_authentic: Optional[bool] = Field(
        None, description="true = pare autentică, false = pare AI"
    )
    summary: str = Field(description="Rezumat scurt în română")
    would_recommend: RecommendationStatus = Field(
        RecommendationStatus.not_specified, 
        description="Dacă autorul recomandă produsul"
    )
    confidence_score: float = Field(ge=0, le=100, description="Procent încredere asupra analizei în ansamblu (0-100) 0 = foarte nesigur, 100 = foarte sigur)")
    #reasoning: str = Field(None, description="Explicația deciziei luate de model")


class ReviewRequest(BaseModel):
    """Model pentru input-ul primit prin POST (Input DTO)"""
    review_text: str = Field(
        ..., 
        min_length=10, 
        description="Textul recenziei care trebuie analizat"
    )


# ==============================================================================
# 2. SERVICE LAYER (Business Logic)
# ==============================================================================
# Această clasă se ocupă DOAR de logica de analiză. (S & O din SOLID)


class ReviewAnalyzerService():
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set properly.")
        
        # Inițializare client Instructor cu Groq
        self.client = instructor.from_groq(
                                            Groq(api_key=api_key),
                                            mode=instructor.Mode.TOOLS
                                        )
        # Modelul specificat în exemplul tău
        self.model_name = "openai/gpt-oss-120b" 

    def analyze(self, text: str) -> ReviewAnalysis:
        """
        Trimite textul către LLM și returnează obiectul structurat.
        """
        try:
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ești un analizator de recenzii. Extragi date structurat. "
                            "Pentru Recomandare folosește doar 'recommended', 'not_recommended', "
                            "'neutral' sau 'not_specified'. Pentru verified_buyer folosește true/false. "
                            "Returnează doar JSON valid conform modelului ReviewAnalysis. Nu adăuga text explicativ suplimentar."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Analizează această recenzie:\n\n{text}"
                    }
                ],
                temperature=0.1, # Pentru rezultate mai determinate
                max_tokens=1024,
                response_model=ReviewAnalysis
            )
        except Exception as e:
            # Aici am putea loga eroarea intern
            print(f"Internal LLM Error: {e}")
            raise e


# ==============================================================================
# 3. DEPENDENCIES (Dependency Injection)
# ==============================================================================
# (Dependency Inversion din SOLID) - Decuplăm crearea serviciului de rute.

def get_analyzer_service() -> ReviewAnalyzerService:
    api_key = os.getenv("GROQ_TOKEN")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Server configuration error: Missing API Key"
        )
    return ReviewAnalyzerService(api_key)

# ==============================================================================
# 4. API LAYER (FastAPI App & Routes)
# ==============================================================================


app = FastAPI(
    title="AI Review Analyzer API",
    description="API pentru extragerea datelor structurate din recenzii folosind LLM.",
    version="1.0.0"
)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Endpoint de verificare a stării serverului.
    """
    return {"status": "ok", "service": "Review Analyzer API"}


@app.post(
    "/api/v1/analyze", 
    response_model=ReviewAnalysis, 
    status_code=status.HTTP_200_OK,
    summary="Analizează o recenzie",
    description="Primește un text de recenzie și returnează o structură JSON detaliată cu sentiment, pros/cons și detalii autor."
)
async def analyze_review(
    request: ReviewRequest, 
    service: ReviewAnalyzerService = Depends(get_analyzer_service)
):
    """
    Endpoint principal pentru analiză. 
    Folosește Dependency Injection pentru a accesa serviciul de analiză.
    """
    try:
        result = service.analyze(request.review_text)
        return result
    except Exception as e:
        # Transformăm erorile interne în răspunsuri HTTP corespunzătoare
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Eroare la procesarea AI: {str(e)}"
        )


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Rulează serverul pe localhost:8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)