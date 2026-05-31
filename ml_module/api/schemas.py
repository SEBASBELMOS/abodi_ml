from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class ProcesoInput(BaseModel):
    tipo_proceso: str = Field(..., description="Tipo de proceso judicial")
    tipo_ultima_actuacion: str = Field(..., description="Tipo de la última actuación registrada")
    ciudad: str = Field(..., description="Ciudad del juzgado")
    despacho: str = Field(..., description="Nombre del despacho/juzgado")
    dias_sin_actividad: int = Field(..., ge=0, description="Días desde la última actuación")
    num_partes: int = Field(..., ge=1, le=50, description="Número de partes en el proceso")
    total_actuaciones: int = Field(..., ge=0, description="Total de actuaciones registradas")
    frecuencia_actualizaciones: float = Field(..., ge=0, description="Actuaciones por mes")
    tiene_termino_legal: int = Field(..., ge=0, le=1, description="1 si la última actuación tiene término legal")
    plan_suscripcion: str = Field(..., description="Plan de suscripción del usuario")

class BatchInput(BaseModel):
    procesos: List[ProcesoInput]

class RiskFactor(BaseModel):
    feature: str
    impact: float
    direction: str

class RiskOutput(BaseModel):
    riesgo: int
    probabilidad: float
    nivel: str
    factores_riesgo: List[RiskFactor]

class BatchOutput(BaseModel):
    resultados: List[RiskOutput]
    total: int
    en_riesgo: int

class HealthOutput(BaseModel):
    status: str
    model_version: Optional[str] = None
    model_f1: Optional[float] = None
    total_predictions: int

class ModelInfo(BaseModel):
    model_name: str
    version: Optional[str] = None
    metrics: Dict[str, float]
    features: List[str]
    fecha_entrenamiento: Optional[str] = None
