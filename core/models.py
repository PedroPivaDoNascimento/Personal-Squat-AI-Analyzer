"""
Models do Django para a aplicação de Análise de Agachamento.

Nota: Esta aplicação não utiliza banco de dados. Os models abaixo são apenas
estruturas de dados para representar as entidades do domínio e manter a 
organização do padrão MVC do Django.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class AnalysisParams:
    """Representa os parâmetros de configuração para análise de vídeo."""
    descent_threshold: float = 0.05
    ascent_return_threshold: float = 0.02
    
    # Parâmetros específicos do plano sagital
    trunk_error_threshold: Optional[int] = None
    knee_error_threshold: Optional[int] = None
    head_error_threshold: Optional[int] = None
    foot_error_threshold: Optional[int] = None
    
    # Parâmetros específicos do plano frontal
    hip_error_threshold: Optional[int] = None
    knee_valgus_error_threshold: Optional[int] = None
    foot_pronation_error_threshold: Optional[int] = None


@dataclass
class AnalysisSession:
    """
    Representa uma sessão de análise de agachamento.
    
    Esta classe armazena todos os dados relacionados a uma análise específica,
    incluindo informações do usuário, parâmetros e resultados.
    """
    session_id: str
    analysis_type: str  # 'sagittal_right', 'sagittal_left', 'frontal_right', 'frontal_left'
    person_name: str
    user_height_cm: Optional[int] = None
    video_path: Optional[str] = None
    params: Optional[AnalysisParams] = None
    
    # Resultados da análise
    repetitions_detected: int = 0
    repetition_timestamps: List[Optional[float]] = field(default_factory=list)
    
    # Históricos de erro (contagem de instantes)
    error_history: Dict[str, List[Optional[int]]] = field(default_factory=dict)
    
    # Status por repetição (0=OK, 1=DESVIO)
    reps_status: Dict[str, List[Optional[int]]] = field(default_factory=dict)
    
    # DataFrames de desvios ponto a ponto
    dataframes: Dict[str, Any] = field(default_factory=dict)
    
    # Mensagens de feedback
    feedback_messages: List[Dict[str, str]] = field(default_factory=list)
    
    # Checkboxes para plano frontal (quais repetições salvar)
    selected_repetitions: List[int] = field(default_factory=list)


@dataclass
class RepetitionResult:
    """Representa o resultado de uma única repetição analisada."""
    repetition_number: int
    timestamp: Optional[float]
    
    # Status dos desvios (0=OK, 1=DESVIO)
    trunk_status: Optional[int] = None
    knee_status: Optional[int] = None
    head_status: Optional[int] = None
    heel_status: Optional[int] = None
    hip_status: Optional[int] = None
    knee_valgus_status: Optional[int] = None
    foot_pronation_status: Optional[int] = None
    
    # Contagem de instantes de desvio
    trunk_error_count: Optional[int] = None
    knee_error_count: Optional[int] = None
    head_error_count: Optional[int] = None
    heel_error_count: Optional[int] = None
    hip_error_count: Optional[int] = None
    knee_valgus_error_count: Optional[int] = None
    foot_pronation_error_count: Optional[int] = None
    
    # Feedback gerado
    feedback: List[str] = field(default_factory=list)

