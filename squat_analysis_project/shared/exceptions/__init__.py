"""
Exceptions module - Contains custom exceptions for the application.
"""


class AnalysisException(Exception):
    """Exceção base para erros de análise."""
    pass


class VideoProcessingError(AnalysisException):
    """Erro ao processar vídeo."""
    pass


class PoseDetectionError(AnalysisException):
    """Erro na detecção de pose."""
    pass


class ReportGenerationError(AnalysisException):
    """Erro ao gerar relatório."""
    pass


class InvalidParameterError(AnalysisException):
    """Erro de parâmetro inválido."""
    pass
