"""
Models for the Core application.

Estes modelos representam as entidades persistidas no banco de dados Django,
separadas da lógica de negócio que reside nos services e domain objects.
"""

from django.db import models
from django.utils import timezone
from typing import Optional


class AnalysisSession(models.Model):
    """
    Modelo para armazenar sessões de análise de agachamento.
    
    Este modelo armazena metadados sobre cada sessão de análise,
    enquanto os dados detalhados da análise são mantidos em arquivos
    ou em outros modelos especializados.
    
    Atributos:
        person_name: Nome da pessoa analisada
        analysis_type: Tipo de análise (sagittal_right, sagittal_left, frontal_right, frontal_left)
        video_file: Arquivo de vídeo enviado
        user_height_cm: Altura do usuário em centímetros
        parameters: Parâmetros de análise utilizados (JSON)
        created_at: Data de criação da sessão
        status: Status da análise (pending, processing, completed, failed)
        result_summary: Resumo dos resultados (JSON)
    """
    
    ANALYSIS_TYPE_CHOICES = [
        ('sagittal_right', 'Sagital Direito'),
        ('sagittal_left', 'Sagital Esquerdo'),
        ('frontal_right', 'Frontal Direito'),
        ('frontal_left', 'Frontal Esquerdo'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'Pendente'),
        ('processing', 'Processando'),
        ('completed', 'Completo'),
        ('failed', 'Falhou'),
    ]
    
    person_name = models.CharField(max_length=100, verbose_name="Nome da Pessoa")
    analysis_type = models.CharField(
        max_length=20,
        choices=ANALYSIS_TYPE_CHOICES,
        verbose_name="Tipo de Análise"
    )
    video_file = models.FileField(
        upload_to='videos/%Y/%m/%d/',
        verbose_name="Arquivo de Vídeo"
    )
    user_height_cm = models.IntegerField(
        null=True,
        blank=True,
        verbose_name="Altura (cm)"
    )
    parameters = models.JSONField(
        default=dict,
        blank=True,
        verbose_name="Parâmetros de Análise"
    )
    created_at = models.DateTimeField(
        default=timezone.now,
        verbose_name="Data de Criação"
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending',
        verbose_name="Status"
    )
    result_summary = models.JSONField(
        default=dict,
        blank=True,
        verbose_name="Resumo dos Resultados"
    )
    excel_report_path = models.CharField(
        max_length=500,
        blank=True,
        verbose_name="Caminho do Relatório Excel"
    )
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Sessão de Análise"
        verbose_name_plural = "Sessões de Análise"
    
    def __str__(self) -> str:
        return f"{self.person_name} - {self.get_analysis_type_display()} ({self.status})"
    
    def is_completed(self) -> bool:
        """Verifica se a análise foi completada com sucesso."""
        return self.status == 'completed'
    
    def get_analysis_plane(self) -> str:
        """Retorna o plano de análise (sagittal ou frontal)."""
        if 'sagittal' in self.analysis_type:
            return 'sagittal'
        return 'frontal'
    
    def get_analysis_side(self) -> str:
        """Retorna o lado da análise (right ou left)."""
        if 'right' in self.analysis_type:
            return 'direito'
        return 'esquerdo'


class AnalysisRepetition(models.Model):
    """
    Modelo para armazenar detalhes de cada repetição detectada.
    
    Relaciona-se com AnalysisSession para armazenar os dados
    detalhados de cada repetição do agachamento.
    """
    
    session = models.ForeignKey(
        AnalysisSession,
        on_delete=models.CASCADE,
        related_name='repetitions',
        verbose_name="Sessão"
    )
    repetition_number = models.IntegerField(verbose_name="Número da Repetição")
    timestamp_seconds = models.FloatField(
        null=True,
        blank=True,
        verbose_name="Timestamp (segundos)"
    )
    trunk_error = models.BooleanField(default=False, verbose_name="Erro de Tronco")
    knee_error = models.BooleanField(default=False, verbose_name="Erro de Joelho")
    head_error = models.BooleanField(default=False, verbose_name="Erro de Cabeça")
    heel_error = models.BooleanField(default=False, verbose_name="Erro de Calcanhar")
    hip_error = models.BooleanField(default=False, verbose_name="Erro de Quadril")
    foot_pronation_error = models.BooleanField(default=False, verbose_name="Erro de Pronação")
    trunk_error_count = models.IntegerField(default=0, verbose_name="Contagem Erros Tronco")
    knee_error_count = models.IntegerField(default=0, verbose_name="Contagem Erros Joelho")
    head_error_count = models.IntegerField(default=0, verbose_name="Contagem Erros Cabeça")
    heel_error_count = models.IntegerField(default=0, verbose_name="Contagem Erros Calcanhar")
    hip_error_count = models.IntegerField(default=0, verbose_name="Contagem Erros Quadril")
    foot_pronation_error_count = models.IntegerField(default=0, verbose_name="Contagem Erros Pronação")
    
    class Meta:
        ordering = ['repetition_number']
        unique_together = ['session', 'repetition_number']
        verbose_name = "Repetição"
        verbose_name_plural = "Repetições"
    
    def __str__(self) -> str:
        return f"Sessão {self.session.id} - Repetição {self.repetition_number}"
    
    def has_errors(self) -> bool:
        """Verifica se há algum erro nesta repetição."""
        return any([
            self.trunk_error, self.knee_error, self.head_error,
            self.heel_error, self.hip_error, self.foot_pronation_error
        ])
