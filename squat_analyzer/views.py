"""
Views para análise de agachamento - Camada de Controller
Responsável por receber requisições HTTP e orquestrar o fluxo.
"""
import os
from django.shortcuts import render, redirect
from django.core.files.uploadedfile import UploadedFile
from django.http import FileResponse, Http404
from .services.analysis_service import SquatAnalysisService
from django.contrib import messages
from django.core.exceptions import ValidationError
from .validators.file_validators import MP4VideoValidator



# Thresholds padrão por tipo de análise e lado
FRONTAL_LEFT_THRESHOLDS = {
    'descent_th': 0.05,
    'hip_err_th': 1,
    'ascent_return_th': 0.02,
    'knee_valgus_th': 5,
    'foot_pronation_th': 7,
}

FRONTAL_RIGHT_THRESHOLDS = {
    'descent_th': 0.05,
    'hip_err_th': 1,
    'ascent_return_th': 0.02,
    'knee_valgus_th': 12,
    'foot_pronation_th': 7,
}

SAGITTAL_LEFT_THRESHOLDS = {
    'descent_th': 0.05,
    'trunk_err_th': 23,
    'head_err_th': 2,
    'ascent_return_th': 0.02,
    'knee_err_th': 6,
    'foot_err_th': 8,
}

SAGITTAL_RIGHT_THRESHOLDS = {
    'descent_th': 0.05,
    'trunk_err_th': 23,
    'head_err_th': 2,
    'ascent_return_th': 0.02,
    'knee_err_th': 6,
    'foot_err_th': 8,
}


def index(request):
    """Pagina inicial com selecao do tipo de analise."""
    return render(request, 'squat_analyzer/index.html')


def _parse_selected_reps(post_data):
    """Extrai as repeticoes marcadas (1, 2, 3) do POST do formulario frontal."""
    return [rep for rep in (1, 2, 3) if post_data.get(f'rep_{rep}')]


def _process_frontal_analysis(request, template_name, side, thresholds):
    """
    Fluxo unificado de analise frontal (Otimizacao Fase 1.2 - eliminacao de
    codigo duplicado entre as 4 views). A view permanece como Controller
    puro: valida entrada e delega o processamento ao SquatAnalysisService,
    conforme docs/RULES.md secao 2.1.
    """
    context = {
        'side': side,
        'analysis_type': 'frontal',
        'title': f'Análise Frontal {"Esquerdo" if side == "esquerdo" else "Direito"}',
        'thresholds': thresholds,
    }

    if request.method == 'POST':
        video_file = request.FILES.get('video')
        person_name = request.POST.get('person_name')

        if video_file:
            try:
                validator = MP4VideoValidator()
                validator(video_file)
            except ValidationError as e:
                messages.error(request, str(e))
                return render(request, template_name, context)

        # Parametros de analise
        params = {
            'descent_threshold': float(request.POST.get('descent_threshold', thresholds['descent_th'])),
            'ascent_return_threshold': float(request.POST.get('ascent_return_threshold', thresholds['ascent_return_th'])),
            'hip_error_threshold': int(request.POST.get('hip_err_th', thresholds['hip_err_th'])),
            'knee_valgus_error_threshold': int(request.POST.get('knee_valgus_th', thresholds['knee_valgus_th'])),
            'foot_pronation_error_threshold': int(request.POST.get('foot_pronation_th', thresholds['foot_pronation_th'])),
        }

        selected_reps = _parse_selected_reps(request.POST)

        if video_file and person_name:
            service = SquatAnalysisService()
            result = service.analyze_frontal(video_file, person_name, side, params, selected_reps)
            context['result'] = result
            context['person_name'] = person_name

    return render(request, template_name, context)


def _process_sagittal_analysis(request, template_name, side, thresholds):
    """
    Fluxo unificado de analise sagital (Otimizacao Fase 1.2 - eliminacao de
    codigo duplicado). View como Controller: valida e delega ao servico.
    """
    context = {
        'side': side,
        'analysis_type': 'sagittal',
        'title': f'Análise Sagital {"Esquerdo" if side == "esquerdo" else "Direito"}',
        'thresholds': thresholds,
    }

    if request.method == 'POST':
        video_file = request.FILES.get('video')
        person_name = request.POST.get('person_name')
        user_height_cm = float(request.POST.get('user_height_cm', 170))

        if video_file:
            try:
                validator = MP4VideoValidator()
                validator(video_file)
            except ValidationError as e:
                messages.error(request, str(e))
                return render(request, template_name, context)

        # Parametros de analise
        params = {
            'descent_threshold': float(request.POST.get('descent_threshold', thresholds['descent_th'])),
            'ascent_return_threshold': float(request.POST.get('ascent_return_threshold', thresholds['ascent_return_th'])),
            'trunk_error_threshold': int(request.POST.get('trunk_err_th', thresholds['trunk_err_th'])),
            'knee_error_threshold': int(request.POST.get('knee_err_th', thresholds['knee_err_th'])),
            'head_error_threshold': int(request.POST.get('head_err_th', thresholds['head_err_th'])),
            'foot_error_threshold': int(request.POST.get('foot_err_th', thresholds['foot_err_th'])),
        }

        if video_file and person_name:
            service = SquatAnalysisService()
            result = service.analyze_sagittal(video_file, person_name, side, user_height_cm, params)
            context['result'] = result
            context['person_name'] = person_name

    return render(request, template_name, context)


def frontal_left_analysis(request):
    """View para analise frontal - lado esquerdo (Controller fino)."""
    return _process_frontal_analysis(
        request, 'squat_analyzer/frontal_left_analysis.html', 'esquerdo', FRONTAL_LEFT_THRESHOLDS)


def frontal_right_analysis(request):
    """View para analise frontal - lado direito (Controller fino)."""
    return _process_frontal_analysis(
        request, 'squat_analyzer/frontal_right_analysis.html', 'direito', FRONTAL_RIGHT_THRESHOLDS)


def sagittal_left_analysis(request):
    """View para analise sagital - lado esquerdo (Controller fino)."""
    return _process_sagittal_analysis(
        request, 'squat_analyzer/sagittal_left_analysis.html', 'esquerdo', SAGITTAL_LEFT_THRESHOLDS)


def sagittal_right_analysis(request):
    """View para analise sagital - lado direito (Controller fino)."""
    return _process_sagittal_analysis(
        request, 'squat_analyzer/sagittal_right_analysis.html', 'direito', SAGITTAL_RIGHT_THRESHOLDS)


def download_excel(request, analysis_type, side):
    """
    View para download do arquivo Excel gerado pela análise.
    
    Args:
        request: Requisição HTTP
        analysis_type: 'frontal' ou 'sagittal'
        side: 'direito' ou 'esquerdo'
    
    Returns:
        FileResponse com o arquivo Excel ou Http404 se não encontrado
    """
    if analysis_type not in ['frontal', 'sagittal']:
        raise Http404("Tipo de análise inválido.")
    
    if side not in ['direito', 'esquerdo']:
        raise Http404("Lado inválido.")
    
    # Obtém o nome da pessoa via parâmetro GET
    person_name = request.GET.get('person_name')
    
    if not person_name:
        raise Http404("Nome da pessoa não fornecido.")
    
    # Usa o serviço para obter o caminho do arquivo
    file_path = SquatAnalysisService.get_excel_file_path(person_name, analysis_type, side)
    
    # Verifica se o arquivo existe
    if not os.path.exists(file_path):
        raise Http404(f"Arquivo de relatório não encontrado para {person_name} ({analysis_type} - {side}).")
    
    # Nome do arquivo para download
    filename = f"Relatorio_{person_name}_{analysis_type}_{side}.xlsx"
    
    # Retorna o arquivo como resposta de download
    response = FileResponse(
        open(file_path, 'rb'),
        as_attachment=True,
        filename=filename,
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    
    return response
