"""
Views para análise de agachamento - Camada de Controller
Responsável por receber requisições HTTP e orquestrar o fluxo.
"""
import os
from django.shortcuts import render, redirect
from django.core.files.uploadedfile import UploadedFile
from django.http import FileResponse, Http404
from .services.analysis_service import SquatAnalysisService


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
    """Página inicial com seleção do tipo de análise."""
    return render(request, 'squat_analyzer/index.html')


def frontal_left_analysis(request):
    """
    View para análise frontal - lado esquerdo.
    GET: Exibe formulário de upload
    POST: Processa vídeo e exibe resultados
    """
    context = {
        'side': 'esquerdo',
        'analysis_type': 'frontal',
        'title': 'Análise Frontal Esquerdo',
        'thresholds': FRONTAL_LEFT_THRESHOLDS,
    }
    
    if request.method == 'POST':
        video_file = request.FILES.get('video')
        person_name = request.POST.get('person_name')
        
        # Parâmetros de análise
        params = {
            'descent_threshold': float(request.POST.get('descent_threshold', FRONTAL_LEFT_THRESHOLDS['descent_th'])),
            'ascent_return_threshold': float(request.POST.get('ascent_return_threshold', FRONTAL_LEFT_THRESHOLDS['ascent_return_th'])),
            'hip_error_threshold': int(request.POST.get('hip_err_th', FRONTAL_LEFT_THRESHOLDS['hip_err_th'])),
            'knee_valgus_error_threshold': int(request.POST.get('knee_valgus_th', FRONTAL_LEFT_THRESHOLDS['knee_valgus_th'])),
            'foot_pronation_error_threshold': int(request.POST.get('foot_pronation_th', FRONTAL_LEFT_THRESHOLDS['foot_pronation_th']))
        }
        
        # Repetições selecionadas
        selected_reps = []
        if request.POST.get('rep_1'):
            selected_reps.append(1)
        if request.POST.get('rep_2'):
            selected_reps.append(2)
        if request.POST.get('rep_3'):
            selected_reps.append(3)
        
        if video_file and person_name:
            service = SquatAnalysisService()
            result = service.analyze_frontal(video_file, person_name, 'esquerdo', params, selected_reps)
            context['result'] = result
            context['person_name'] = person_name
    
    return render(request, 'squat_analyzer/frontal_left_analysis.html', context)


def frontal_right_analysis(request):
    """
    View para análise frontal - lado direito.
    GET: Exibe formulário de upload
    POST: Processa vídeo e exibe resultados
    """
    context = {
        'side': 'direito',
        'analysis_type': 'frontal',
        'title': 'Análise Frontal Direito',
        'thresholds': FRONTAL_RIGHT_THRESHOLDS,
    }
    
    if request.method == 'POST':
        video_file = request.FILES.get('video')
        person_name = request.POST.get('person_name')
        
        # Parâmetros de análise
        params = {
            'descent_threshold': float(request.POST.get('descent_threshold', FRONTAL_RIGHT_THRESHOLDS['descent_th'])),
            'ascent_return_threshold': float(request.POST.get('ascent_return_threshold', FRONTAL_RIGHT_THRESHOLDS['ascent_return_th'])),
            'hip_error_threshold': int(request.POST.get('hip_err_th', FRONTAL_RIGHT_THRESHOLDS['hip_err_th'])),
            'knee_valgus_error_threshold': int(request.POST.get('knee_valgus_th', FRONTAL_RIGHT_THRESHOLDS['knee_valgus_th'])),
            'foot_pronation_error_threshold': int(request.POST.get('foot_pronation_th', FRONTAL_RIGHT_THRESHOLDS['foot_pronation_th']))
        }
        
        # Repetições selecionadas
        selected_reps = []
        if request.POST.get('rep_1'):
            selected_reps.append(1)
        if request.POST.get('rep_2'):
            selected_reps.append(2)
        if request.POST.get('rep_3'):
            selected_reps.append(3)
        
        if video_file and person_name:
            service = SquatAnalysisService()
            result = service.analyze_frontal(video_file, person_name, 'direito', params, selected_reps)
            context['result'] = result
            context['person_name'] = person_name
    
    return render(request, 'squat_analyzer/frontal_right_analysis.html', context)


def sagittal_left_analysis(request):
    """
    View para análise sagital - lado esquerdo.
    GET: Exibe formulário de upload
    POST: Processa vídeo e exibe resultados
    """
    context = {
        'side': 'esquerdo',
        'analysis_type': 'sagittal',
        'title': 'Análise Sagital Esquerdo',
        'thresholds': SAGITTAL_LEFT_THRESHOLDS,
    }
    
    if request.method == 'POST':
        video_file = request.FILES.get('video')
        person_name = request.POST.get('person_name')
        user_height_cm = float(request.POST.get('user_height_cm', 170))
        
        # Parâmetros de análise
        params = {
            'descent_threshold': float(request.POST.get('descent_threshold', SAGITTAL_LEFT_THRESHOLDS['descent_th'])),
            'ascent_return_threshold': float(request.POST.get('ascent_return_threshold', SAGITTAL_LEFT_THRESHOLDS['ascent_return_th'])),
            'trunk_error_threshold': int(request.POST.get('trunk_err_th', SAGITTAL_LEFT_THRESHOLDS['trunk_err_th'])),
            'knee_error_threshold': int(request.POST.get('knee_err_th', SAGITTAL_LEFT_THRESHOLDS['knee_err_th'])),
            'head_error_threshold': int(request.POST.get('head_err_th', SAGITTAL_LEFT_THRESHOLDS['head_err_th'])),
            'foot_error_threshold': int(request.POST.get('foot_err_th', SAGITTAL_LEFT_THRESHOLDS['foot_err_th']))
        }
        
        if video_file and person_name:
            service = SquatAnalysisService()
            result = service.analyze_sagittal(video_file, person_name, 'esquerdo', user_height_cm, params)
            context['result'] = result
            context['person_name'] = person_name
    
    return render(request, 'squat_analyzer/sagittal_left_analysis.html', context)


def sagittal_right_analysis(request):
    """
    View para análise sagital - lado direito.
    GET: Exibe formulário de upload
    POST: Processa vídeo e exibe resultados
    """
    context = {
        'side': 'direito',
        'analysis_type': 'sagittal',
        'title': 'Análise Sagital Direito',
        'thresholds': SAGITTAL_RIGHT_THRESHOLDS,
    }
    
    if request.method == 'POST':
        video_file = request.FILES.get('video')
        person_name = request.POST.get('person_name')
        user_height_cm = float(request.POST.get('user_height_cm', 170))
        
        # Parâmetros de análise
        params = {
            'descent_threshold': float(request.POST.get('descent_threshold', SAGITTAL_RIGHT_THRESHOLDS['descent_th'])),
            'ascent_return_threshold': float(request.POST.get('ascent_return_threshold', SAGITTAL_RIGHT_THRESHOLDS['ascent_return_th'])),
            'trunk_error_threshold': int(request.POST.get('trunk_err_th', SAGITTAL_RIGHT_THRESHOLDS['trunk_err_th'])),
            'knee_error_threshold': int(request.POST.get('knee_err_th', SAGITTAL_RIGHT_THRESHOLDS['knee_err_th'])),
            'head_error_threshold': int(request.POST.get('head_err_th', SAGITTAL_RIGHT_THRESHOLDS['head_err_th'])),
            'foot_error_threshold': int(request.POST.get('foot_err_th', SAGITTAL_RIGHT_THRESHOLDS['foot_err_th']))
        }
        
        if video_file and person_name:
            service = SquatAnalysisService()
            result = service.analyze_sagittal(video_file, person_name, 'direito', user_height_cm, params)
            context['result'] = result
            context['person_name'] = person_name
    
    return render(request, 'squat_analyzer/sagittal_right_analysis.html', context)


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
