"""
Views para análise de agachamento - Camada de Controller
Responsável por receber requisições HTTP e orquestrar o fluxo.
"""
import os
from django.shortcuts import render, redirect
from django.core.files.uploadedfile import UploadedFile
from django.http import FileResponse, Http404
from .services.analysis_service import SquatAnalysisService


def index(request):
    """Página inicial com seleção do tipo de análise."""
    return render(request, 'squat_analyzer/index.html')


def frontal_analysis(request, side):
    """
    View para análise frontal (direito ou esquerdo).
    GET: Exibe formulário de upload
    POST: Processa vídeo e exibe resultados
    """
    if side not in ['direito', 'esquerdo']:
        return redirect('index')
    
    context = {
        'side': side,
        'analysis_type': 'frontal',
        'title': f'Análise Frontal {side.capitalize()}'
    }
    
    if request.method == 'POST':
        video_file = request.FILES.get('video')
        person_name = request.POST.get('person_name')
        
        # Parâmetros de análise
        params = {
            'descent_threshold': float(request.POST.get('descent_threshold', 0.05)),
            'ascent_return_threshold': float(request.POST.get('ascent_return_threshold', 0.02)),
            'hip_error_threshold': int(request.POST.get('hip_error_threshold', 1)),
            'knee_valgus_error_threshold': int(request.POST.get('knee_valgus_error_threshold', 12)),
            'foot_pronation_error_threshold': int(request.POST.get('foot_pronation_error_threshold', 7))
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
            result = service.analyze_frontal(video_file, person_name, side, params, selected_reps)
            context['result'] = result
            context['person_name'] = person_name
    
    return render(request, 'squat_analyzer/frontal_analysis.html', context)


def sagittal_analysis(request, side):
    """
    View para análise sagital (direito ou esquerdo).
    GET: Exibe formulário de upload
    POST: Processa vídeo e exibe resultados
    """
    if side not in ['direito', 'esquerdo']:
        return redirect('index')
    
    context = {
        'side': side,
        'analysis_type': 'sagittal',
        'title': f'Análise Sagital {side.capitalize()}'
    }
    
    if request.method == 'POST':
        video_file = request.FILES.get('video')
        person_name = request.POST.get('person_name')
        user_height_cm = float(request.POST.get('user_height_cm', 170))
        
        # Parâmetros de análise
        params = {
            'descent_threshold': float(request.POST.get('descent_threshold', 0.05)),
            'ascent_return_threshold': float(request.POST.get('ascent_return_threshold', 0.02)),
            'trunk_error_threshold': int(request.POST.get('trunk_error_threshold', 23)),
            'knee_error_threshold': int(request.POST.get('knee_error_threshold', 6)),
            'head_error_threshold': int(request.POST.get('head_error_threshold', 2)),
            'foot_error_threshold': int(request.POST.get('foot_error_threshold', 8))
        }
        
        if video_file and person_name:
            service = SquatAnalysisService()
            result = service.analyze_sagittal(video_file, person_name, side, user_height_cm, params)
            context['result'] = result
            context['person_name'] = person_name
    
    return render(request, 'squat_analyzer/sagittal_analysis.html', context)


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
