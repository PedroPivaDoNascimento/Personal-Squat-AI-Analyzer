"""
Views para análise de agachamento - Camada de Controller
Responsável por receber requisições HTTP e orquestrar o fluxo.
"""
from django.shortcuts import render, redirect
from django.core.files.uploadedfile import UploadedFile
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
