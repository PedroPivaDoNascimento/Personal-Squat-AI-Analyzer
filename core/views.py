"""
Views do Django para a aplicação de Análise de Agachamento.

Estas views implementam o padrão MVC do Django:
- Recebem requisições HTTP (GET/POST)
- Processam dados usando o serviço de análise (regra de negócio)
- Renderizam templates HTML com os resultados

Fluxo que antes era do Streamlit agora é tratado via Request/Response:
- Navegação entre páginas → URLs e Views separadas
- Session state → Formulários POST e contexto do template
- st.rerun() → Redirecionamentos Django
- st.file_uploader → Django File Upload
- st.slider/st.number_input → Formulários HTML
"""

import os
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.conf import settings

from .services.analysis_service import SquatAnalysisService


def index(request):
    """
    Página inicial - Seleção do tipo de análise.
    
    Substitui a tela de seleção do Streamlit (show_selection_page).
    """
    return render(request, 'core/index.html')


def sagittal_right_analysis(request):
    """
    Análise Sagital Direita.
    
    GET: Exibe formulário com parâmetros
    POST: Processa vídeo e exibe resultados
    """
    if request.method == 'POST':
        return _process_analysis(
            request,
            analysis_type='sagittal_right',
            side='right',
            requires_height=True
        )
    
    params = SquatAnalysisService.get_default_params('sagittal_right')
    return render(request, 'core/sagittal_analysis.html', {
        'analysis_type': 'Sagital Direito',
        'side': 'Direito',
        'requires_height': True,
        'params': params,
    })


def sagittal_left_analysis(request):
    """
    Análise Sagital Esquerda.
    """
    if request.method == 'POST':
        return _process_analysis(
            request,
            analysis_type='sagittal_left',
            side='left',
            requires_height=True
        )
    
    params = SquatAnalysisService.get_default_params('sagittal_left')
    return render(request, 'core/sagittal_analysis.html', {
        'analysis_type': 'Sagital Esquerdo',
        'side': 'Esquerdo',
        'requires_height': True,
        'params': params,
    })


def frontal_right_analysis(request):
    """
    Análise Frontal Direita.
    
    Diferente do sagital, inclui checkboxes para selecionar repetições.
    """
    if request.method == 'POST':
        return _process_frontal_analysis(request, side='right')
    
    params = SquatAnalysisService.get_default_params('frontal_right')
    return render(request, 'core/frontal_analysis.html', {
        'analysis_type': 'Frontal Direito',
        'side': 'Direito',
        'params': params,
    })


def frontal_left_analysis(request):
    """
    Análise Frontal Esquerda.
    """
    if request.method == 'POST':
        return _process_frontal_analysis(request, side='left')
    
    params = SquatAnalysisService.get_default_params('frontal_left')
    return render(request, 'core/frontal_analysis.html', {
        'analysis_type': 'Frontal Esquerdo',
        'side': 'Esquerdo',
        'params': params,
    })


def _process_analysis(request, analysis_type: str, side: str, requires_height: bool):
    """
    Processa análise de vídeo (sagital ou frontal).
    
    Esta função encapsula a lógica comum de processamento:
    1. Valida dados do formulário
    2. Salva arquivo temporário
    3. Chama o serviço de análise
    4. Retorna resultados para o template
    """
    person_name = request.POST.get('person_name', '').strip()
    
    if requires_height:
        try:
            user_height_cm = int(request.POST.get('user_height_cm', 170))
        except (ValueError, TypeError):
            user_height_cm = 170
    else:
        user_height_cm = None
    
    # Extrair parâmetros do formulário
    params = {
        'descent_threshold': float(request.POST.get('descent_threshold', 0.05)),
        'ascent_return_threshold': float(request.POST.get('ascent_return_threshold', 0.02)),
    }
    
    # Parâmetros específicos do sagital
    if 'sagittal' in analysis_type:
        params.update({
            'trunk_error_threshold': int(request.POST.get('trunk_error_threshold', 23)),
            'knee_error_threshold': int(request.POST.get('knee_error_threshold', 6)),
            'head_error_threshold': int(request.POST.get('head_error_threshold', 2)),
            'foot_error_threshold': int(request.POST.get('foot_error_threshold', 8)),
        })
    
    # Validar upload de arquivo
    uploaded_file = request.FILES.get('video_file')
    if not uploaded_file or not person_name:
        return render(request, 'core/error.html', {
            'error_message': 'Por favor, preencha o nome e envie um vídeo.'
        })
    
    # Salvar arquivo temporário
    fs = FileSystemStorage(location=settings.MEDIA_ROOT)
    filename = fs.save(f"temp_{analysis_type}_{uploaded_file.name}", uploaded_file)
    video_path = fs.path(filename)
    
    try:
        # Executar análise
        service = SquatAnalysisService()
        
        if 'sagittal' in analysis_type:
            result = service.analyze_sagittal(
                video_path=video_path,
                person_name=person_name,
                side=side,
                user_height_cm=user_height_cm,
                params=params
            )
        else:
            selected_reps = _get_selected_repetitions(request)
            result = service.analyze_frontal(
                video_path=video_path,
                person_name=person_name,
                side=side,
                params=params,
                selected_repetitions=selected_reps
            )
        
        # Renderizar resultados
        return render(request, 'core/results.html', {
            'result': result,
            'analysis_type': analysis_type,
        })
        
    except Exception as e:
        # Limpar arquivo em caso de erro
        if os.path.exists(video_path):
            os.remove(video_path)
        return render(request, 'core/error.html', {
            'error_message': f'Erro durante a análise: {str(e)}'
        })


def _process_frontal_analysis(request, side: str):
    """Processa especificamente análise frontal."""
    return _process_analysis(
        request,
        analysis_type=f'frontal_{side}',
        side=side,
        requires_height=False
    )


def _get_selected_repetitions(request) -> list:
    """
    Extrai repetições selecionadas dos checkboxes do formulário frontal.
    
    No Streamlit era: c1 = st.checkbox("Salvar repetição 1")
    No Django: request.POST.getlist('selected_repetitions')
    """
    selected = request.POST.getlist('selected_repetitions')
    return [int(rep) for rep in selected if rep.isdigit()]

