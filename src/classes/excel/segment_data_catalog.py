"""
Catálogo de segmentos biomecânicos monitorados - docs/tasks/(3)NEW-DATAXANALYTICS-SHEETS.md

Define, para cada segmento (Cabeça, Tronco, Quadril, Joelho Frontal,
Joelho Sagital, Pé Frontal e Pé Sagital), os landmarks MediaPipe Pose
correspondentes (nome, índice PoseLandmarker e coordenadas/eixos de
interesse) exatamente conforme a tabela de especificação do plano de
implementação.

Responsabilidade única: ser a fonte canônica ("single source of truth")
de mapeamento segmento -> landmark, consumida pelos coletores de dados
brutos/estatísticos dos analisadores frontal e sagital.
"""

# Coordenadas coletadas por landmark nas séries temporais brutas
# (X, Y, Z normalizados + visibilidade, conforme tarefa 2.1).
SEGMENT_COORDINATE_AXES = ('x', 'y', 'z', 'visibility')


def _lm(name, index, axes):
    """Constrói a descrição de um landmark do catálogo."""
    return {'name': name, 'index': index, 'axes': tuple(axes)}


# Índices oficiais PoseLandmarker (MediaPipe):
# 0 nose | 2/5 eyes | 7/8 ears | 11/12 shoulders | 23/24 hips
# 25/26 knees | 27/28 ankles | 29/30 heels | 31/32 big toes
FRONT_SEGMENT_LANDMARKS = {
    'cabeca': {
        'display_name': 'Cabeca',
        'description': 'Posicao vertical (Y) para maquina de fases e alinhamento axial',
        'landmarks': (
            _lm('nose', 0, 'x,y'),
            _lm('left_eye', 2, 'x,y'),
            _lm('right_eye', 5, 'x,y'),
            _lm('left_ear', 7, 'x,y'),
            _lm('right_ear', 8, 'x,y'),
        ),
    },
    'tronco': {
        'display_name': 'Tronco',
        'description': 'Inclinacao anterior/posterior do tronco em relacao a vertical (X, Y)',
        'landmarks': (
            _lm('left_shoulder', 11, 'x,y'),
            _lm('right_shoulder', 12, 'x,y'),
        ),
    },
    'quadril': {
        'display_name': 'Quadril',
        'description': 'Bascula pelvica no plano frontal e deslocamento sagital (X, Y)',
        'landmarks': (
            _lm('left_hip', 23, 'x,y'),
            _lm('right_hip', 24, 'x,y'),
        ),
    },
    'joelho_frontal': {
        'display_name': 'Joelho Frontal',
        'description': 'Angulo de projecao frontal (Valgo Dinamico / HKA) (X, Y)',
        'landmarks': (
            _lm('left_knee', 25, 'x,y'),
            _lm('right_knee', 26, 'x,y'),
        ),
    },
    'pe_frontal': {
        'display_name': 'Pe Frontal',
        'description': 'Deslocamento medial/lateral para predicao de pronacao dinamica (X, Y, Z)',
        'landmarks': (
            _lm('left_ankle', 27, 'x,y,z'),
            _lm('right_ankle', 28, 'x,y,z'),
            _lm('left_heel', 29, 'x,y,z'),
            _lm('right_heel', 30, 'x,y,z'),
            _lm('left_big_toe', 31, 'x,y,z'),
            _lm('right_big_toe', 32, 'x,y,z'),
        ),
    },
}

SAGITTAL_SEGMENT_LANDMARKS = {
    'cabeca': FRONT_SEGMENT_LANDMARKS['cabeca'],
    'tronco': FRONT_SEGMENT_LANDMARKS['tronco'],
    'quadril': FRONT_SEGMENT_LANDMARKS['quadril'],
    'joelho_sagital': {
        'display_name': 'Joelho Sagital',
        'description': 'Angulo de flexao do joelho e relacao joelho-calcanhar/ponta (X, Y)',
        'landmarks': (
            _lm('left_knee', 25, 'x,y'),
            _lm('right_knee', 26, 'x,y'),
        ),
    },
    'pe_sagital': {
        'display_name': 'Pe Sagital',
        'description': 'Elevacao precoce do calcanhar e dorsiflexao do tornozelo (X, Y)',
        'landmarks': (
            _lm('left_ankle', 27, 'x,y'),
            _lm('right_ankle', 28, 'x,y'),
            _lm('left_heel', 29, 'x,y'),
            _lm('right_heel', 30, 'x,y'),
            _lm('left_big_toe', 31, 'x,y'),
            _lm('right_big_toe', 32, 'x,y'),
        ),
    },
}


def get_segment_catalog(plane_folder_name):
    """
    Retorna o catálogo de segmentos do plano informado.

    Args:
        plane_folder_name (str): 'frontal' ou 'sagital'.

    Returns:
        dict: mapeamento chave-do-segmento -> especificação (display_name,
              description e landmarks). Dicionário vazio para plano desconhecido.
    """
    if plane_folder_name == 'frontal':
        return FRONT_SEGMENT_LANDMARKS
    if plane_folder_name == 'sagital':
        return SAGITTAL_SEGMENT_LANDMARKS
    return {}
