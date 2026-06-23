"""
Feedback Messages - Mensagens de feedback para análise de exercícios.

Este módulo foi migrado do código original (src/ultils/feedback_messages.py)
e mantém exatamente as mesmas mensagens para preservação da regra de negócio.

Responsabilidade: Fornecer mensagens padronizadas de feedback para o usuário
sobre desvios de postura detectados durante a análise.
"""

from typing import Dict


# Dicionário de mensagens de feedback para cada tipo de erro
feedback_messages: Dict[str, str] = {
    'trunk_error': (
        "Verifique a inclinação do tronco: Mantenha o peito aberto e a coluna neutra. "
        "Evite curvar as costas excessivamente para frente ou para trás."
    ),
    'knee_error': (
        "Atenção ao alinhamento dos joelhos: Certifique-se de que seus joelhos "
        "sigam a linha dos pés, evitando que eles se desviem para dentro."
    ),
    'head_error': (
        "Cuidado com a posição da cabeça: Mantenha o olhar neutro e o pescoço alinhado "
        "com a coluna. Evite olhar excessivamente para cima ou para baixo."
    ),
    'heel_error': (
        "Não levante os calcanhares: Mantenha os calcanhares firmemente plantados no chão "
        "durante todo o movimento. Se necessário, ajuste a base dos pés."
    ),
    'hip_error': (
        "Verifique a estabilidade lateral do quadril."
    ),
    'knee_valgus_error': (
        "Valgo ou Varo detectado. Fortaleça abdutores."
    ),
    'foot_pronation_error': (
        "Pronação excessiva. Fortaleça a musculatura intrínseca do pé."
    ),
}


def get_feedback_message(error_type: str, default: str = "Execute o movimento corretamente.") -> str:
    """
    Retorna a mensagem de feedback para um tipo específico de erro.
    
    Args:
        error_type: Tipo do erro (ex: 'trunk_error', 'knee_error')
        default: Mensagem padrão caso o tipo não seja encontrado
        
    Returns:
        Mensagem de feedback formatada
        
    Exemplo:
        >>> get_feedback_message('trunk_error')
        'Verifique a inclinação do tronco...'
    """
    return feedback_messages.get(error_type, default)


def get_all_feedback_messages() -> Dict[str, str]:
    """
    Retorna todas as mensagens de feedback disponíveis.
    
    Returns:
        Dicionário completo com todas as mensagens
    """
    return feedback_messages.copy()
