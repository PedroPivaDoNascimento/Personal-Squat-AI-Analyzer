"""
Strategy Pattern - Base class for implementing the Strategy design pattern.

O padrão Strategy permite definir uma família de algoritmos, encapsular cada um deles
e torná-los intercambiáveis. Isso permite que o algoritmo varie independentemente
dos clientes que o utilizam.

No contexto deste projeto, usamos Strategy para:
- Diferentes tipos de análise (Sagital vs Frontal)
- Diferentes geradores de relatório (Excel, PDF, etc.)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class Strategy(ABC):
    """
    Classe base abstrata para todas as estratégias do sistema.
    
    Atributos:
        name (str): Nome identificador da estratégia
    
    Exemplo de uso:
        class SagittalAnalysisStrategy(Strategy):
            def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
                # Implementação específica para análise sagital
                pass
    """
    
    def __init__(self, name: str = "base_strategy"):
        """
        Inicializa a estratégia com um nome identificador.
        
        Args:
            name: Nome identificador da estratégia
        """
        self._name = name
    
    @property
    def name(self) -> str:
        """Retorna o nome da estratégia."""
        return self._name
    
    @abstractmethod
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa a estratégia com os dados fornecidos.
        
        Args:
            data: Dados de entrada para processamento
            
        Returns:
            Dicionário com os resultados do processamento
            
        Raises:
            NotImplementedError: Se a subclasse não implementar este método
        """
        raise NotImplementedError("Subclasses devem implementar o método execute()")
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """
        Valida os dados de entrada antes da execução.
        
        Args:
            data: Dados a serem validados
            
        Returns:
            True se os dados são válidos, False caso contrário
        """
        return data is not None and isinstance(data, dict)
