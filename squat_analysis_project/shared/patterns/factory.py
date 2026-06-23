"""
Factory Pattern - Base class for implementing the Factory design pattern.

O padrão Factory fornece uma interface para criar objetos em uma superclasse,
enquanto permite às subclasses alterar o tipo de objetos que serão criados.

No contexto deste projeto, usamos Factory para:
- Criar diferentes tipos de analisadores (Sagital vs Frontal)
- Criar diferentes geradores de relatório baseado no plano de análise
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Type


class Factory(ABC):
    """
    Classe base abstrata para todas as fábricas do sistema.
    
    A fábrica é responsável por criar instâncias de classes concretas
    baseando-se em parâmetros de entrada, sem expor a lógica de criação
    ao código cliente.
    
    Exemplo de uso:
        class AnalyzerFactory(Factory):
            def create(self, analyzer_type: str, **kwargs) -> Any:
                if analyzer_type == "sagittal":
                    return SagittalAnalyzer(**kwargs)
                elif analyzer_type == "frontal":
                    return FrontalAnalyzer(**kwargs)
    """
    
    @abstractmethod
    def create(self, product_type: str, **kwargs: Any) -> Any:
        """
        Cria uma instância do produto solicitado.
        
        Args:
            product_type: Tipo do produto a ser criado
            **kwargs: Argumentos adicionais para inicialização do produto
            
        Returns:
            Instância do produto criado
            
        Raises:
            NotImplementedError: Se a subclasse não implementar este método
            ValueError: Se o tipo de produto não for reconhecido
        """
        raise NotImplementedError("Subclasses devem implementar o método create()")
    
    def _validate_product_type(self, product_type: str, valid_types: list) -> bool:
        """
        Valida se o tipo de produto é válido.
        
        Args:
            product_type: Tipo do produto a validar
            valid_types: Lista de tipos válidos
            
        Returns:
            True se o tipo é válido
            
        Raises:
            ValueError: Se o tipo não for válido
        """
        if product_type not in valid_types:
            raise ValueError(
                f"Tipo de produto '{product_type}' não é válido. "
                f"Tipos válidos: {', '.join(valid_types)}"
            )
        return True
    
    def register_product(self, product_type: str, product_class: Type) -> None:
        """
        Registra um novo tipo de produto na fábrica.
        
        Este método permite extensão dinâmica da fábrica sem modificação
        do código existente (princípio Open/Closed).
        
        Args:
            product_type: Nome do tipo de produto
            product_class: Classe do produto a ser registrada
        """
        if not hasattr(self, '_registry'):
            self._registry = {}
        self._registry[product_type] = product_class
