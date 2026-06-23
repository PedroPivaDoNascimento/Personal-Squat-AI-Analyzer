"""
Vector Calculator - Utility class for geometric calculations.

Esta classe foi migrada do código original (src/classes/vector_calculator.py)
e mantém exatamente a mesma lógica de negócio para preservação dos cálculos.

Responsabilidade: Realizar cálculos vetoriais e geométricos utilizados na
análise de movimento do agachamento.
"""

import math
from typing import Tuple


class VectorCalculator:
    """
    Calculadora de operações vetoriais para análise biomecânica.
    
    Esta classe fornece métodos estáticos para cálculos de distância,
    ângulos e interseções utilizados na análise de postura.
    
    Design Pattern: Utility Class (Singleton implícito via métodos estáticos)
    - Todos os métodos são estáticos pois não mantêm estado interno
    - Facilita o uso sem necessidade de instanciação
    """
    
    @staticmethod
    def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        """
        Calcula a distância euclidiana entre dois pontos.
        
        Args:
            x1: Coordenada X do primeiro ponto
            y1: Coordenada Y do primeiro ponto
            x2: Coordenada X do segundo ponto
            y2: Coordenada Y do segundo ponto
            
        Returns:
            Distância euclidiana entre os pontos
            
        Exemplo:
            >>> VectorCalculator.calculate_distance(0, 0, 3, 4)
            5.0
        """
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    @staticmethod
    def calculate_angle_three_points(
        point_a: Tuple[float, float],
        point_b: Tuple[float, float],
        point_c: Tuple[float, float]
    ) -> float:
        """
        Calcula o ângulo formado por três pontos (A-B-C).
        
        O ângulo é calculado no ponto B (vértice).
        
        Args:
            point_a: Primeiro ponto como tupla (x, y)
            point_b: Ponto do vértice como tupla (x, y)
            point_c: Terceiro ponto como tupla (x, y)
            
        Returns:
            Ângulo em graus (0-180)
            
        Exemplo:
            >>> VectorCalculator.calculate_angle_three_points((0, 0), (1, 1), (2, 0))
            90.0
        """
        # Vetores BA e BC
        ba_x = point_a[0] - point_b[0]
        ba_y = point_a[1] - point_b[1]
        bc_x = point_c[0] - point_b[0]
        bc_y = point_c[1] - point_b[1]
        
        # Produto escalar
        dot_product = ba_x * bc_x + ba_y * bc_y
        
        # Magnitudes dos vetores
        magnitude_ba = math.sqrt(ba_x ** 2 + ba_y ** 2)
        magnitude_bc = math.sqrt(bc_x ** 2 + bc_y ** 2)
        
        # Evitar divisão por zero
        if magnitude_ba == 0 or magnitude_bc == 0:
            return 0.0
        
        # Calcular cosseno do ângulo
        cos_angle = dot_product / (magnitude_ba * magnitude_bc)
        
        # Clamp para evitar erros de precisão
        cos_angle = max(-1.0, min(1.0, cos_angle))
        
        # Converter para graus
        angle_degrees = math.degrees(math.acos(cos_angle))
        
        return angle_degrees
    
    @staticmethod
    def calculate_line_intersection(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        p4: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Calcula o ponto de interseção entre duas linhas definidas por pares de pontos.
        
        Args:
            p1: Primeiro ponto da linha 1 como tupla (x, y)
            p2: Segundo ponto da linha 1 como tupla (x, y)
            p3: Primeiro ponto da linha 2 como tupla (x, y)
            p4: Segundo ponto da linha 2 como tupla (x, y)
            
        Returns:
            Tupla (x, y) com as coordenadas do ponto de interseção
            Retorna (None, None) se as linhas forem paralelas
            
        Exemplo:
            >>> VectorCalculator.calculate_line_intersection((0, 0), (2, 2), (0, 2), (2, 0))
            (1.0, 1.0)
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # Denominador da fórmula
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        # Verificar se linhas são paralelas
        if abs(denom) < 1e-10:
            return (None, None)
        
        # Calcular coordenadas da interseção
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        
        return (px, py)
    
    @staticmethod
    def normalize_vector(x: float, y: float) -> Tuple[float, float]:
        """
        Normaliza um vetor unitário.
        
        Args:
            x: Componente X do vetor
            y: Componente Y do vetor
            
        Returns:
            Tupla (x, y) com o vetor normalizado
        """
        magnitude = math.sqrt(x ** 2 + y ** 2)
        if magnitude == 0:
            return (0.0, 0.0)
        return (x / magnitude, y / magnitude)
    
    @staticmethod
    def calculate_slope(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """
        Calcula a inclinação (slope) de uma linha entre dois pontos.
        
        Args:
            p1: Primeiro ponto como tupla (x, y)
            p2: Segundo ponto como tupla (x, y)
            
        Returns:
            Inclinação da linha (float) ou float('inf') se vertical
        """
        x1, y1 = p1
        x2, y2 = p2
        
        if x2 == x1:
            return float('inf')  # Linha vertical
        
        return (y2 - y1) / (x2 - x1)
