import numpy as np
from typing import Tuple
import math

class VectorCalculator:
    """
    Classe responsável por realizar cálculos vetoriais para análise de movimentos.
    """

    @staticmethod
    def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        """
        Calcula a distância euclidiana entre dois pontos (x, y).

        Args:
            x1: A coordenada x do primeiro ponto.
            y1: A coordenada y do primeiro ponto.
            x2: A coordenada x do segundo ponto.
            y2: A coordenada y do segundo ponto.

        Returns:
            A distância euclidiana entre os dois pontos.
        """
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    @staticmethod
    def get_line_equation(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float]:
        """
        Calcula a equação da reta (ax + by + c = 0) que passa por dois pontos.

        Args:
            x1: A coordenada x do primeiro ponto.
            y1: A coordenada y do primeiro ponto.
            x2: A coordenada x do segundo ponto.
            y2: A coordenada y do segundo ponto.

        Returns:
            Uma tupla (a, b, c) com os coeficientes da equação da reta.
        """
        a = y2 - y1
        b = x1 - x2
        c = -a * x1 - b * y1

        return a, b, c

    @staticmethod
    def find_line_intersection(line1: Tuple[float, float, float], line2: Tuple[float, float, float]):
        """
        Encontra o ponto de interseção de duas retas.

        Args:
            line1: Uma tupla (a, b, c) com os coeficientes da primeira reta.
            line2: Uma tupla (a, b, c) com os coeficientes da segunda reta.

        Returns:
            Uma tupla (x, y) com as coordenadas do ponto de interseção, ou None se as retas forem paralelas.
        """
        a1, b1, c1 = line1
        a2, b2, c2 = line2

        determinant = a1 * b2 - a2 * b1

        if determinant == 0:
            # As retas são paralelas ou coincidentes
            return None
        else:
            x = (b1 * c2 - b2 * c1) / determinant
            y = (a2 * c1 - a1 * c2) / determinant
            return x, y
        
    @staticmethod
    def angle_to_horizontal(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1 
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        angle_deg = angle_deg % 360
        return min(abs(angle_deg), abs(180 - angle_deg))
    
    @staticmethod
    def calculate_angle_3p(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> float:
        """
        Calcula o ângulo em graus formado no ponto central (p2)
        pelos três pontos: p1(x1, y1), p2(x2, y2) e p3(x3, y3).

        Parâmetros:
        x1, y1: Coordenadas do primeiro ponto (p1).
        x2, y2: Coordenadas do ponto central/vértice do ângulo (p2).
        x3, y3: Coordenadas do terceiro ponto (p3).

        Retorna:
        O ângulo em graus.
        """
        p1 = np.array((x1, y1))
        p2 = np.array((x2, y2))
        p3 = np.array((x3, y3))

        # Vetores v21 (de p2 a p1) e v23 (de p2 a p3)
        v21 = p1 - p2
        v23 = p3 - p2

        # Produto escalar (Lei dos Cossenos)
        dot_product = np.dot(v21, v23)
        norm_v21 = np.linalg.norm(v21)
        norm_v23 = np.linalg.norm(v23)

        # Trata o caso de pontos colineares ou coincidentes
        if norm_v21 == 0 or norm_v23 == 0:
            return 180.0

        cos_angle = dot_product / (norm_v21 * norm_v23)

        # Garante que o argumento do arccos esteja entre -1.0 e 1.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # Converte para graus
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        return angle_deg