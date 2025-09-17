import numpy as np
from typing import Tuple

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