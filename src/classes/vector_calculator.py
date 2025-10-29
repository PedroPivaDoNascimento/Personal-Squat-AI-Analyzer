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
        return angle_deg
    
    @staticmethod
    def calculate_angle_3p(x1, y1, x2, y2, x3, y3):
        """
        Calcula o ângulo em graus com base no ATAN2, o que permite determinar
        o sentido (horário/anti-horário) do ângulo, retornando valores
        entre -180.0 e +180.0 graus.

        Parâmetros:
        x1, y1: Coordenadas do primeiro ponto (p1, ex: Quadril).
        x2, y2: Coordenadas do ponto central/vértice (p2, ex: Joelho).
        x3, y3: Coordenadas do terceiro ponto (p3, ex: Tornozelo).

        Retorna:
        O ângulo em graus (valor entre -180.0 e 180.0).
        """
        # 1. Converte as coordenadas para arrays numpy
        p1 = np.array((x1, y1))
        p2 = np.array((x2, y2))
        p3 = np.array((x3, y3))

        # 2. Cria os vetores a partir do vértice (p2)
        v21 = p1 - p2
        v23 = p3 - p2

        # 3. Calcula os componentes necessários para o atan2
        
        # O "seno" (componente y) é o Produto Vetorial (z-componente em 2D)
        # Produto Vetorial em 2D: v1_x * v2_y - v1_y * v2_x
        # O sinal deste valor indica se o ângulo é positivo ou negativo (sentido).
        cross_product_z = v21[0] * v23[1] - v21[1] * v23[0]

        # O "cosseno" (componente x) é o Produto Escalar (o que você já estava usando)
        dot_product = np.dot(v21, v23)

        # 4. Usa np.arctan2 (a versão do atan2 no numpy)
        # O atan2 usa seno e cosseno para obter o ângulo no intervalo [-pi, pi]
        angle_rad = np.arctan2(cross_product_z, dot_product)
        
        # 5. Converte para Graus
        angle_deg = np.degrees(angle_rad)

        return angle_deg