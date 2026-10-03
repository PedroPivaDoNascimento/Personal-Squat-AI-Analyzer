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
        # np.hypot e implementado em C no NumPy: evita as operacoes
        # intermediarias de **2 (alocacao de arrays temporarios), sendo
        # mais rapido e a prova de overflow numerico (Otimizacao Fase 3.4).
        return np.hypot(x2 - x1, y2 - y1)

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
        # 1. Vetores a partir do vertice (p2) calculados com operacoes
        #    elementares diretas via NumPy/scalar math: evita a alocacao de
        #    tres arrays intermediarios (p1, p2, p3) por chamada - esta
        #    funcao roda a cada frame e o custo se acumula no loop de video
        #    (Otimizacao Fase 3.4). O contrato matematico e os resultados
        #    permanecem identicos.
        v21_x = x1 - x2
        v21_y = y1 - y2
        v23_x = x3 - x2
        v23_y = y3 - y2

        # 2. Componentes para o atan2:
        #    "seno" = Produto Vetorial (z-componente em 2D); o sinal deste
        #    valor indica se o angulo e positivo ou negativo (sentido).
        cross_product_z = v21_x * v23_y - v21_y * v23_x

        #    "cosseno" = Produto Escalar.
        dot_product = v21_x * v23_x + v21_y * v23_y

        # 3. np.arctan2 usa seno e cosseno para obter o angulo no intervalo
        #    [-pi, pi] e retorna graus em [-180.0, +180.0], conforme o
        #    contrato de angulos orientados da docs/RULES.md item 1.
        return float(np.degrees(np.arctan2(cross_product_z, dot_product)))