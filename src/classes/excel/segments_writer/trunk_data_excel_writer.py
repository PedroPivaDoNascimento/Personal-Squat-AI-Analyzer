"""
Escritor de dados de TRONCO (task (3)NEW-DATA_ANALYTICS-SHEETS.md, Fase 2.7).

Segmento `tronco`: ombros e quadris (landmarks 11/12 ombros, 23/24 quadris).
Coleta coordenadas normalizadas (X, Y, Z) e calcula inclinação e descritores
estatísticos da série temporal da repetição.
"""
import pandas as pd

from classes.excel.segments_writer.segments_writer import SegmentsWriter
from classes.vector_calculator import VectorCalculator


class TrunkDataExcelWriter(SegmentsWriter):
    """Gera as planilhas de dados brutos e estatísticos do segmento `tronco`."""

    @property
    def segment_name(self):
        return 'tronco'

    def _build_features(self, df_temp):
        """
        Deriva métricas biomecânicas de inclinação do tronco frame a frame quando
        as coordenadas de ombro e quadril estiverem presentes.
        """
        df = df_temp.copy()

        def find_col(*candidates):
            for cand in candidates:
                for col in df.columns:
                    if col == cand or col.endswith(cand):
                        return col
            return None

        shoulder_x = find_col('shoulder_x')
        shoulder_y = find_col('shoulder_y')
        hip_x = find_col('hip_x')
        hip_y = find_col('hip_y')

        if shoulder_x and shoulder_y and hip_x and hip_y:
            angulos = []
            for sx, sy, hx, hy in zip(df[shoulder_x], df[shoulder_y], df[hip_x], df[hip_y]):
                try:
                    angulos.append(VectorCalculator.angle_to_horizontal(
                        float(hx), float(hy), float(sx), float(sy)
                    ))
                except (TypeError, ValueError):
                    angulos.append(float('nan'))
            df['tronco_inclinacao_horizontal_graus'] = angulos

        return df

    def convert_data_to_statistic_pandas(self):
        """
        Calcula estatísticas por coluna do tronco: média, mediana, desvio padrão,
        mínimo, máximo, amplitude e IQR.
        """
        df_temp = self._to_dataframe()
        if df_temp is None:
            return pd.DataFrame()

        df_enriched = self._build_features(df_temp)

        estatisticas_linha = [self.person_name, self.repetition]
        nomes_colunas = ['voluntario', 'repeticao']

        for coluna in df_enriched.columns:
            dados = pd.to_numeric(df_enriched[coluna], errors='coerce').dropna()
            if dados.empty:
                media = mediana = desvio_padrao = p5 = p95 = amplitude = iqr = 0.0
            else:
                desvio_padrao = float(dados.std(ddof=0))
                iqr = float(dados.quantile(0.75) - dados.quantile(0.25))
                media = float(dados.mean())
                mediana = float(dados.median())
                p5 = float(dados.quantile(0.05))
                p95 = float(dados.quantile(0.95))
                amplitude = p95 - p5

            estatisticas_linha.extend([media, mediana, desvio_padrao, p5, p95, amplitude, iqr])
            prefixo = str(coluna)
            nomes_colunas.extend([
                f'{prefixo}_media',
                f'{prefixo}_mediana',
                f'{prefixo}_std',
                f'{prefixo}_p5',
                f'{prefixo}_p95',
                f'{prefixo}_amplitude',
                f'{prefixo}_iqr',
            ])

        return pd.DataFrame([estatisticas_linha], columns=nomes_colunas)
