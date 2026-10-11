"""
Escritor de dados de QUADRIL (task (3)NEW-DATA_ANALYTICS-SHEETS.md, Fase 2.6).

Segmento `quadril`: articulação coxofemoral (landmarks MediaPipe 23/24 —
`left_hip`, `right_hip`). Coleta coordenadas normalizadas (X, Y, Z) e
visibilidade nos planos sagital e frontal. Estatísticas por coluna: média,
mediana, std, min, max, amplitude e IQR, além da derivada biomecânica
`quadril_inclinacao_horizontal_graus` (ângulo da linha entre os dois quadris
relativa à horizontal — inclinação pélvica no plano frontal), calculada com o
`VectorCalculator` central conforme docs/RULES.md.
"""
import pandas as pd

from classes.excel.segments_writer.segments_writer import SegmentsWriter
from classes.vector_calculator import VectorCalculator


class HipDataExcelWriter(SegmentsWriter):
    """Gera as planilhas de dados brutos e estatísticos do segmento `quadril`."""

    @property
    def segment_name(self):
        return 'quadril'

    def _build_features(self, df_temp):
        """
        Deriva a inclinação pélvica horizontal frame a frame quando houver
        colunas de quadril esquerdo e direito simultaneamente (plano frontal).

        Args:
            df_temp (pandas.DataFrame): Série temporal bruta do quadril.

        Returns:
            pandas.DataFrame: Série acrescida da coluna derivada (se aplicável).
        """
        df = df_temp.copy()

        left_x = next((c for c in df.columns if 'left_hip_x' in c), None)
        left_y = next((c for c in df.columns if 'left_hip_y' in c), None)
        right_x = next((c for c in df.columns if 'right_hip_x' in c), None)
        right_y = next((c for c in df.columns if 'right_hip_y' in c), None)

        if left_x and left_y and right_x and right_y:
            angulos = []
            for lx, ly, rx, ry in zip(df[left_x], df[left_y], df[right_x], df[right_y]):
                try:
                    # Ângulo orientado da crista ilíaca relativa ao horizonte
                    angulos.append(VectorCalculator.angle_to_horizontal(
                        float(lx), float(ly), float(rx), float(ry)
                    ))
                except (TypeError, ValueError):
                    angulos.append(float('nan'))
            df['quadril_inclinacao_horizontal_graus'] = angulos

        return df

    def convert_data_to_statistic_pandas(self):
        """
        Estatísticas completas por coluna temporal do quadril: média, mediana,
        std, min, max, amplitude e IQR.

        Returns:
            pandas.DataFrame: DataFrame estatístico de uma linha.
        """
        df_temp = self._to_dataframe()
        if df_temp is None:
            return pd.DataFrame()

        df_feat = self._build_features(df_temp)

        estatisticas_linha = [self.person_name, self.repetition]
        nomes_colunas = ['voluntario', 'repeticao']

        for coluna in df_feat.columns:
            dados = pd.to_numeric(df_feat[coluna], errors='coerce')

            media = dados.mean()
            mediana = dados.median()
            desvio_padrao = dados.std()
            minimo = dados.min()
            maximo = dados.max()
            amplitude = maximo - minimo
            iqr = dados.quantile(0.75) - dados.quantile(0.25)

            estatisticas_linha.extend([media, mediana, desvio_padrao, minimo, maximo, amplitude, iqr])

            prefixo = str(coluna)
            nomes_colunas.extend([
                f'{prefixo}_media',
                f'{prefixo}_mediana',
                f'{prefixo}_std',
                f'{prefixo}_min',
                f'{prefixo}_max',
                f'{prefixo}_amplitude',
                f'{prefixo}_iqr',
            ])

        return pd.DataFrame([estatisticas_linha], columns=nomes_colunas)
