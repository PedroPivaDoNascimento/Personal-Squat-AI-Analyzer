"""
Escritor de dados de CABEÇA (task (3)NEW-DATA_ANALYTICS-SHEETS.md, Fase 2.8).

Segmento `cabeca`: nariz, olhos e orelhas (landmarks 0 nariz, 2/5 olhos, 7/8 orelhas).
Coleta coordenadas normalizadas (X, Y, Z) e calcula métricas posturais da cabeça
e descritores estatísticos da repetição.
"""
import pandas as pd

from classes.excel.segments_writer.segments_writer import SegmentsWriter
from classes.vector_calculator import VectorCalculator


class HeadDataExcelWriter(SegmentsWriter):
    """Gera as planilhas de dados brutos e estatísticos do segmento `cabeca`."""

    @property
    def segment_name(self):
        return 'cabeca'

    def _build_features(self, df_temp):
        """
        Deriva métricas de alinhamento cefálico frame a frame (ex: ângulo
        olho-orelha ou nariz-orelha relativo à horizontal).
        """
        df = df_temp.copy()

        def find_col(*candidates):
            for cand in candidates:
                for col in df.columns:
                    if col == cand or col.endswith(cand):
                        return col
            return None

        ear_x = find_col('ear_x')
        ear_y = find_col('ear_y')
        eye_x = find_col('eye_x')
        eye_y = find_col('eye_y')
        nose_x = find_col('nose_x')
        nose_y = find_col('nose_y')

        ref_x = eye_x or nose_x
        ref_y = eye_y or nose_y

        if ear_x and ear_y and ref_x and ref_y:
            angulos = []
            for ex, ey, rx, ry in zip(df[ear_x], df[ear_y], df[ref_x], df[ref_y]):
                try:
                    angulos.append(VectorCalculator.angle_to_horizontal(
                        float(ex), float(ey), float(rx), float(ry)
                    ))
                except (TypeError, ValueError):
                    angulos.append(float('nan'))
            df['cabeca_inclinacao_horizontal_graus'] = angulos

        return df

    def convert_data_to_statistic_pandas(self):
        """
        Calcula estatísticas completas por coluna temporal da cabeça: média, mediana,
        desvio padrão, mínimo, máximo, amplitude e IQR.
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
                media = mediana = desvio_padrao = maximo = minimo = amplitude = iqr = 0.0
            else:
                desvio_padrao = float(dados.std(ddof=0))
                iqr = float(dados.quantile(0.75) - dados.quantile(0.25))
                media = float(dados.mean())
                mediana = float(dados.median())
                maximo = float(dados.max())
                minimo = float(dados.min())
                amplitude = maximo - minimo

            estatisticas_linha.extend([media, mediana, desvio_padrao, maximo, minimo, amplitude, iqr])
            prefixo = str(coluna)
            nomes_colunas.extend([
                f'{prefixo}_media',
                f'{prefixo}_mediana',
                f'{prefixo}_std',
                f'{prefixo}_max',
                f'{prefixo}_min',
                f'{prefixo}_amplitude',
                f'{prefixo}_iqr',
            ])

        return pd.DataFrame([estatisticas_linha], columns=nomes_colunas)
