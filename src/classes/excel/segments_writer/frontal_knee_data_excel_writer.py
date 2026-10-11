"""
Escritor de dados de JOELHO no plano FRONTAL (task (3)NEW-DATA_ANALYTICS-SHEETS.md, Fase 2.4).

Metodologia específica do plano frontal: alinhamento angular frontal / HKA
(Quadril-Joelho-Tornozelo), valgo/varo dinâmico e colapso medial. A série
temporal bruta é acrescida de features biomecânicas calculadas frame a frame
com o `VectorCalculator` central (produto vetorial/escalar com arctan2, graus
em [-180, +180], conforme docs/RULES.md):

- `hka_angulo_graus`: ângulo orientado H-K-A (sinal indica valgo/varo);
- `colapso_medial_x`: deslocamento horizontal do joelho em relação à linha
  quadril-tornozelo (proxy direto de colapso medial dinâmico).
As estatísticas agregadas por coluna incluem média, mediana, std, min, max,
amplitude e IQR — com destaque para os descritores do ângulo HKA e do colapso.
"""
import pandas as pd

from classes.excel.segments_writer.segments_writer import SegmentsWriter
from classes.vector_calculator import VectorCalculator


class FrontalKneeDataExcelWriter(SegmentsWriter):
    """Gera as planilhas de dados brutos e estatísticos do segmento `joelho` no plano `frontal`."""

    @property
    def segment_name(self):
        return 'joelho'

    def _build_frontal_features(self, df_temp):
        """
        Deriva as colunas biomecânicas frontais do joelho (HKA orientado e
        colapso medial) a partir das coordenadas normalizadas de quadril,
        joelho e tornozelo.

        Args:
            df_temp (pandas.DataFrame): Série temporal bruta do joelho frontal.

        Returns:
            pandas.DataFrame: Série temporal acrescida das colunas derivadas.
        """
        df = df_temp.copy()

        def find_col(*candidates):
            for cand in candidates:
                for col in df.columns:
                    if col == cand or col.endswith(cand):
                        return col
            return None

        hip_x = find_col('hip_x')          # quadril de apoio (referência do lado)
        hip_y = find_col('hip_y')
        knee_x = find_col('knee_x')
        knee_y = find_col('knee_y')
        ankle_x = find_col('ankle_x')
        ankle_y = find_col('ankle_y')

        # Ângulo HKA orientado via VectorCalculator (arctan2, [-180, +180])
        if hip_x and hip_y and knee_x and knee_y and ankle_x and ankle_y:
            hka = []
            for hx, hy, kx, ky, ax, ay in zip(
                df[hip_x], df[hip_y], df[knee_x], df[knee_y], df[ankle_x], df[ankle_y]
            ):
                try:
                    hka.append(VectorCalculator.calculate_angle_3p(
                        float(hx), float(hy), float(kx), float(ky), float(ax), float(ay)
                    ))
                except (TypeError, ValueError):
                    hka.append(float('nan'))
            df['hka_angulo_graus'] = hka

            # Colapso medial: distância horizontal do centro do joelho até a
            # reta vertical de referência quadril->tornozelo (interpolada em X).
            colapso = []
            for hx, hy, kx, ky, ax, ay in zip(
                df[hip_x], df[hip_y], df[knee_x], df[knee_y], df[ankle_x], df[ankle_y]
            ):
                try:
                    hx, hy, kx, ky, ax, ay = map(float, (hx, hy, kx, ky, ax, ay))
                    span = (hy - ay)
                    if abs(span) < 1e-9:
                        ref_x = hx
                    else:
                        # Interpolação linear de X na altura Y do joelho
                        t = (ky - ay) / span
                        ref_x = ax + t * (hx - ax)
                    colapso.append(kx - ref_x)
                except (TypeError, ValueError):
                    colapso.append(float('nan'))
            df['colapso_medial_x'] = colapso

        return df

    def convert_data_to_statistic_pandas(self):
        """
        Estatísticas completas por coluna temporal do joelho frontal (incluindo
        HKA orientado e colapso medial): média, mediana, std, min, max,
        amplitude e IQR.

        Returns:
            pandas.DataFrame: DataFrame estatístico de uma linha.
        """
        df_temp = self._to_dataframe()
        if df_temp is None:
            return pd.DataFrame()

        df_feat = self._build_frontal_features(df_temp)

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
