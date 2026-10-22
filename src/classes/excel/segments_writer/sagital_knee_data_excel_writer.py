"""
Escritor de dados de JOELHO no plano SAGITAL (task (3)NEW-DATA_ANALYTICS-SHEETS.md, Fase 2.5).

Metodologia específica do plano sagital: grau de flexão máxima, velocidade
angular de descida e projeção do joelho sobre a ponta do pé (avanço
anteroposterior), além da profundidade do agachamento. Features biomecânicas
derivadas frame a frame com o `VectorCalculator` central (arctan2 orientado,
graus em [-180, +180], conforme docs/RULES.md):

- `flexao_angulo_graus`: ângulo articular quadril-joelho-tornozelo (flexão);
- `velocidade_angular_graus_s`: derivada temporal do ângulo de flexão;
- `avanco_knee_to_toe_x`: avanço horizontal do joelho além do hálux;
- `profundidade_delta_y`: descenso vertical do joelho relativo ao frame inicial.
"""
import pandas as pd

from classes.excel.segments_writer.segments_writer import SegmentsWriter
from classes.vector_calculator import VectorCalculator


class SagitalKneeDataExcelWriter(SegmentsWriter):
    """Gera as planilhas de dados brutos e estatísticos do segmento `joelho` no plano `sagital`."""

    @property
    def segment_name(self):
        return 'joelho'

    def _build_sagital_features(self, df_temp):
        """
        Deriva as colunas biomecânicas sagitais do joelho (flexão angular,
        velocidade angular, avanço sobre o pé e profundidade).

        Args:
            df_temp (pandas.DataFrame): Série temporal bruta do joelho sagital.

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

        hip_x = find_col('hip_x')
        hip_y = find_col('hip_y')
        knee_x = find_col('knee_x')
        knee_y = find_col('knee_y')
        ankle_x = find_col('ankle_x')
        ankle_y = find_col('ankle_y')
        toe_x = find_col('big_toe_x', 'toe_x')

        # Coluna de tempo (ms) para cálculo de velocidade angular; se ausente,
        # assume cadência padrão de 30 FPS (~33.33 ms por frame).
        time_col = find_col('timestamp_ms', 'tempo_ms', 'time_ms')

        if hip_x and hip_y and knee_x and knee_y and ankle_x and ankle_y:
            flexao = []
            for hx, hy, kx, ky, ax, ay in zip(
                df[hip_x], df[hip_y], df[knee_x], df[knee_y], df[ankle_x], df[ankle_y]
            ):
                try:
                    # Ângulo interno no vértice do joelho (produto vetorial/
                    # escalar com arctan2 no VectorCalculator central).
                    ang = abs(VectorCalculator.calculate_angle_3p(
                        float(hx), float(hy), float(kx), float(ky), float(ax), float(ay)
                    ))
                    if ang > 90.0:
                        ang = 180.0 - ang  # suplementar => grau de flexão real
                    flexao.append(ang)
                except (TypeError, ValueError):
                    flexao.append(float('nan'))
            df['flexao_angulo_graus'] = flexao

            # Velocidade angular de descida (graus/s) por diferença finita.
            if time_col:
                t = pd.to_numeric(df[time_col], errors='coerce') / 1000.0
            else:
                t = pd.Series(range(len(df))) * (1.0 / 30.0)
            angle_series = pd.Series(df['flexao_angulo_graus'], index=df.index)
            dt = t.diff()
            dangle = angle_series.diff()
            df['velocidade_angular_graus_s'] = (dangle / dt.replace(0, pd.NA)).astype(float)

        # Avanço do joelho além da ponta do pé (projeção anteroposterior).
        if knee_x and toe_x:
            df['avanco_knee_to_toe_x'] = pd.to_numeric(df[knee_x], errors='coerce') - \
                pd.to_numeric(df[toe_x], errors='coerce')

        # Profundidade: descenso vertical do joelho relativo ao frame inicial.
        if knee_y:
            df['profundidade_delta_y'] = pd.to_numeric(df[knee_y], errors='coerce') - \
                pd.to_numeric(df[knee_y], errors='coerce').iloc[0]

        return df

    def convert_data_to_statistic_pandas(self):
        """
        Estatísticas completas por coluna temporal do joelho sagital (incluindo
        flexão máxima, velocidade angular e avanço sobre o pé): média, mediana,
        std, min, max, amplitude e IQR.

        Returns:
            pandas.DataFrame: DataFrame estatístico de uma linha.
        """
        df_temp = self._to_dataframe()
        if df_temp is None:
            return pd.DataFrame()

        df_feat = self._build_sagital_features(df_temp)

        estatisticas_linha = [self.person_name, self.repetition]
        nomes_colunas = ['voluntario', 'repeticao']

        for coluna in df_feat.columns:
            dados = pd.to_numeric(df_feat[coluna], errors='coerce')

            media = dados.mean()
            mediana = dados.median()
            desvio_padrao = dados.std()
            p5 = dados.quantile(0.05)
            p95 = dados.quantile(0.95)
            amplitude = p95 - p5
            iqr = dados.quantile(0.75) - dados.quantile(0.25)

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
