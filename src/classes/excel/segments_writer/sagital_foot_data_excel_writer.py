"""
Escritor de dados de PÉ no plano SAGITAL (task (3)NEW-DATA_ANALYTICS-SHEETS.md, Fase 2.3).

Metodologia específica do plano sagital: dorsiflexão de tornozelo e
elevação/descolamento do calcanhar do solo. As variáveis temporais coletadas
(tornozelo X/Y, calcanhar X/Y, hálux X/Y) recebem estatísticas descritivas
completas por coluna (média, mediana, std, min, max, amplitude), além de
derivadas biomecânicas sagitais calculadas frame a frame antes da agregação:

- `dorsiflexao_angulo_graus`: ângulo tornozelo-hálux relativo à vertical;
- `elevacao_calcanhar_delta_y`: deslocamento vertical do calcanhar em relação
  ao primeiro frame da repetição (positivo = calcanhar elevado);
- `projecao_antrop_x`: avanço anteroposterior do tornozelo em relação ao
  calcanhar (projeção AP do pé sobre o solo).
"""
import math

import pandas as pd

from classes.excel.segments_writer.segments_writer import SegmentsWriter


class SagitalFootDataExcelWriter(SegmentsWriter):
    """Gera as planilhas de dados brutos e estatísticos do segmento `pe` no plano `sagital`."""

    @property
    def segment_name(self):
        return 'pe'

    def _build_sagital_features(self, df_temp):
        """
        Deriva as colunas biomecânicas sagitais a partir das coordenadas
        normalizadas dos landmarks do pé (tornozelo, calcanhar e hálux).

        Args:
            df_temp (pandas.DataFrame): Série temporal bruta do pé sagital.

        Returns:
            pandas.DataFrame: Série temporal acrescida das colunas derivadas.
        """
        df = df_temp.copy()

        # Localiza as colunas de interesse com nomes flexíveis
        # (left_/right_ prefixados ou não pelo analisador).
        def find_col(*candidates):
            for cand in candidates:
                for col in df.columns:
                    if col == cand or col.endswith(cand):
                        return col
            return None

        ankle_x = find_col('ankle_x')
        ankle_y = find_col('ankle_y')
        heel_x = find_col('heel_x')
        heel_y = find_col('heel_y')
        toe_x = find_col('big_toe_x', 'toe_x')
        toe_y = find_col('big_toe_y', 'toe_y')

        # Dorsiflexão: ângulo entre o vetor tornozelo->hálux e a vertical
        # (referência do eixo da perna tornozelo->joelho aproximada pela
        # própria vertical da imagem no plano sagital). Cálculo orientado
        # com atan2 retornando graus em [-180, +180], conforme docs/RULES.md.
        if ankle_x and ankle_y and toe_x and toe_y:
            dx = pd.to_numeric(df[toe_x], errors='coerce') - pd.to_numeric(df[ankle_x], errors='coerce')
            dy = pd.to_numeric(df[toe_y], errors='coerce') - pd.to_numeric(df[ankle_y], errors='coerce')
            df['dorsiflexao_angulo_graus'] = pd.Series(
                [math.degrees(math.atan2(dy_val, dx_val)) if pd.notnull(dx_val) and pd.notnull(dy_val)
                 else float('nan')
                 for dx_val, dy_val in zip(dx, dy)],
                index=df.index,
            )

        # Elevação do calcanhar: variação de Y em relação ao frame inicial
        # (em imagens, Y cresce para baixo; logo subida => delta negativo
        # invertido aqui para positivo = elevação).
        if heel_y:
            df['elevacao_calcanhar_delta_y'] = df[heel_y].iloc[0] - df[heel_y]

        # Projeção anteroposterior: avanço do tornozelo sobre o calcanhar.
        if ankle_x and heel_x:
            df['projecao_antrop_x'] = df[ankle_x] - df[heel_x]

        return df

    def convert_data_to_statistic_pandas(self):
        """
        Estatísticas completas por coluna temporal (incluindo as derivadas
        biomecânicas sagitais): média, mediana, std, min, max e amplitude.

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
