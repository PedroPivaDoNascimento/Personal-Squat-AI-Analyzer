"""
Escritor de dados de PÉ no plano FRONTAL (task (3)NEW-DATA_ANALYTICS-SHEETS.md, Fase 2.2).

Refatoração/migração da implementação legada `FootDataExcelWriter` para o pacote
`segments_writer`, passando a herdar obrigatoriamente de `SegmentsWriter`.

Metodologia específica do plano frontal: avaliação de pronação subtalar dinâmica
e desvios laterais do tornozelo/hálux. As estatísticas por coluna temporal
(média, mediana, std, max, min, amplitude — ou apenas std/IQR no lado esquerdo,
que alimenta os descritores do modelo ML de classificação de pisada) foram
preservadas rigorosamente do writer original para não alterar a inferência dos
modelos `modelo_pe_frontal_direito.pkl` / `modelo_pe_frontal_esquerdo.pkl`.
"""
import pandas as pd

from classes.excel.segments_writer.segments_writer import SegmentsWriter


class FrontalFootDataExcelWriter(SegmentsWriter):
    """
    Gera as planilhas de dados brutos e estatísticos do segmento `pe`
    no plano `frontal`.

    Args (herdados de SegmentsWriter):
        repetition (int): A repetição do agachamento.
        segment_repeat_data (pandas.DataFrame | list[dict]): Dados de repetição de pé.
        person_name (str): O nome da pessoa.
        plane_folder_name (str): 'frontal'.
        side (str): 'direito' ou 'esquerdo'.
    """

    @property
    def segment_name(self):
        return 'pe'

    def convert_data_to_statistic_pandas(self):
        """
        Calcula estatísticas (média, mediana, desvio padrão, IQR, etc.) para
        cada coluna temporal do pé e organiza tudo em uma única linha.

        Metodologia do plano frontal (pronação/desvios laterais):
        - Lado direito: média, mediana, std, max, min, amplitude por coluna.
        - Lado esquerdo: std e IQR por coluna (descritores de entrada do
          modelo ML de pronação, preservando exatamente o layout legado).

        Returns:
            pandas.DataFrame: DataFrame estatístico de uma linha.
        """
        df_temp = self._to_dataframe()
        if df_temp is None:
            return pd.DataFrame()

        # Lista para armazenar os valores calculados e os nomes das colunas
        estatisticas_linha = [self.person_name, self.repetition]
        nomes_colunas = ['voluntario', 'repeticao']

        # Iterar por cada coluna original (ankle_x, ankle_y, etc.)
        for coluna in df_temp.columns:
            dados = df_temp[coluna]

            # Cálculos estatísticos
            desvio_padrao = dados.std()
            iqr = dados.quantile(0.75) - dados.quantile(0.25)
            media = dados.mean()
            mediana = dados.median()
            maximo = dados.max()
            minimo = dados.min()
            amplitude = maximo - minimo

            if self.side == 'esquerdo':
                # Adiciona os valores à nossa lista de dados
                estatisticas_linha.extend([desvio_padrao, iqr])

                # Cria rótulos descritivos para facilitar a leitura no Excel
                prefixo = str(coluna)
                nomes_colunas.extend([
                    f'{prefixo}_std',
                    f'{prefixo}_iqr'
                ])
            else:
                # Adiciona os valores à nossa lista de dados
                estatisticas_linha.extend([media, mediana, desvio_padrao, maximo, minimo, amplitude])

                # Cria rótulos descritivos para facilitar a leitura no Excel
                prefixo = str(coluna)
                nomes_colunas.extend([
                    f'{prefixo}_media',
                    f'{prefixo}_mediana',
                    f'{prefixo}_std',
                    f'{prefixo}_max',
                    f'{prefixo}_min',
                    f'{prefixo}_amplitude'
                ])

        # 4. Cria o DataFrame final com uma única linha e colunas rotuladas
        return pd.DataFrame([estatisticas_linha], columns=nomes_colunas)

    # ------------------------------------------------------------------ #
    # Aliases de compatibilidade com a API do writer legado              #
    # ------------------------------------------------------------------ #
    def write_raw_foot_data(self):
        """Alias legado para `write_raw_data()` (transição suave)."""
        return self.write_raw_data()

    def write_statistic_foot_data(self):
        """Alias legado para `write_statistic_data()` (transição suave)."""
        return self.write_statistic_data()
