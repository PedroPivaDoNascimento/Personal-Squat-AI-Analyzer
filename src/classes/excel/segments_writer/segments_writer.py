"""
Classe base abstrata `SegmentsWriter` (task (3)NEW-DATA_ANALYTICS-SHEETS.md, Fase 2.1).

Define o Template Method de persistência em Excel para TODOS os escritores de
dados segmentares (pe, joelho, quadril, tronco, cabeca), nos planos sagital e
frontal, garantindo:

- Estrutura de pastas dinâmica `planilhas/<plano>/<lado>/dados_<segmento>/`
  via `SetFolders(segment_name=...)`;
- Convenção rigorosa de nome de arquivo:
  `dados_<segmento>_<person_slug>_rep<repetition>.xlsx`;
- Serialização flatten padronizada da série temporal bruta
  (`['voluntario', 'repeticao', 'dado_0', ...]`);
- Gravação incremental/persistente com openpyxl + `pd.concat(ignore_index=True)`;
- Exportação completa de todas as repetições concluídas (um arquivo por rep,
  sem omissão ou descarte de ciclos).

Os cálculos estatísticos/biomecânicos são específicos por segmento e plano e
ficam a cargo das subclasses através do método abstrato
`convert_data_to_statistic_pandas()`.
"""
import os
import re
from abc import ABC, abstractmethod

import pandas as pd

from classes.excel.set_folders import SetFolders


class SegmentsWriter(ABC):
    """
    Classe pai abstrata de todos os escritores de dados de segmentos em Excel.

    Args:
        repetition (int): Número da repetição concluída (1, 2, 3...).
        segment_repeat_data (pandas.DataFrame | list[dict]): Dados coletados do
            segmento ao longo do ciclo da repetição (série temporal de frames).
        person_name (str): Nome do voluntário.
        plane_folder_name (str): 'frontal' ou 'sagital'.
        side (str): 'direito' ou 'esquerdo'.
    """

    # Subpastas padrão de cada segmento (Fase 1.3 da task)
    RAW_SUBFOLDER = "dados brutos"
    STATISTIC_SUBFOLDER = "dados estatisticos"

    def __init__(
        self,
        repetition,
        segment_repeat_data=None,
        person_name=None,
        plane_folder_name=None,
        side=None,
        **kwargs,
    ):
        # Suporte a alias de compatibilidade legado (ex: foot_repeat_data)
        if segment_repeat_data is None and 'foot_repeat_data' in kwargs:
            segment_repeat_data = kwargs.pop('foot_repeat_data')

        self.repetition = repetition
        self.segment_repeat_data = segment_repeat_data
        self.person_name = person_name
        self.plane_folder_name = plane_folder_name
        self.side = side

    
    @property
    @abstractmethod
    def segment_name(self):
        """
        Nome identificador da pasta do segmento ('pe', 'joelho', 'quadril',
        'tronco' ou 'cabeca'). Definido obrigatoriamente pela subclasse.
        """
        pass

    
    def _build_set_folders(self):
        """Instancia o `SetFolders` configurado para o segmento específico."""
        return SetFolders(
            person_name=self.person_name,
            plane_folder_name=self.plane_folder_name,
            side=self.side,
            segment_name=self.segment_name,
        )

    def _create_folder_raw_data(self):
        """Cria/resolve o caminho `.../dados_<segmento>/dados brutos/`."""
        path_folder_side = self._build_set_folders().create_folders()
        path_folder_raw_data = os.path.join(path_folder_side, self.RAW_SUBFOLDER)
        os.makedirs(path_folder_raw_data, exist_ok=True)
        return path_folder_raw_data

    def _create_folder_statistic_data(self):
        """Cria/resolve o caminho `.../dados_<segmento>/dados estatisticos/`."""
        path_folder_side = self._build_set_folders().create_folders()
        path_folder_statistic_data = os.path.join(path_folder_side, self.STATISTIC_SUBFOLDER)
        os.makedirs(path_folder_statistic_data, exist_ok=True)
        return path_folder_statistic_data

    def _build_file_name(self):
        """
        Monta o nome do arquivo Excel identificando o usuário (voluntário) e a
        repetição à qual os dados pertencem.

        Mantém rigorosamente a convenção de nome base existente para o segmento:
        `dados_<segmento>_<person_slug>_rep<repeticao>.xlsx`
        (ex: `dados_pe_joao_silva_rep1.xlsx`, `dados_joelho_maria_souza_rep2.xlsx`).

        Returns:
            str: Nome do arquivo .xlsx.
        """
        person_slug = re.sub(r'[^\w\-]+', '_', str(self.person_name).strip())
        return f'dados_{self.segment_name}_{person_slug}_rep{self.repetition}.xlsx'

    def _to_dataframe(self):
        """
        Normaliza `segment_repeat_data` (DataFrame ou lista de dicts) em um
        DataFrame único de séries temporais.

        Returns:
            pandas.DataFrame | None: DataFrame dos frames do segmento, ou None
                se não houver dados utilizáveis.
        """
        if isinstance(self.segment_repeat_data, pd.DataFrame):
            df = self.segment_repeat_data.copy()
        else:
            try:
                df = pd.DataFrame(self.segment_repeat_data)
            except (ValueError, TypeError):
                return None

        if df.empty:
            return None
        return df

    def _convert_data_to_raw_pandas(self):
        """
        Converte os dados brutos em um DataFrame de linha única com colunas
        rotuladas (serialização flatten padronizada de toda a série temporal):
        Coluna 0: voluntario
        Coluna 1: repeticao
        Demais: dado_0, dado_1, ... (achatar linha a linha, preservando a
        ordem temporal original das colunas do segmento).

        Returns:
            pandas.DataFrame: DataFrame de uma linha (vazio se sem dados).
        """
        df_temp = self._to_dataframe()
        if df_temp is None:
            return pd.DataFrame()

        dados_brutos = df_temp.values.flatten().tolist()

        linha_unificada = [self.person_name, self.repetition] + dados_brutos

        nomes_colunas = ['voluntario', 'repeticao'] + [f'dado_{i}' for i in range(len(dados_brutos))]

        return pd.DataFrame([linha_unificada], columns=nomes_colunas)

    # ------------------------------------------------------------------ #
    # Métodos abstratos / especializados por segmento e plano            #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def convert_data_to_statistic_pandas(self):
        """
        Cálculo específico de métricas estatísticas e biomecânicas adequadas ao
        plano do segmento (ex: colunas de pronação no pé frontal vs elevação no
        pé sagital; colapso medial no joelho frontal vs flexão no joelho
        sagital). Implementado obrigatoriamente por cada subclasse.

        Returns:
            pandas.DataFrame: DataFrame estatístico de uma linha.
        """
        pass

   
    def _write_dataframe_incrementally(self, folder_path, df_novo):
        """
        Gravação incremental/persistente com openpyxl + `pd.concat` com
        `ignore_index=True`: se o arquivo da repetição já existir, o novo dado
        é concatenado abaixo do existente; caso contrário, cria-se o arquivo.

        É mandatório exportar sempre TODOS os dados de todas as repetições
        concluídas, gerando o arquivo individual com o sufixo da repetição.

        Args:
            folder_path (str): Diretório destino (dados brutos ou estatísticos).
            df_novo (pandas.DataFrame): Nova linha/lote de dados a persistir.

        Returns:
            bool: True se a gravação foi bem-sucedida.
        """
        if df_novo is None or df_novo.empty:
            print("❌ DataFrame vazio, nada para salvar.")
            return False

        file_path = os.path.join(os.path.normpath(folder_path), self._build_file_name())

        try:
            os.makedirs(folder_path, exist_ok=True)

            if os.path.exists(file_path):
                df_existente = pd.read_excel(file_path, engine='openpyxl')
                df_final = pd.concat([df_existente, df_novo], ignore_index=True)
            else:
                df_final = df_novo

            df_final.to_excel(file_path, index=False, engine='openpyxl')
            print(f"✅ Dados salvos com sucesso em: {file_path}")
            return True

        except Exception as e:
            print(f"❌ ERRO AO ESCREVER EXCEL: {e}")
            return False

    # ------------------------------------------------------------------ #
    # Pontos de entrada de exportação                                    #
    # ------------------------------------------------------------------ #
    def write_raw_data(self):
        """Exporta e salva os dados brutos da repetição do segmento em Excel."""
        return self._write_dataframe_incrementally(
            self._create_folder_raw_data(),
            self._convert_data_to_raw_pandas(),
        )

    def write_statistic_data(self):
        """Exporta e salva os dados estatísticos da repetição do segmento em Excel."""
        return self._write_dataframe_incrementally(
            self._create_folder_statistic_data(),
            self.convert_data_to_statistic_pandas(),
        )

    def write_all(self):
        """
        Ponto de entrada agregado: exporta obrigatoriamente TODOS os dados
        (brutos + estatísticos) da repetição concluída.

        Returns:
            bool: True se ambas as exportações foram bem-sucedidas.
        """
        raw_ok = self.write_raw_data()
        stat_ok = self.write_statistic_data()
        return raw_ok and stat_ok
