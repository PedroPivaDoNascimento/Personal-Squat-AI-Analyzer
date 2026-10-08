"""
Gerador de planilhas de dados de segmentos biomecânicos (brutos e estatísticos).

Implementa as Fases 2 e 3 do plano `docs/tasks/(3)NEW-DATAXANALYTICS-SHEETS.md`:

- Fase 2 (`dados brutos/`): série temporal por segmento com colunas
  `frame`, `timestamp_ms`, `segmento`, `landmark_id` e as coordenadas
  tridimensionais normalizadas + visibilidade de cada landmark.
- Fase 3 (`estatistica/`): descritores agregados por segmento/landmark/eixo
  (média, desvio padrão, amplitude [max-min] e IQR), no mesmo padrão de
  "uma linha larga por voluntário/repetição" já consumido pelos modelos
  Scikit-Learn de pronação (`FootDataExcelWriter.convert_data_to_statistic_pandas`).

Os arquivos são gravados em:
    planilhas/<plano>/<lado>/dados_do_segmento/dados brutos/dados_segmento_<voluntario>_rep<N>.xlsx
    planilhas/<plano>/<lado>/dados_do_segmento/estatistica/dados_segmento_<voluntario>_rep<N>.xlsx

Responsabilidade única: apenas serializar em Excel os frames já coletados
pelas camadas de análise; toda a orquestração de quando salvar pertence à
camada de serviço (`SegmentDataService`), conforme docs/RULES.md.
"""
import logging
import os
import re

import pandas as pd

from classes.excel.set_folders import SetFolders

logger = logging.getLogger(__name__)


class SegmentDataExcelWriter:
    """Escreve as planilhas de dados brutos e estatísticos dos segmentos."""

    def __init__(self, person_name, plane_folder_name, side, repetition):
        """
        Args:
            person_name (str): Nome do voluntário.
            plane_folder_name (str): 'frontal' ou 'sagital'.
            side (str): Lado do corpo ('direito'/'esquerdo').
            repetition (int): Número da repetição concluída (1, 2 ou 3).
        """
        self.person_name = person_name
        self.plane_folder_name = plane_folder_name
        self.side = side
        self.repetition = repetition

    # ------------------------------------------------------------------ #
    # Construção dos DataFrames (Fases 2.2 e 3.1)
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_raw_dataframe(segment_frames):
        """
        Estrutura o DataFrame bruto relacionando `frame`, `timestamp_ms`,
        `segmento`, `landmark_id` e coordenadas tridimensionais + visibilidade
        (tarefa 2.2).

        Args:
            segment_frames (list[dict]): um dicionário por frame no formato
                {'frame': int, 'timestamp_ms': float, 'segments':
                 {segmento: {landmark: {'x','y','z','visibility'}}}}.

        Returns:
            pandas.DataFrame: uma linha por frame/segmento/landmark (vazio
            quando não há dados).
        """
        rows = []
        for frame_entry in segment_frames or []:
            segments = frame_entry.get('segments') or {}
            for segmento, landmarks in segments.items():
                for landmark_name, coords in (landmarks or {}).items():
                    rows.append({
                        'frame': frame_entry.get('frame'),
                        'timestamp_ms': round(float(frame_entry.get('timestamp_ms', 0)), 2),
                        'segmento': segmento,
                        'landmark_id': landmark_name,
                        'x': coords.get('x'),
                        'y': coords.get('y'),
                        'z': coords.get('z'),
                        'visibility': coords.get('visibility'),
                    })
        return pd.DataFrame(rows)

    def build_statistic_dataframe(self, df_raw):
        """
        Calcula os descritores estatísticos agregados por segmento/landmark/eixo
        (média, desvio padrão, amplitude e IQR - tarefas 3.1/3.2) e consolida
        tudo em uma única linha larga identificada por voluntário e repetição,
        no padrão compatível com os modelos Scikit-Learn de pronação.

        Args:
            df_raw (pandas.DataFrame): DataFrame bruto de `build_raw_dataframe`.

        Returns:
            pandas.DataFrame: uma linha de estatísticas (vazio se sem dados).
        """
        if df_raw is None or df_raw.empty:
            return pd.DataFrame()

        valores_linha = [self.person_name, self.repetition]
        nomes_colunas = ['voluntario', 'repeticao']

        # Agrupa por (segmento, landmark) preservando a ordem de aparição.
        grupos = df_raw.groupby(['segmento', 'landmark_id'], sort=False)
        for (segmento, landmark), sub_df in grupos:
            prefixo = f'{segmento}_{landmark}'
            for eixo in ('x', 'y', 'z', 'visibility'):
                dados = pd.to_numeric(sub_df[eixo], errors='coerce').dropna()
                if dados.empty:
                    continue
                media = dados.mean()
                desvio_padrao = dados.std()
                minimo = dados.min()
                maximo = dados.max()
                amplitude = maximo - minimo
                iqr = dados.quantile(0.75) - dados.quantile(0.25)

                valores_linha.extend([media, desvio_padrao, amplitude, iqr])
                nomes_colunas.extend([
                    f'{prefixo}_{eixo}_media',
                    f'{prefixo}_{eixo}_std',
                    f'{prefixo}_{eixo}_amplitude',
                    f'{prefixo}_{eixo}_iqr',
                ])

        return pd.DataFrame([valores_linha], columns=nomes_colunas)

    # ------------------------------------------------------------------ #
    # Persistência em Excel (Fases 2.3 e 3.3)
    # ------------------------------------------------------------------ #
    def _build_file_name(self):
        """
        Monta o nome descritivo do arquivo contendo o voluntário e a
        repetição (tarefa 4.2), no mesmo padrão de `FootDataExcelWriter`.

        Returns:
            str: `dados_segmento_<voluntario>_rep<N>.xlsx`.
        """
        person_slug = re.sub(r'[^\w\-]+', '_', str(self.person_name).strip())
        return f'dados_segmento_{person_slug}_rep{self.repetition}.xlsx'

    def _write_accumulative_sheet(self, df_novo, folder_path):
        """
        Grava (acumulando, sem sobrescrever linhas anteriores) um DataFrame
        em `folder_path`, sempre dentro de try/except para que falhas de
        escrita nunca interrompam o pipeline principal (tarefa 4.1).

        Args:
            df_novo (pandas.DataFrame): dados a acrescentar.
            folder_path (str): diretório de destino (já criado por SetFolders).

        Returns:
            bool: True se o arquivo foi escrito com sucesso.
        """
        if df_novo is None or df_novo.empty:
            logger.info("Sem dados de segmento para salvar em '%s'.", folder_path)
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
            logger.info("✅ Dados de segmento salvos em: %s", file_path)
            return True
        except Exception as exc:  # noqa: BLE001 - resiliência exigida (4.1)
            logger.error("❌ ERRO AO ESCREVER PLANILHA DE SEGMENTO (%s): %s", file_path, exc)
            return False

    def write_raw_segment_data(self, df_raw):
        """
        Salva a série temporal bruta na pasta
        `planilhas/<plano>/<lado>/dados_do_segmento/dados brutos/` (tarefa 2.3).
        """
        folders = SetFolders(self.person_name, self.plane_folder_name, self.side)
        return self._write_accumulative_sheet(df_raw, folders.get_segment_raw_data_folder())

    def write_statistic_segment_data(self, df_stat):
        """
        Salva a planilha consolidada de estatísticas na pasta
        `planilhas/<plano>/<lado>/dados_do_segmento/estatistica/` (tarefa 3.3).
        """
        folders = SetFolders(self.person_name, self.plane_folder_name, self.side)
        return self._write_accumulative_sheet(df_stat, folders.get_segment_statistic_data_folder())
