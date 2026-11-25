"""
Serviço de persistência de dados de TODOS os segmentos corporais em Excel.

Substitui o FootDataService de escopo único, orquestrando a gravação de brutos
e estatísticos para cada segmento detectado na repetição concluída:

  Plano frontal : pé  (FrontalFootDataExcelWriter)
                  joelho (FrontalKneeDataExcelWriter)
                  quadril (HipDataExcelWriter)

  Plano sagital : pé  (SagitalFootDataExcelWriter)
                  joelho (SagitalKneeDataExcelWriter)
                  tronco (TrunkDataExcelWriter)
                  cabeça (HeadDataExcelWriter)

Cada writer herda de SegmentsWriter e expõe `write_all()`, que persiste os dados
brutos e estatísticos seguindo rigorosamente a mesma estrutura do pé frontal
(pasta, convenção de nome e formato de planilha).
"""
import logging
import os

from src.classes.excel.segments_writer.frontal_foot_data_excel_writer import FrontalFootDataExcelWriter
from src.classes.excel.segments_writer.frontal_knee_data_excel_writer import FrontalKneeDataExcelWriter
from src.classes.excel.segments_writer.hip_data_excel_writer import HipDataExcelWriter
from src.classes.excel.segments_writer.sagital_foot_data_excel_writer import SagitalFootDataExcelWriter
from src.classes.excel.segments_writer.sagital_knee_data_excel_writer import SagitalKneeDataExcelWriter
from src.classes.excel.segments_writer.trunk_data_excel_writer import TrunkDataExcelWriter
from src.classes.excel.segments_writer.head_data_excel_writer import HeadDataExcelWriter

logger = logging.getLogger(__name__)


class SegmentsDataService:
    """
    Serviço responsável por salvar os dados brutos e estatísticos de TODOS os
    segmentos corporais de uma repetição concluída em planilhas Excel.

    A cada repetição o analisador chama `on_repetition_completed(rep_num, segment_data)`;
    este serviço recebe esse sinal e delega a gravação para cada writer específico,
    mantendo a camada de modelo livre de efeitos colaterais de I/O.
    """

    # ---------------------------------------------------------------------- #
    # Ponto de entrada — Frontal                                             #
    # ---------------------------------------------------------------------- #
    def save_frontal_repetition(self, analyzer, repetition_number, segment_data):
        """
        Salva dados brutos e estatísticos dos segmentos frontais da repetição.

        Args:
            analyzer: Instância do analisador frontal (fornece `person_name` e `side`).
            repetition_number (int): Número da repetição concluída.
            segment_data (dict): Chaves 'foot', 'knee', 'hip' com listas de frames.

        Returns:
            bool: True se todos os segmentos com dados foram gravados com sucesso.
        """
        person_name = analyzer.person_name
        side = analyzer.side

        writers = []

        foot_frames = segment_data.get('foot') or []
        if foot_frames:
            writers.append(FrontalFootDataExcelWriter(
                repetition=repetition_number,
                segment_repeat_data=foot_frames,
                person_name=person_name,
                plane_folder_name='frontal',
                side=side,
            ))

        knee_frames = segment_data.get('knee') or []
        if knee_frames:
            writers.append(FrontalKneeDataExcelWriter(
                repetition=repetition_number,
                segment_repeat_data=knee_frames,
                person_name=person_name,
                plane_folder_name='frontal',
                side=side,
            ))

        hip_frames = segment_data.get('hip') or []
        if hip_frames:
            writers.append(HipDataExcelWriter(
                repetition=repetition_number,
                segment_repeat_data=hip_frames,
                person_name=person_name,
                plane_folder_name='frontal',
                side=side,
            ))

        return self._run_writers(writers, repetition_number, 'frontal')

    # ---------------------------------------------------------------------- #
    # Ponto de entrada — Sagital                                             #
    # ---------------------------------------------------------------------- #
    def save_sagittal_repetition(self, analyzer, repetition_number, segment_data):
        """
        Salva dados brutos e estatísticos dos segmentos sagitais da repetição.

        Args:
            analyzer: Instância do analisador sagital (fornece `person_name` e `side`).
            repetition_number (int): Número da repetição concluída.
            segment_data (dict): Chaves 'foot', 'knee', 'trunk', 'head' com listas de frames.

        Returns:
            bool: True se todos os segmentos com dados foram gravados com sucesso.
        """
        person_name = analyzer.person_name
        side = analyzer.side

        writers = []

        foot_frames = segment_data.get('foot') or []
        if foot_frames:
            writers.append(SagitalFootDataExcelWriter(
                repetition=repetition_number,
                segment_repeat_data=foot_frames,
                person_name=person_name,
                plane_folder_name='sagital',
                side=side,
            ))

        knee_frames = segment_data.get('knee') or []
        if knee_frames:
            writers.append(SagitalKneeDataExcelWriter(
                repetition=repetition_number,
                segment_repeat_data=knee_frames,
                person_name=person_name,
                plane_folder_name='sagital',
                side=side,
            ))

        trunk_frames = segment_data.get('trunk') or []
        if trunk_frames:
            writers.append(TrunkDataExcelWriter(
                repetition=repetition_number,
                segment_repeat_data=trunk_frames,
                person_name=person_name,
                plane_folder_name='sagital',
                side=side,
            ))

        head_frames = segment_data.get('head') or []
        if head_frames:
            writers.append(HeadDataExcelWriter(
                repetition=repetition_number,
                segment_repeat_data=head_frames,
                person_name=person_name,
                plane_folder_name='sagital',
                side=side,
            ))

        return self._run_writers(writers, repetition_number, 'sagital')

    # ---------------------------------------------------------------------- #
    # Helpers internos                                                       #
    # ---------------------------------------------------------------------- #
    def _run_writers(self, writers, repetition_number, plane):
        """Executa `write_all()` em cada writer e agrega o resultado."""
        if not writers:
            logger.warning(
                "Repetição %s (%s): nenhum segmento com dados para salvar.",
                repetition_number,
                plane,
            )
            return False

        all_ok = True
        for writer in writers:
            try:
                ok = writer.write_all()
                if not ok:
                    logger.warning(
                        "Repetição %s (%s): writer %s retornou False.",
                        repetition_number,
                        plane,
                        type(writer).__name__,
                    )
                    all_ok = False
            except OSError as exc:
                logger.error(
                    "Repetição %s (%s): falha de I/O no writer %s — %s",
                    repetition_number,
                    plane,
                    type(writer).__name__,
                    exc,
                )
                all_ok = False

        return all_ok
