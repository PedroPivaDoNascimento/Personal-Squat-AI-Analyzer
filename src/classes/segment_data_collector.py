"""
Coletor frame a frame das séries temporais dos segmentos biomecânicos.

Implementa a tarefa 2.1 de `docs/tasks/(3)NEW-DATAXANALYTICS-SHEETS.md`:
extrai as coordenadas (X, Y, Z) e a visibilidade dos landmarks MediaPipe de
cada segmento monitorado (cabeça, tronco, quadril, joelho e pé) durante a
execução do vídeo, organizando os frames por repetição (recorte "auto-curado"
entre os contadores cumulativos de fim de ciclo da máquina de estados).

Responsabilidade única: coleta em memória. A gravação em Excel é delegada à
camada de serviço (`SegmentDataService`) via callback ao fim de cada
repetição, mantendo o padrão já estabelecido para os dados de pé
(`on_repetition_completed` em `BaseFrontal`) e evitando I/O dentro dos
analisadores (docs/RULES.md, Seção 2.1).
"""
import logging

from classes.excel.segment_data_catalog import get_segment_catalog

logger = logging.getLogger(__name__)


class SegmentDataCollector:
    """Acumula os landmarks segmentados de cada frame, separados por repetição."""

    def __init__(self, plane_folder_name):
        """
        Args:
            plane_folder_name (str): 'frontal' ou 'sagital' - define o catálogo
                de segmentos extraídos (joelho frontal vs. sagital, etc.).
        """
        self.plane_folder_name = plane_folder_name
        self.catalog = get_segment_catalog(plane_folder_name)

        # Frames da repetição em curso (buffer corrente).
        self.current_frames = []
        # Contador cumulativo de frames ao fim de cada repetição concluída.
        self.repetition_frame_counts = []

    def collect_frame(self, landmarks_obj, timestamp_ms, frame_number):
        """
        Extrai X/Y/Z + visibilidade dos landmarks de todos os segmentos do
        catálogo para o frame informado (tarefa 2.1).

        Nunca lança exceção para fora: falhas de landmark ausente apenas
        resultam em nenhum frame adicionado neste ciclo (resiliência 4.1).

        Args:
            landmarks_obj: lista de landmarks do MediaPipe (ou None).
            timestamp_ms (float): timestamp do frame em milissegundos.
            frame_number (int): número sequencial do frame no vídeo.
        """
        if not self.catalog or landmarks_obj is None:
            return

        try:
            segments_payload = {}
            for segment_key, spec in self.catalog.items():
                landmarks_payload = {}
                for lm_spec in spec['landmarks']:
                    lm = landmarks_obj[lm_spec['index']]
                    axes = set(lm_spec['axes'])
                    entry = {'visibility': getattr(lm, 'visibility', None)}
                    if 'x' in axes:
                        entry['x'] = lm.x
                    if 'y' in axes:
                        entry['y'] = lm.y
                    if 'z' in axes:
                        entry['z'] = lm.z
                    landmarks_payload[lm_spec['name']] = entry
                segments_payload[segment_key] = landmarks_payload

            self.current_frames.append({
                'frame': int(frame_number),
                'timestamp_ms': float(timestamp_ms),
                'segments': segments_payload,
            })
        except Exception as exc:  # noqa: BLE001 - coleta nunca derruba o pipeline
            logger.warning("Falha ao coletar landmarks de segmento no frame %s: %s",
                           frame_number, exc)

    def mark_repetition_end(self):
        """
        Registra o fim de uma repetição: fecha o buffer corrente, devolve os
        frames exatos dela (auto-curação) e zera o acumulador para o próximo
        ciclo.

        Returns:
            list[dict]: frames da repetição recém-concluída.
        """
        frames = list(self.current_frames)
        self.repetition_frame_counts.append(
            (self.repetition_frame_counts[-1] if self.repetition_frame_counts else 0)
            + len(frames)
        )
        self.current_frames = []
        return frames
