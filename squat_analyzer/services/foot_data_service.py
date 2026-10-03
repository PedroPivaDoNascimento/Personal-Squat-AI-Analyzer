"""
Serviço de persistência dos dados de pé (brutos e estatísticos) - Camada de Service
Encapsula a lógica de gravação das planilhas `dados_pe.xlsx` no diretório
`planilhas/<plano>/<lado>/dados_pe/dados brutos|dados estatisticos/`.

Responsabilidade única: orquestrar a escrita dos DataFrames produzidos pela
análise frontal nas planilhas acumulativas, sem depender de frameworks web.
Aplica o princípio de Responsabilidade Única (SOLID), conforme docs/RULES.md.
"""
import logging
import os

from src.classes.excel.foot_data_excel_writer import FootDataExcelWriter
from src.classes.excel.set_folders import SetFolders

logger = logging.getLogger(__name__)


class FootDataService:
    """
    Serviço responsável por salvar os dados de repetição de pé em Excel.

    A classe analisadora (`_check_foot_pronation_error`) apenas calcula o
    status de pronação usando o modelo ML; a gravação em disco é delegada
    a este serviço, mantendo a camada de modelo livre de efeitos colaterais
    de I/O não intencionais.
    """

    def save_repetition_foot_data(self, analyzer, repetition_number, frames=None):
        """
        Salva os dados brutos e estatísticos da repetição concluída.

        Os frames de pé são recortados exatamente para a repetição informada
        (auto-curação), usando o contador cumulativo do analisador; se o
        recorte não estiver disponível, usa o buffer completo como fallback.

        Args:
            analyzer: Instância do analisador frontal (possui `foot_repeat_data`,
                `person_name` e `side`).
            repetition_number (int): Número da repetição recém-concluída.
            frames (list, opcional): Frames de pé exatos da repetição (usado
                pelo callback de tempo real; se omitido, recorta do histórico).

        Returns:
            bool: True se ambos os arquivos foram gravados com sucesso.
        """
        if frames is None:
            frames = []
            if hasattr(analyzer, "_get_foot_frames_for_repetition"):
                frames = analyzer._get_foot_frames_for_repetition(repetition_number)
            if not frames:
                frames = list(getattr(analyzer, "foot_repeat_data", []) or [])

        if not frames:
            logger.warning(
                "Repetição %s finalizada sem dados de pé acumulados; nada a salvar.",
                repetition_number,
            )
            return False

        writer = FootDataExcelWriter(
            repetition=repetition_number,
            foot_repeat_data=frames,
            person_name=analyzer.person_name,
            plane_folder_name="frontal",
            side=analyzer.side,
        )

        try:
            writer.write_raw_foot_data()
            writer.write_statistic_foot_data()
        except OSError as exc:
            logger.error(
                "Falha de I/O ao salvar dados de pé (repetição %s): %s",
                repetition_number,
                exc,
            )
            return False

        logger.info(
            "Dados de pé da repetição %s salvos em: %s",
            repetition_number,
            os.path.abspath(os.path.join('planilhas', 'frontal', str(analyzer.side).lower(), 'dados_pe')),
        )
        return True

    def get_foot_data_dir(self, person_name, side):
        """
        Retorna (e garante a existência de) onde os arquivos `dados_pe.xlsx`
        são gravados, para fins de rastreio/auditoria.

        Args:
            person_name (str): Nome da pessoa (cria subpasta de mesmo nome).
            side (str): Lado do corpo ('direito' ou 'esquerdo').

        Returns:
            str: Caminho relativo raiz do projeto até a pasta `dados_pe`.
        """
        set_folders = SetFolders(
            person_name=person_name,
            plane_folder_name="frontal",
            side=side,
        )
        return os.path.normpath(set_folders.create_folders())
