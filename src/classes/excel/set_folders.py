import os

# Segmentos biomecânicos suportados pelo módulo de organização e
# armazenamento de dados (task (3)NEW-DATA_ANALYTICS-SHEETS.md).
# Cada segmento gera a pasta dinâmica `planilhas/<plano>/<lado>/dados_<segmento>/`.
VALID_SEGMENTS = ('pe', 'joelho', 'quadril', 'tronco', 'cabeca')


class SetFolders():

    def __init__(self, person_name, plane_folder_name, side="direito", segment_name=None):
        """
        Inicializa o criador de pastas.

        Args:
            person_name (str): O nome da pessoa.
            plane_folder_name (str): O nome da subpasta específica do plano ('sagital' ou 'frontal').
            side (str): O lado do corpo (default='direito').
            segment_name (str, opcional): Nome do segmento biomecânico analisado
                ('pe', 'joelho', 'quadril', 'tronco', 'cabeca'). Quando informado,
                o caminho final passa a ser `planilhas/<plano>/<lado>/dados_<segmento>/`.
                Quando omitido em análises frontais, mantém-se o comportamento
                legado ('dados_pe') para retrocompatibilidade com o Streamlit e
                com as rotinas de limpeza/relatórios já existentes.
        """

        self.side = side
        self.person_name = person_name
        self.plane_folder_name = plane_folder_name
        self.segment_name = self._normalize_segment(segment_name)

    @staticmethod
    def _normalize_segment(segment_name):
        """
        Normaliza e valida o nome do segmento biomecânico.

        Args:
            segment_name (str | None): Nome bruto do segmento.

        Returns:
            str | None: Nome normalizado em minúsculas, ou None se não informado.

        Raises:
            ValueError: Se o segmento não pertencer à lista de segmentos suportados.
        """
        if segment_name is None:
            return None

        normalized = str(segment_name).strip().lower()
        if not normalized:
            return None

        if normalized not in VALID_SEGMENTS:
            raise ValueError(
                f"Segmento biomecânico inválido: '{segment_name}'. "
                f"Segmentos suportados: {', '.join(VALID_SEGMENTS)}."
            )
        return normalized

    def create_folders(self):
        """
        Cria a estrutura de pastas (planilhas/plano/lado[/dados_segmento]) se não existir.

        A resolução é dinâmica para qualquer segmento biomecânico (Fase 1.2 da task
        (3)NEW-DATA_ANALYTICS-SHEETS.md): `planilhas/<plano>/<lado>/dados_<segmento>/`.

        Returns:
            str: O caminho da pasta criada.
        """

        output_folder = 'planilhas'

        # 1. Caminho da pasta do Plano (planilhas/sagital ou planilhas/frontal)
        plane_output_folder = os.path.join(output_folder, self.plane_folder_name)

        # 2. Caminho da pasta do Lado (planilhas/sagital/direito ou planilhas/frontal/esquerdo)
        # O nome do lado deve ser minúsculo para consistência.
        side_folder_name = self.side.lower()

        final_output_folder = os.path.join(plane_output_folder, side_folder_name)

        # 3. Pasta dinâmica do segmento (planilhas/<plano>/<lado>/dados_<segmento>/).
        # Fallback legado: plano frontal sem segmento explícito => dados_pe.
        segment_folder = self.segment_name
        if segment_folder is None and self.plane_folder_name == 'frontal':
            segment_folder = 'pe'

        if segment_folder:
            final_output_folder = os.path.join(
                plane_output_folder, side_folder_name, f'dados_{segment_folder}'
            )

        # Cria a estrutura de pastas de forma determinística e idempotente
        # (Fase 1.3): exist_ok=True evita corrida entre writers concorrentes.
        os.makedirs(final_output_folder, exist_ok=True)

        return final_output_folder

    def create_segment_subfolders(self):
        """
        Cria deterministicamente as subpastas de dados do segmento:
        `dados brutos/` e `dados estatisticos/` dentro de
        `planilhas/<plano>/<lado>/dados_<segmento>/` (Fase 1.3 da task).

        Returns:
            dict: {'raw': <caminho dados brutos>, 'statistic': <caminho dados estatisticos>}
        """

        path_folder_side = self.create_folders()

        path_folder_raw_data = os.path.join(path_folder_side, "dados brutos")
        os.makedirs(path_folder_raw_data, exist_ok=True)

        path_folder_statistic_data = os.path.join(path_folder_side, "dados estatisticos")
        os.makedirs(path_folder_statistic_data, exist_ok=True)

        return {
            'raw': path_folder_raw_data,
            'statistic': path_folder_statistic_data,
        }
