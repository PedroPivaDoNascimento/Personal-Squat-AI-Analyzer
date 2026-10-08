import os

class SetFolders():

    # Nome da subpasta que concentra os dados de segmentos biomecânicos
    # (séries temporais brutas e estatísticas) definidos em
    # docs/tasks/(3)NEW-DATAXANALYTICS-SHEETS.md.
    SEGMENT_DATA_FOLDER_NAME = "dados_do_segmento"
    SEGMENT_RAW_SUBFOLDER_NAME = "dados brutos"
    SEGMENT_STAT_SUBFOLDER_NAME = "estatistica"

    def __init__(self, person_name, plane_folder_name, side="direito"):
        """
        Inicializa o criador de pastas.

        Args:
            person_name (str): O nome da pessoa.
            plane_folder_name (str): O nome da subpasta específica do plano ('sagital' ou 'frontal').
            side (str): O lado do corpo (default='direito').
        """

        self.side = side
        self.person_name = person_name
        self.plane_folder_name = plane_folder_name

    def create_folders(self):
        """
        Cria a estrutura de pastas (planilhas/plano/lado) se não existir.

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
        
        if self.plane_folder_name == 'frontal':
            final_output_folder = os.path.join(plane_output_folder, side_folder_name, "dados_pe")

        # Cria a estrutura de pastas (planilhas/plano/lado)
        if not os.path.exists(final_output_folder):
            os.makedirs(final_output_folder)
        
        return final_output_folder

    def _side_folder_path(self):
        """
        Retorna o caminho da pasta do lado sem criar nada em disco:
        `planilhas/<plano>/<lado>`.
        """
        return os.path.join(
            'planilhas', self.plane_folder_name, str(self.side).lower()
        )

    def create_segment_folders(self):
        """
        Cria (de forma determinística e idempotente, via os.makedirs com
        exist_ok=True) a árvore aninhada de dados de segmentos biomecânicos:

            planilhas/<plano>/<lado>/dados_do_segmento/
            ├── dados brutos/
            └── estatistica/

        Returns:
            dict: {'base': <dados_do_segmento>, 'raw': <dados brutos>,
                   'statistic': <estatistica>} com os caminhos relativos.
        """
        base_folder = os.path.join(
            self._side_folder_path(), self.SEGMENT_DATA_FOLDER_NAME
        )
        raw_folder = os.path.join(base_folder, self.SEGMENT_RAW_SUBFOLDER_NAME)
        statistic_folder = os.path.join(base_folder, self.SEGMENT_STAT_SUBFOLDER_NAME)

        for folder in (base_folder, raw_folder, statistic_folder):
            os.makedirs(folder, exist_ok=True)

        return {
            'base': base_folder,
            'raw': raw_folder,
            'statistic': statistic_folder,
        }

    def get_segment_raw_data_folder(self):
        """Garante e retorna o caminho de `.../dados_do_segmento/dados brutos/`."""
        return self.create_segment_folders()['raw']

    def get_segment_statistic_data_folder(self):
        """Garante e retorna o caminho de `.../dados_do_segmento/estatistica/`."""
        return self.create_segment_folders()['statistic']