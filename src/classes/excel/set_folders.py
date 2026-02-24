import os

class SetFolders():

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