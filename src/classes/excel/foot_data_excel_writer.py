import os
import pandas as pd

from classes.excel.set_folders import SetFolders


class FootDataExcelWriter:
    def __init__(self, repetition, foot_repeat_data, person_name, plane_folder_name, side):        
        """
        Inicializa o gerador de relatórios Excel de dados de repetição de pé.

        Args:
            repetition (int): A repetição do agachamento.
            foot_repeat_data (pandas.DataFrame): O DataFrame com os dados de repetição de pé.
            person_name (str): O nome da pessoa.
            plane_folder_name (str): O nome da subpasta específica do plano ('sagital' ou 'frontal').
            side (str): O lado do corpo (default='direito').
        """
        self.repetition = repetition
        self.foot_repeat_data = foot_repeat_data
        self.person_name = person_name
        self.plane_folder_name = plane_folder_name
        self.side = side

    def _get_repetition_in_str(self):
        """
        Retorna a string com o número da repetição (ex: "Repeticao 1").

        Returns:
            str: A string com o número da repetição.
        """
        return "Repeticao " + str(self.repetition)

    def _create_folder_repetition(self):
        """
        Cria a pasta específica para a repetição de pé.

        Returns:
            str: O caminho da pasta criada.
        """
        set_folders = SetFolders(self.person_name, self.plane_folder_name, self.side)
        path_folder_side = set_folders.create_folders()
        repetition_str = self._get_repetition_in_str()

        path_folder_repetition = os.path.join(path_folder_side, self.person_name ,repetition_str)
        if not os.path.exists(path_folder_repetition):
            os.makedirs(path_folder_repetition)

        return path_folder_repetition
    
    def _convert_data_to_pandas(self):
        """
        Converte os dados de repetição de pé em um DataFrame.

        Returns:
            pandas.DataFrame: O DataFrame com os dados de repetição de pé.
        """
        df = pd.DataFrame(self.foot_repeat_data)
        return df

    def write_foot_data(self):
        """
        Salva os dados de repetição de pé em um arquivo Excel.

        O caminho do arquivo é criado com base na pasta do plano e lado do corpo.
        O nome do arquivo é 'dados_pe.xlsx'.

        Se o DataFrame estiver vazio, não salva nada e imprime uma mensagem.

        Se o arquivo não existe, força a criação da pasta com o método os.makedirs.
        Se o arquivo existe, salva usando o engine openpyxl explicitamente.
        Se o arquivo for criado com sucesso, imprime uma mensagem de sucesso.
        Se houver um erro, imprime uma mensagem de erro.

        Returns:
            None
        """
        path_folder_repetition = os.path.normpath(self._create_folder_repetition())
        file_path = os.path.join(path_folder_repetition, 'dados_pe.xlsx')

        # 2. Converte os dados (já com as 6 linhas que definimos)
        df = self._convert_data_to_pandas()

        if not df.empty:
            try:
                # 3. Força a criação da pasta caso o SetFolders tenha falhado silenciosamente
                if not os.path.exists(path_folder_repetition):
                    os.makedirs(path_folder_repetition, exist_ok=True)

                # 4. Salva usando o engine openpyxl explicitamente
                df.to_excel(file_path, index=False, engine='openpyxl')
                
                # Se chegar aqui, o arquivo TEM que existir
                if os.path.exists(file_path):
                    print(f"✅ ARQUIVO CRIADO COM SUCESSO EM: {file_path}")
                else:
                    print(f"⚠️ O comando rodou, mas o arquivo não foi encontrado em: {file_path}")

            except Exception as e:
                print(f"❌ ERRO AO ESCREVER EXCEL: {e}")
        else:
            print("❌ DataFrame vazio, nada para salvar.")