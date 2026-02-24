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

    def _create_folder_raw_data(self):

        set_folders = SetFolders(self.person_name, self.plane_folder_name, self.side)
        path_folder_side = set_folders.create_folders()

        path_folder_raw_data = os.path.join(path_folder_side, "dados brutos")
        if not os.path.exists(path_folder_raw_data):
            os.makedirs(path_folder_raw_data)

        return path_folder_raw_data
    
    def _create_folder_statistic_data(self):

        set_folders = SetFolders(self.person_name, self.plane_folder_name, self.side)
        path_folder_side = set_folders.create_folders()

        path_folder_statistic_data = os.path.join(path_folder_side, "dados estatisticos")
        if not os.path.exists(path_folder_statistic_data):
            os.makedirs(path_folder_statistic_data)

        return path_folder_statistic_data
    
    def _convert_data_to_raw_pandas(self):
        """
        Converte os dados em um DataFrame de linha única com colunas rotuladas.
        Coluna 0: voluntario
        Coluna 1: repeticao
        """
        # 1. Organiza os dados numéricos e achata em uma lista
        df_temp = pd.DataFrame(self.foot_repeat_data)
        dados_brutos = df_temp.values.flatten().tolist()

        # 2. Monta a linha com as variáveis de identificação
        linha_unificada = [self.person_name, self.repetition] + dados_brutos

        # 3. Cria nomes para as colunas
        # As duas primeiras são fixas, as outras serão numeradas (ex: dado_0, dado_1...)
        nomes_colunas = ['voluntario', 'repeticao'] + [f'dado_{i}' for i in range(len(dados_brutos))]

        # 4. Retorna o DataFrame com os nomes definidos
        df = pd.DataFrame([linha_unificada], columns=nomes_colunas)
        
        return df
    
    def convert_data_to_statistic_pandas(self):
        """
        Calcula estatísticas (média, mediana, desvio padrão, etc.) para cada coluna 
        e organiza tudo em uma única linha.
        """
        # 1. Cria o DataFrame temporário com os dados brutos (da imagem)
        df_temp = pd.DataFrame(self.foot_repeat_data)
        
        # 2. Lista para armazenar os valores calculados e os nomes das colunas
        estatisticas_linha = [self.person_name, self.repetition]
        nomes_colunas = ['voluntario', 'repeticao']
        
        # 3. Iterar por cada coluna original (ankle_x, ankle_y, etc.)
        for coluna in df_temp.columns:
            dados = df_temp[coluna]
            
            # Cálculos estatísticos
            media = dados.mean()
            mediana = dados.median()
            desvio_padrao = dados.std()
            maximo = dados.max()
            minimo = dados.min()
            amplitude = maximo - minimo
            
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
        df_estatistico = pd.DataFrame([estatisticas_linha], columns=nomes_colunas)
        
        return df_estatistico
        

    def write_raw_foot_data(self):
        """
        Salva os dados de repetição de pé em um arquivo Excel, acumulando novas linhas.
        """
        path_folder_repetition = os.path.normpath(self._create_folder_raw_data())
        file_path = os.path.join(path_folder_repetition, 'dados_pe.xlsx')

        # Converte os dados atuais para o formato de uma linha larga
        df_novo = self._convert_data_to_raw_pandas()
        
        if not df_novo.empty:
            try:
                # Garante que a pasta existe
                os.makedirs(path_folder_repetition, exist_ok=True)

                # Lógica para acumular dados
                if os.path.exists(file_path):
                    # Se o arquivo existe, lê o conteúdo atual
                    df_existente = pd.read_excel(file_path, engine='openpyxl')
                    # Concatena o novo dado abaixo do existente
                    df_final = pd.concat([df_existente, df_novo], ignore_index=True)
                else:
                    # Se não existe, o dado final é apenas o novo dado
                    df_final = df_novo

                # Salva o resultado final (sobrescrevendo o arquivo com a lista atualizada)
                df_final.to_excel(file_path, index=False, engine='openpyxl')
                print(f"✅ Dados salvos com sucesso em: {file_path}")

            except Exception as e:
                print(f"❌ ERRO AO ESCREVER EXCEL: {e}")
        else:
            print("❌ DataFrame vazio, nada para salvar.")

    def write_statistic_foot_data(self):
        """
        Salva os dados de repetição de pé em um arquivo Excel, acumulando novas linhas.
        """
        path_folder_repetition = os.path.normpath(self._create_folder_statistic_data())
        file_path = os.path.join(path_folder_repetition, 'dados_pe.xlsx')

        # Converte os dados atuais para o formato de uma linha larga
        df_novo = self.convert_data_to_statistic_pandas()
        
        if not df_novo.empty:
            try:
                # Garante que a pasta existe
                os.makedirs(path_folder_repetition, exist_ok=True)

                # Lógica para acumular dados
                if os.path.exists(file_path):
                    # Se o arquivo existe, lê o conteúdo atual
                    df_existente = pd.read_excel(file_path, engine='openpyxl')
                    # Concatena o novo dado abaixo do existente
                    df_final = pd.concat([df_existente, df_novo], ignore_index=True)
                else:
                    # Se não existe, o dado final é apenas o novo dado
                    df_final = df_novo

                # Salva o resultado final (sobrescrevendo o arquivo com a lista atualizada)
                df_final.to_excel(file_path, index=False, engine='openpyxl')
                print(f"✅ Dados salvos com sucesso em: {file_path}")

            except Exception as e:
                print(f"❌ ERRO AO ESCREVER EXCEL: {e}")
        else:
            print("❌ DataFrame vazio, nada para salvar.")
