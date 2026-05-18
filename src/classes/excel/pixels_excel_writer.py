import os
import pandas as pd
import traceback
from typing import Dict, Optional

from classes.excel.set_folders import SetFolders


class PixelsDataExcelWriter:
    def __init__(self, person_name, plane_folder_name, side):        
        self.person_name = person_name
        self.plane_folder_name = plane_folder_name
        self.side = side

    def _create_folder_num_pixels_data(self) -> str:
        set_folders = SetFolders(self.person_name, self.plane_folder_name, self.side)
        path_folder_side = set_folders.create_folders()

        path_folder_num_pixels = os.path.join(path_folder_side, "numeros de pixels")
        os.makedirs(path_folder_num_pixels, exist_ok=True)

        return os.path.normpath(path_folder_num_pixels)
     

    def write_num_pixels_data(self, history_whites_pixels_sequential: Dict[int, int]):
        """Orquestra a leitura, atualização e salvamento dos dados por frame na planilha Excel.
        
        Args:
            history_whites_pixels_sequential: Dicionário com dados indexados sequencialmente (1, 2, 3...)
                                              onde as chaves representam a ordem de coleta (frame1, frame2, frame3...)
        """
        if not history_whites_pixels_sequential:
            print("⚠️ Dados vazios. Nada será salvo.")
            return

        path_folder = self._create_folder_num_pixels_data()
        file_path = os.path.join(path_folder, 'dados_pixels.xlsx')
        os.makedirs(path_folder, exist_ok=True)

        try:
            self._check_file_lock(file_path)
            nome_atual = str(self.person_name)
            formatted_data, new_frame_cols = self._format_frame_data(history_whites_pixels_sequential)
            
            df = self._load_and_clean_dataframe(file_path)
            df = self._merge_and_update_dataframe(df, nome_atual, formatted_data, new_frame_cols)
            
            self._save_dataframe(df, file_path)
            print(f"✅ Processo finalizado para: {nome_atual}")

        except PermissionError:
            print("❌ ERRO: Arquivo bloqueado. Feche o Excel/LibreOffice e tente novamente.")
        except Exception as e:
            print(f"❌ ERRO CRÍTICO AO SALVAR EXCEL:")
            traceback.print_exc()

    def _check_file_lock(self, file_path: str) -> None:
        """Verifica se o arquivo está sendo usado por outro processo."""
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r+') as f:
                    pass
            except PermissionError:
                raise PermissionError(f"Arquivo em uso: {file_path}")
            except IOError as e:
                raise IOError(f"Falha ao acessar arquivo: {e}")

    def _format_frame_data(self, history_dict: Dict) -> tuple:
        """Formata os dados de pixels por número sequencial do frame (frame1, frame2, frame3...).
        
        As chaves do dicionário devem ser números sequenciais começando de 1 (1, 2, 3...)
        representando a ordem em que os frames foram coletados.
        
        Args:
            history_dict: Dicionário com chaves numéricas sequenciais (1, 2, 3...) e valores de pixels
            
        Returns:
            tuple: (dict formatado com chaves 'frame1', 'frame2'..., lista das novas colunas)
        """
        formatted = {f"frame{int(frame)}": val for frame, val in history_dict.items()}
        return formatted, list(formatted.keys())

    def _load_and_clean_dataframe(self, file_path: str) -> Optional[pd.DataFrame]:
        """Lê o Excel e aplica limpeza de dados corrompidos/execuções passadas."""
        if not os.path.exists(file_path):
            return None

        df = pd.read_excel(file_path, engine='openpyxl')
        if 'Nome Voluntário' not in df.columns:
            raise ValueError("Arquivo existe mas não contém a coluna 'Nome Voluntário'.")

        # Remove linhas vazias ou cabeçalhos duplicados acidentais
        df = df[df['Nome Voluntário'].notna()]
        df = df[df['Nome Voluntário'].astype(str) != 'Nome Voluntário']
        df = df.loc[:, ~df.columns.duplicated()]  # Remove colunas repetidas
        return df.reset_index(drop=True)

    def _merge_and_update_dataframe(self, df: Optional[pd.DataFrame], 
                                    nome_atual: str, 
                                    formatted_data: Dict, 
                                    new_frame_cols: list) -> pd.DataFrame:
        """Alinha colunas, ordena frames e atualiza/cria a linha do voluntário."""
        if df is None:
            all_frame_cols = sorted(new_frame_cols, key=lambda x: int(x.replace('frame', '')))
            nova_linha = {'Nome Voluntário': nome_atual}
            for col in all_frame_cols:
                nova_linha[col] = formatted_data.get(col, None)
            return pd.DataFrame([nova_linha])

        existing_frame_cols = [c for c in df.columns if c != 'Nome Voluntário']
        all_frame_cols = sorted(list(set(existing_frame_cols + new_frame_cols)), key=lambda x: int(x.replace('frame', '')))
        
        # 📐 Reindexa para adicionar colunas novas de forma vetorizada (evita fragmentação)
        df = df.reindex(columns=['Nome Voluntário'] + all_frame_cols)

        mask = df['Nome Voluntário'].astype(str) == nome_atual
        if mask.any():
            idx = df[mask].index[0]
            # ✅ Atualização vetorizada
            df.loc[idx, list(formatted_data.keys())] = list(formatted_data.values())
            print(f"✅ Atualizado: {nome_atual} ({len(formatted_data)} frames)")
        else:
            nova_linha = {'Nome Voluntário': nome_atual}
            for col in all_frame_cols:
                nova_linha[col] = formatted_data.get(col, None)
            # ✅ Evita FutureWarning: append seguro e compatível com pandas 2.0+
            df = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)
            print(f"🆕 Nova linha: {nome_atual}")

        return df

    def _save_dataframe(self, df: pd.DataFrame, file_path: str) -> None:
        """Salva o DataFrame sobrescrevendo o arquivo Excel."""
        df.to_excel(file_path, index=False, engine='openpyxl')
        print(f"💾 Salvo com sucesso em: {file_path}")