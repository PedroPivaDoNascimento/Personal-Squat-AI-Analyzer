import cv2 as cv
import os
import shutil
import numpy as np

class VideoProcessor():
    def __init__(self, body_part):
        # 1. Normaliza o nome
        self.body_part_name = body_part.strip()
        self.cont_frame = 0
        self.first_frame = None

        # 🔍 LÓGICA ROBUSTA: Sobe diretórios até achar a pasta 'src'
        # O root do projeto será sempre o PAI da pasta 'src'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = current_dir
        
        # Sobe no máximo 10 níveis para evitar loops infinitos em sistemas estranhos
        for _ in range(10):
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:  # Chegou na raiz do SO
                break
            if os.path.isdir(os.path.join(current_dir, 'src')):
                self.project_root = current_dir  # ✅ Achou! O root é o dir que contém 'src'
                break
            current_dir = parent_dir

        # Define o caminho fixo de destino
        self.pasta_destino = os.path.join(self.project_root, "imagens", f"frames_{self.body_part_name}")


    def set_up_folders(self):
        """Cria e limpa a pasta de destino antes do processamento"""
        pasta_imagens = os.path.join(self.project_root, "imagens")
        os.makedirs(pasta_imagens, exist_ok=True)
        
        if os.path.exists(self.pasta_destino):
            shutil.rmtree(self.pasta_destino)
        os.makedirs(self.pasta_destino, exist_ok=True)
        
    def _save_frame(self, frame):
        """Salva o frame na pasta destino com verificação de segurança"""
        if frame is None or frame.size == 0:
            return

        # Fallback: garante criação mesmo se set_up_folders falhar
        try:
            os.makedirs(self.pasta_destino, exist_ok=True)
        except Exception as e:
            print(f"❌ Erro crítico ao criar pasta: {e}")
            return

        nome_arquivo = os.path.join(self.pasta_destino, f"frame_{str(self.cont_frame).zfill(4)}.jpg")
        
        try:
            sucesso, buffer = cv.imencode(".jpg", frame)
            if sucesso:
                buffer.tofile(nome_arquivo)
                self.cont_frame += 1
            else:
                print(f"❌ Falha ao codificar frame {self.cont_frame}")
        except Exception as e:
            print(f"❌ Erro ao salvar: {e} | Caminho: {nome_arquivo}")

    def crop_roi(self, frame, norm_x, norm_y, sz_crop=40):
        """Recorta ROI e salva automaticamente"""
        if frame is None: 
            return None
            
        img_h, img_w = frame.shape[:2]
        px_x = int(max(0, min(1, norm_x)) * img_w)
        px_y = int(max(0, min(1, norm_y)) * img_h) 
        radius = sz_crop // 2

        y_min, y_max = max(0, px_y - radius), min(img_h, px_y + radius) + 15
        x_min, x_max = max(0, px_x - radius) + 5, min(img_w, px_x + radius) + 5

        roi_extraida = frame[y_min:y_max, x_min:x_max]
        self._save_frame(roi_extraida)
        return roi_extraida
    
    def count_white_pixels(self, frame_atual):
        if frame_atual is None: return 0
        
        # Armazena o primeiro frame
        if self.first_frame is None:
            self.first_frame = frame_atual
            return 0
        
        # Diferença entre as imagens 
        diff = cv.absdiff(self.first_frame, frame_atual)        
        gray_diff = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)

        # Binarização
        _, binary_image = cv.threshold(gray_diff, 50, 255, cv.THRESH_BINARY)
        
        return cv.countNonZero(binary_image)
