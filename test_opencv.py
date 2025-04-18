import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os  # <--- Importante para manipular diretórios

# Verificar se o nome da imagem foi passado como argumento
if len(sys.argv) < 2:
    print("Uso: python3 script.py nome_da_imagem.jpg")
    sys.exit()

# Capturar o nome do arquivo da imagem
caminho_imagem = sys.argv[1]
nome_arquivo = os.path.basename(caminho_imagem)  # Ex: ufpr.jpg
nome_base = os.path.splitext(nome_arquivo)[0]    # Ex: ufpr

# Carregar imagem original
imagem = cv2.imread(caminho_imagem)

if imagem is None:
    print(f"Erro ao carregar a imagem '{caminho_imagem}'")
    exit()

# Filtro de textura vertical
kernel_vertical = np.array([
    [-1,  2, -1],
    [-1,  2, -1],
    [-1,  2, -1]
], dtype=np.float32)

# Converter para escala de cinza para aplicar o filtro
imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Aplicar o filtro
resposta = cv2.filter2D(imagem_gray, -1, kernel_vertical)

# Normalizar para 0–255
resposta_normalizada = cv2.normalize(resposta, None, 0, 255, cv2.NORM_MINMAX)
resposta_normalizada = np.uint8(resposta_normalizada)

# Aplicar colormap estilo mapa de calor (azul = fraco, vermelho = forte)
mapa_calor = cv2.applyColorMap(resposta_normalizada, cv2.COLORMAP_JET)

# Criar a figura com matplotlib
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
plt.title("Imagem Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(mapa_calor, cv2.COLOR_BGR2RGB))
plt.title("Resposta do Filtro Vertical (Mapa de Calor)")
plt.axis('off')

plt.tight_layout()

# Criar diretório de saída, se não existir
os.makedirs("resultados", exist_ok=True)

# Nome do arquivo de saída
saida = f"resultados/heatmap_{nome_base}.png"
plt.savefig(saida)
print(f"Mapa de calor salvo como {saida}")
