import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Verificar se o nome da imagem foi passado como argumento
if len(sys.argv) < 2:
    print("Uso: python3 script.py nome_da_imagem.jpg")
    sys.exit()

# Capturar o nome do arquivo da imagem
caminho_imagem = sys.argv[1]
nome_arquivo = os.path.basename(caminho_imagem)
nome_base = os.path.splitext(nome_arquivo)[0]

# Criar diretório de saída específico para a imagem
pasta_resultados = os.path.join("resultados", nome_base)
os.makedirs(pasta_resultados, exist_ok=True)

# Carregar imagem original
imagem = cv2.imread(caminho_imagem)

if imagem is None:
    print(f"Erro ao carregar a imagem '{caminho_imagem}'")
    exit()

# Filtros de textura
kernel_vertical = np.array([
    [-1,  2, -1],
    [-1,  2, -1],
    [-1,  2, -1]
], dtype=np.float32)

kernel_horizontal = np.array([
    [-1, -1, -1],
    [ 2,  2,  2],
    [-1, -1, -1]
], dtype=np.float32)

# Lista para somar as respostas de todos os filtros
respostas_acumuladas = []

# Função para aplicar filtro, normalizar, colorir e salvar
def aplicar_filtro(nome, kernel):
    resposta = cv2.filter2D(imagem, -1, kernel)
    resposta_gray = cv2.cvtColor(resposta, cv2.COLOR_BGR2GRAY)
    resposta_normalizada = cv2.normalize(resposta_gray, None, 0, 255, cv2.NORM_MINMAX)
    resposta_normalizada = np.uint8(resposta_normalizada)
    mapa_calor = cv2.applyColorMap(resposta_normalizada, cv2.COLORMAP_JET)

    # Adicionar ao acumulado
    respostas_acumuladas.append(resposta_normalizada.astype(np.float32))

    # Plot e salvar resultado
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    plt.title("Imagem Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(mapa_calor, cv2.COLOR_BGR2RGB))
    plt.title(f"Filtro {nome} (Mapa de Calor)")
    plt.axis('off')

    plt.tight_layout()
    saida = os.path.join(pasta_resultados, f"heatmap_{nome.lower()}.png")
    plt.savefig(saida)
    plt.close()
    print(f"Mapa de calor ({nome}) salvo em: {saida}")

# Aplicar os filtros definidos
aplicar_filtro("Vertical", kernel_vertical)
aplicar_filtro("Horizontal", kernel_horizontal)

# Criar o heatmap final acumulado
soma = np.sum(respostas_acumuladas, axis=0)
soma_normalizada = cv2.normalize(soma, None, 0, 255, cv2.NORM_MINMAX)
soma_normalizada = np.uint8(soma_normalizada)
mapa_calor_final = cv2.applyColorMap(soma_normalizada, cv2.COLORMAP_JET)

# Plotar e salvar o heatmap final
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(mapa_calor_final, cv2.COLOR_BGR2RGB))
plt.title("Heatmap Final (Soma dos Filtros)")
plt.axis('off')
plt.tight_layout()
saida_final = os.path.join(pasta_resultados, f"heatmap_final.png")
plt.savefig(saida_final)
plt.close()
print(f"Heatmap final salvo em: {saida_final}")
