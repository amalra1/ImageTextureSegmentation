import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import shutil

# Se o argumento "clear" for passado sozinho, limpa a pasta 'resultados' inteira
if len(sys.argv) == 2 and sys.argv[1] == "clear":
    pasta_resultados = "resultados"
    if os.path.exists(pasta_resultados):
        shutil.rmtree(pasta_resultados)
        print(f"Pasta '{pasta_resultados}' limpa com sucesso.")
    else:
        print(f"Pasta '{pasta_resultados}' não existe.")
    sys.exit()

# Verificar se o nome da imagem foi passado como argumento
if len(sys.argv) < 2:
    print("Uso: python3 script.py nome_da_imagem.jpg")
    print("Ou:   python3 script.py clear  (para limpar todos os resultados)")
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

# Filtros de Sobel (mais robustos)
kernel_sobel_vertical = np.array([
    [-1,  0,  1],
    [-2,  0,  2],
    [-1,  0,  1]
], dtype=np.float32)

kernel_sobel_horizontal = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

# Filtro Laplaciano
kernel_laplaciano = np.array([
    [0,  1, 0],
    [1, -4, 1],
    [0,  1, 0]
], dtype=np.float32)

# Filtro de borda diagonal 45°
kernel_45 = np.array([
    [0, -1, -1],
    [1,  0, -1],
    [1,  1,  0]
], dtype=np.float32)

# Filtro de borda diagonal 135°
kernel_135 = np.array([
    [1,  1,  0],
    [-1, 0,  1],
    [-1, -1,  0]
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

    respostas_acumuladas.append(resposta_normalizada.astype(np.float32))

    plt.figure(figsize=(14, 6))
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

def aplicar_dog():
    imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Dois níveis de desfoque (sigmas diferentes)
    blur1 = cv2.GaussianBlur(imagem_gray, (9, 9), sigmaX=1)
    blur2 = cv2.GaussianBlur(imagem_gray, (9, 9), sigmaX=2)

    dog = cv2.subtract(blur1, blur2)

    # Normalizar e aplicar colormap como nos outros filtros
    dog_shifted = dog - dog.min()
    dog_normalizado = 255 * (dog_shifted / dog_shifted.max())
    dog_normalizado = np.uint8(dog_normalizado)
    mapa_calor = cv2.applyColorMap(dog_normalizado, cv2.COLORMAP_JET)

    # Acumular
    respostas_acumuladas.append(dog_normalizado.astype(np.float32))

    # Plotar e salvar
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    plt.title("Imagem Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(mapa_calor, cv2.COLOR_BGR2RGB))
    plt.title("Filtro Circular (DoG)")
    plt.axis('off')

    plt.tight_layout()
    saida = os.path.join(pasta_resultados, "heatmap_dog.png")
    plt.savefig(saida)
    plt.close()
    print(f"Mapa de calor (DoG) salvo em: {saida}")


# Aplicar os filtros
aplicar_filtro("Vertical", kernel_sobel_vertical)
aplicar_filtro("Horizontal", kernel_sobel_horizontal)
aplicar_filtro("Laplaciano", kernel_laplaciano)
aplicar_filtro("45", kernel_45)
aplicar_filtro("135", kernel_135)
aplicar_dog()  # Fica amarelo o fundo ao invés de azul 

# Criar o heatmap final acumulado
soma = np.sum(respostas_acumuladas, axis=0)
soma_normalizada = cv2.normalize(soma, None, 0, 255, cv2.NORM_MINMAX)
soma_normalizada = np.uint8(soma_normalizada)
mapa_calor_final = cv2.applyColorMap(soma_normalizada, cv2.COLORMAP_JET)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
plt.title("Imagem Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(mapa_calor_final, cv2.COLOR_BGR2RGB))
plt.title("Heatmap Final (Soma dos Filtros)")
plt.axis('off')

plt.tight_layout()
saida_final = os.path.join(pasta_resultados, "heatmap_final.png")
plt.savefig(saida_final)
plt.close()
print(f"Heatmap final salvo em: {saida_final}")

