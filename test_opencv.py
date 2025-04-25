import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import shutil
from sklearn.cluster import KMeans

# Se o argumento "clear" for passado, limpa a pasta 'resultados'
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

# Caminho da imagem e preparação da pasta de resultados
caminho_imagem = sys.argv[1]
nome_arquivo = os.path.basename(caminho_imagem)
nome_base = os.path.splitext(nome_arquivo)[0]
pasta_resultados = os.path.join("resultados", nome_base)
os.makedirs(pasta_resultados, exist_ok=True)

# Carregar imagem
imagem = cv2.imread(caminho_imagem)
if imagem is None:
    print(f"Erro ao carregar a imagem '{caminho_imagem}'")
    exit()

# Filtros
kernel_sobel_vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
kernel_sobel_horizontal = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
kernel_laplaciano = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
kernel_45 = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]], dtype=np.float32)
kernel_135 = np.array([[1, 1, 0], [-1, 0, 1], [-1, -1, 0]], dtype=np.float32)

# Processar por escalas
respostas_por_escala = []
imagem_atual = imagem.copy()

for escala in range(3):
    print(f"\nProcessando escala {escala}...")
    respostas_acumuladas = []

    pasta_escala = os.path.join(pasta_resultados, f"escala_{escala}")
    os.makedirs(pasta_escala, exist_ok=True)

    if escala > 0:
        imagem_blur = cv2.GaussianBlur(imagem_atual, (5, 5), sigmaX=1)
        imagem_atual = cv2.pyrDown(imagem_blur)
    else:
        imagem_blur = imagem_atual

    def aplicar_filtro(nome, kernel):
        resposta = cv2.filter2D(imagem_atual, -1, kernel)
        resposta_gray = cv2.cvtColor(resposta, cv2.COLOR_BGR2GRAY)
        resposta_normalizada = cv2.normalize(resposta_gray, None, 0, 255, cv2.NORM_MINMAX)
        resposta_normalizada = np.uint8(resposta_normalizada)
        mapa_calor = cv2.applyColorMap(resposta_normalizada, cv2.COLORMAP_JET)
        respostas_acumuladas.append(resposta_normalizada.astype(np.float32))

        cv2.imwrite(os.path.join(pasta_escala, f"heatmap_{nome.lower()}.png"), mapa_calor)

    def aplicar_dog():
        imagem_gray = cv2.cvtColor(imagem_atual, cv2.COLOR_BGR2GRAY)
        blur1 = cv2.GaussianBlur(imagem_gray, (9, 9), sigmaX=1)
        blur2 = cv2.GaussianBlur(imagem_gray, (9, 9), sigmaX=2)
        dog = cv2.subtract(blur1, blur2)
        dog_shifted = dog - dog.min()
        dog_normalizado = 255 * (dog_shifted / dog_shifted.max())
        dog_normalizado = np.uint8(dog_normalizado)
        mapa_calor = cv2.applyColorMap(dog_normalizado, cv2.COLORMAP_JET)
        respostas_acumuladas.append(dog_normalizado.astype(np.float32))

        cv2.imwrite(os.path.join(pasta_escala, "heatmap_dog.png"), mapa_calor)

    aplicar_filtro("Vertical", kernel_sobel_vertical)
    aplicar_filtro("Horizontal", kernel_sobel_horizontal)
    aplicar_filtro("Laplaciano", kernel_laplaciano)
    aplicar_filtro("45", kernel_45)
    aplicar_filtro("135", kernel_135)
    aplicar_dog()

    respostas_por_escala.append((respostas_acumuladas.copy(), imagem_atual.copy()))
    print(f"Resultados da escala {escala} salvos em: {pasta_escala}")

# Segmentação por textura
print("\nIniciando segmentação por textura para cada escala...")

segmentacoes = []

for escala, (respostas, imagem_base) in enumerate(respostas_por_escala):
    print(f"Segmentando escala {escala}...")
    janela = 3
    passo = 1
    altura, largura = respostas[0].shape
    vetores = []
    posicoes = []

    for y in range(0, altura - janela, passo):
        for x in range(0, largura - janela, passo):
            vetor_janela = [np.mean(resp[y:y+janela, x:x+janela]) for resp in respostas]
            vetores.append(vetor_janela)
            posicoes.append((x, y))

    vetores = np.array(vetores)
    kmeans = KMeans(n_clusters=4, random_state=10).fit(vetores)
    rotulos = kmeans.labels_

    cores = np.random.randint(0, 255, size=(4, 3), dtype=np.uint8)
    imagem_segmentada = np.zeros((altura, largura, 3), dtype=np.uint8)

    for (x, y), r in zip(posicoes, rotulos):
        imagem_segmentada[y:y+janela, x:x+janela] = cores[r]

    caminho_saida = os.path.join(pasta_resultados, f"segmentacao_textura_escala_{escala}.png")
    cv2.imwrite(caminho_saida, imagem_segmentada)
    segmentacoes.append(imagem_segmentada)
    print(f"Segmentação da escala {escala} salva em: {caminho_saida}")

# Comparativo final de heatmaps
print("\nGerando imagem comparativa final dos heatmaps...")

heatmaps_finais = []
referencia_shape = respostas_por_escala[0][1].shape[:2][::-1]  # largura x altura da escala 0
for escala, (_, imagem_original_escala) in enumerate(respostas_por_escala):
    dog_path = os.path.join(pasta_resultados, f"escala_{escala}", "heatmap_dog.png")
    dog = cv2.imread(dog_path)
    if dog is not None:
        resized_dog = cv2.resize(dog, referencia_shape)
        heatmaps_finais.append((resized_dog, f"Escala {escala}"))

# Adiciona a imagem original redimensionada como referência
img_original_resized = cv2.resize(imagem, referencia_shape)
heatmaps_finais.insert(0, (img_original_resized, "Original"))

# Monta imagem lado a lado
largura_img, altura_img = referencia_shape
largura_total = largura_img * len(heatmaps_finais)
comparativo_heatmap = np.zeros((altura_img + 40, largura_total, 3), dtype=np.uint8) + 255

x_offset = 0
for img, label in heatmaps_finais:
    comparativo_heatmap[40:40+altura_img, x_offset:x_offset+largura_img] = img
    cv2.putText(comparativo_heatmap, label, (x_offset + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    x_offset += largura_img

cv2.imwrite(os.path.join(pasta_resultados, "comparativo_heatmaps.png"), comparativo_heatmap)
print("Comparativo de heatmaps salvo.")

# Comparativo final das segmentações
print("Gerando imagem comparativa final das segmentações...")

segmentacoes_resized = [cv2.resize(seg, referencia_shape) for seg in segmentacoes]
img_original_resized = cv2.resize(imagem, referencia_shape)

# Junta imagem original + segmentações
segmentacoes_com_original = [img_original_resized] + segmentacoes_resized
rotulos_segmentacoes = ["Original"] + [f"Escala {i}" for i in range(len(segmentacoes))]

comparativo_segmentacoes = np.zeros((altura_img + 40, largura_img * len(segmentacoes_com_original), 3), dtype=np.uint8) + 255

x_offset = 0
for idx, img in enumerate(segmentacoes_com_original):
    comparativo_segmentacoes[40:40+altura_img, x_offset:x_offset+largura_img] = img
    cv2.putText(comparativo_segmentacoes, rotulos_segmentacoes[idx], (x_offset + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    x_offset += largura_img

cv2.imwrite(os.path.join(pasta_resultados, "comparativo_segmentacoes.png"), comparativo_segmentacoes)
print("Comparativo de segmentações salvo.")


