import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import shutil
from sklearn.cluster import KMeans

# ---------- Configurações ----------
N_CLUSTERS = 4  # Número de clusters para segmentação
JANELA = 3      # Tamanho da janela para extração de vetores
PASSO = 1       # Passo para varredura
ESCALAS = 3     # Quantidade de escalas
CORES_FIXAS = np.array([
    [255, 0, 0],     # Vermelho
    [200, 200, 200], # Cinza claro
    [0, 0, 255],     # Azul
    [255, 255, 0],   # Amarelo
    [0, 255, 255],   # Ciano
    [255, 0, 255],   # Magenta
    [128, 0, 128],   # Roxo
    [0, 128, 128],   # Verde-água
], dtype=np.uint8)

# ---------- Funções auxiliares ----------

def limpar_resultados():
    pasta_resultados = "resultados"
    if os.path.exists(pasta_resultados):
        shutil.rmtree(pasta_resultados)
        print(f"Pasta '{pasta_resultados}' limpa com sucesso.")
    else:
        print(f"Pasta '{pasta_resultados}' não existe.")
    sys.exit()

def aplicar_filtro(imagem, kernel):
    resposta = cv2.filter2D(imagem, -1, kernel)
    resposta_gray = cv2.cvtColor(resposta, cv2.COLOR_BGR2GRAY)
    resposta_norm = cv2.normalize(resposta_gray, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(resposta_norm)

def aplicar_dog(imagem):
    imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    blur1 = cv2.GaussianBlur(imagem_gray, (9, 9), sigmaX=1)
    blur2 = cv2.GaussianBlur(imagem_gray, (9, 9), sigmaX=2)
    dog = cv2.subtract(blur1, blur2)
    dog_norm = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(dog_norm)

def gerar_cores(n_clusters):
    if n_clusters <= len(CORES_FIXAS):
        return CORES_FIXAS[:n_clusters]
    else:
        np.random.seed(10)  # Sempre gerar as mesmas cores extras
        cores_extras = np.random.randint(0, 255, size=(n_clusters - len(CORES_FIXAS), 3), dtype=np.uint8)
        return np.vstack((CORES_FIXAS, cores_extras))

def salvar_mapa_calor(imagem, caminho):
    heatmap = cv2.applyColorMap(imagem, cv2.COLORMAP_JET)
    cv2.imwrite(caminho, heatmap)

def criar_comparativo(imagens_rotuladas, saida):
    largura_img, altura_img = imagens_rotuladas[0][0].shape[1], imagens_rotuladas[0][0].shape[0]
    largura_total = largura_img * len(imagens_rotuladas)
    comparativo = np.ones((altura_img + 40, largura_total, 3), dtype=np.uint8) * 255

    x_offset = 0
    for img, label in imagens_rotuladas:
        comparativo[40:40+altura_img, x_offset:x_offset+largura_img] = img
        cv2.putText(comparativo, label, (x_offset + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        x_offset += largura_img

    cv2.imwrite(saida, comparativo)

# ---------- Execução principal ----------

# Verificar argumentos
if len(sys.argv) == 2 and sys.argv[1] == "clear":
    limpar_resultados()

if len(sys.argv) < 2:
    print("Uso: python3 script.py nome_da_imagem.jpg")
    print("Ou:   python3 script.py clear  (para limpar todos os resultados)")
    sys.exit()

# Preparação
caminho_imagem = sys.argv[1]
nome_arquivo = os.path.basename(caminho_imagem)
nome_base = os.path.splitext(nome_arquivo)[0]
pasta_resultados = os.path.join("resultados", nome_base)
os.makedirs(pasta_resultados, exist_ok=True)

imagem = cv2.imread(caminho_imagem)
if imagem is None:
    print(f"Erro ao carregar a imagem '{caminho_imagem}'")
    sys.exit()

# Definição dos kernels
filtros = {
    "Vertical": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
    "Horizontal": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
    "Laplaciano": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
    "45": np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]], dtype=np.float32),
    "135": np.array([[1, 1, 0], [-1, 0, 1], [-1, -1, 0]], dtype=np.float32),
}

# Processar escalas
respostas_por_escala = []
imagem_atual = imagem.copy()

for escala in range(ESCALAS):
    print(f"\nProcessando escala {escala}...")
    pasta_escala = os.path.join(pasta_resultados, f"escala_{escala}")
    os.makedirs(pasta_escala, exist_ok=True)

    if escala > 0:
        imagem_atual = cv2.pyrDown(cv2.GaussianBlur(imagem_atual, (5,5), sigmaX=1))

    respostas = []
    for nome, kernel in filtros.items():
        resposta = aplicar_filtro(imagem_atual, kernel)
        salvar_mapa_calor(resposta, os.path.join(pasta_escala, f"heatmap_{nome.lower()}.png"))
        respostas.append(resposta.astype(np.float32))

    dog = aplicar_dog(imagem_atual)
    salvar_mapa_calor(dog, os.path.join(pasta_escala, "heatmap_dog.png"))
    respostas.append(dog.astype(np.float32))

    respostas_por_escala.append((respostas.copy(), imagem_atual.copy()))
    print(f"Resultados da escala {escala} salvos em: {pasta_escala}")

# Segmentação de textura
print("\nIniciando segmentação por textura...")
segmentacoes = []
cores = gerar_cores(N_CLUSTERS)

for escala, (respostas, imagem_base) in enumerate(respostas_por_escala):
    print(f"Segmentando escala {escala}...")

    altura, largura = respostas[0].shape
    vetores = []
    posicoes = []

    for y in range(0, altura - JANELA, PASSO):
        for x in range(0, largura - JANELA, PASSO):
            vetor = [np.mean(resp[y:y+JANELA, x:x+JANELA]) for resp in respostas]
            vetores.append(vetor)
            posicoes.append((x, y))

    vetores = np.array(vetores)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=10).fit(vetores)
    rotulos = kmeans.labels_

    imagem_segmentada = np.zeros((altura, largura, 3), dtype=np.uint8)
    for (x, y), r in zip(posicoes, rotulos):
        imagem_segmentada[y:y+JANELA, x:x+JANELA] = cores[r]

    caminho_saida = os.path.join(pasta_resultados, f"segmentacao_textura_escala_{escala}.png")
    cv2.imwrite(caminho_saida, imagem_segmentada)
    segmentacoes.append(imagem_segmentada)

    print(f"Segmentação da escala {escala} salva em: {caminho_saida}")

# Heatmap combinado
print("\nGerando heatmap combinado (soma dos filtros) por escala...")

for escala, (respostas, _) in enumerate(respostas_por_escala):
    combinacao = np.sum(respostas, axis=0)  # somando as respostas dos filtros
    combinacao_norm = cv2.normalize(combinacao, None, 0, 255, cv2.NORM_MINMAX)
    caminho_heatmap_final = os.path.join(pasta_resultados, f"escala_{escala}", "heatmap_final.png")
    salvar_mapa_calor(np.uint8(combinacao_norm), caminho_heatmap_final)
    print(f"Heatmap final (soma) da escala {escala} salvo em: {caminho_heatmap_final}")

# Imagens comparativas
print("\nGerando imagens comparativas...")

referencia_shape = respostas_por_escala[0][1].shape[:2][::-1]
img_original_resized = cv2.resize(imagem, referencia_shape)

# Comparativo heatmaps (SOMA dos filtros)
heatmaps_finais = [(img_original_resized, "Original")]
for escala in range(ESCALAS):
    heatmap_final = cv2.imread(os.path.join(pasta_resultados, f"escala_{escala}", "heatmap_final.png"))
    if heatmap_final is not None:
        heatmaps_finais.append((cv2.resize(heatmap_final, referencia_shape), f"Escala {escala}"))

criar_comparativo(heatmaps_finais, os.path.join(pasta_resultados, "comparativo_heatmaps.png"))
print("Comparativo de heatmaps salvo.")

# Comparativo segmentações
segmentacoes_resized = [(img_original_resized, "Original")] + [(cv2.resize(seg, referencia_shape), f"Escala {i}") for i, seg in enumerate(segmentacoes)]

criar_comparativo(segmentacoes_resized, os.path.join(pasta_resultados, "comparativo_segmentacoes.png"))
print("Comparativo de segmentações salvo.")