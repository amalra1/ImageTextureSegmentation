import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import shutil
from sklearn.cluster import KMeans

def limpar_resultados(pasta="resultados"):
    if os.path.exists(pasta):
        shutil.rmtree(pasta)
        print(f"Pasta '{pasta}' limpa com sucesso.")
    else:
        print(f"Pasta '{pasta}' não existe.")

def preparar_pasta_resultados(caminho_imagem):
    nome_arquivo = os.path.basename(caminho_imagem)
    nome_base = os.path.splitext(nome_arquivo)[0]
    pasta_resultados = os.path.join("resultados", nome_base)
    os.makedirs(pasta_resultados, exist_ok=True)
    return pasta_resultados, nome_base

def carregar_imagem(caminho_imagem):
    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        print(f"Erro ao carregar a imagem '{caminho_imagem}'")
        sys.exit()
    return imagem

def definir_kernels():
    return {
        "Vertical": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
        "Horizontal": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
        "Laplaciano": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
        "45": np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]], dtype=np.float32),
        "135": np.array([[1, 1, 0], [-1, 0, 1], [-1, -1, 0]], dtype=np.float32)
    }

def aplicar_filtros(imagem, pasta_saida, kernels):
    respostas = []
    for nome, kernel in kernels.items():
        resposta = cv2.filter2D(imagem, -1, kernel)
        resposta_gray = cv2.cvtColor(resposta, cv2.COLOR_BGR2GRAY)
        resposta_normalizada = cv2.normalize(resposta_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mapa_calor = cv2.applyColorMap(resposta_normalizada, cv2.COLORMAP_JET)
        respostas.append(resposta_normalizada.astype(np.float32))
        cv2.imwrite(os.path.join(pasta_saida, f"heatmap_{nome.lower()}.png"), mapa_calor)
    return respostas

def aplicar_dog(imagem, pasta_saida):
    imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    blur1 = cv2.GaussianBlur(imagem_gray, (9, 9), sigmaX=1)
    blur2 = cv2.GaussianBlur(imagem_gray, (9, 9), sigmaX=2)
    dog = cv2.subtract(blur1, blur2)
    dog_normalizado = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mapa_calor = cv2.applyColorMap(dog_normalizado, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(pasta_saida, "heatmap_dog.png"), mapa_calor)
    return dog_normalizado.astype(np.float32)

def processar_escalas(imagem, pasta_resultados, num_escalas=3):
    respostas_por_escala = []
    imagem_atual = imagem.copy()
    kernels = definir_kernels()

    for escala in range(num_escalas):
        print(f"\nProcessando escala {escala}...")
        pasta_escala = os.path.join(pasta_resultados, f"escala_{escala}")
        os.makedirs(pasta_escala, exist_ok=True)

        if escala > 0:
            imagem_blur = cv2.GaussianBlur(imagem_atual, (5, 5), sigmaX=1)
            imagem_atual = cv2.pyrDown(imagem_blur)

        respostas = aplicar_filtros(imagem_atual, pasta_escala, kernels)
        respostas.append(aplicar_dog(imagem_atual, pasta_escala))

        respostas_por_escala.append((respostas, imagem_atual.copy()))
    
    return respostas_por_escala

def segmentar_textura(respostas, janela=3, passo=1, n_clusters=4):
    altura, largura = respostas[0].shape
    vetores, posicoes = [], []

    for y in range(0, altura - janela, passo):
        for x in range(0, largura - janela, passo):
            vetor = [np.mean(resp[y:y+janela, x:x+janela]) for resp in respostas]
            vetores.append(vetor)
            posicoes.append((x, y))

    vetores = np.array(vetores)
    kmeans = KMeans(n_clusters=n_clusters, random_state=10).fit(vetores)
    return kmeans.labels_, posicoes, (largura, altura)

def salvar_segmentacao(rotulos, posicoes, dimensoes, janela, pasta_saida, escala):
    largura, altura = dimensoes
    cores = np.random.randint(0, 255, (np.max(rotulos)+1, 3), dtype=np.uint8)
    segmentacao = np.zeros((altura, largura, 3), dtype=np.uint8)

    for (x, y), r in zip(posicoes, rotulos):
        segmentacao[y:y+janela, x:x+janela] = cores[r]

    caminho_saida = os.path.join(pasta_saida, f"segmentacao_textura_escala_{escala}.png")
    cv2.imwrite(caminho_saida, segmentacao)
    return segmentacao

def gerar_heatmap_final(respostas, pasta_saida, escala):
    combinacao = np.mean(respostas, axis=0)
    combinacao_norm = cv2.normalize(combinacao, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_final = cv2.applyColorMap(combinacao_norm, cv2.COLORMAP_JET)
    caminho_saida = os.path.join(pasta_saida, f"escala_{escala}", "heatmap_final.png")
    cv2.imwrite(caminho_saida, heatmap_final)

def gerar_comparativo(imagens_rotuladas, pasta_saida, nome_saida, referencia_shape):
    largura_img, altura_img = referencia_shape
    largura_total = largura_img * len(imagens_rotuladas)
    comparativo = np.full((altura_img + 40, largura_total, 3), 255, dtype=np.uint8)

    x_offset = 0
    for img, label in imagens_rotuladas:
        comparativo[40:40+altura_img, x_offset:x_offset+largura_img] = img
        cv2.putText(comparativo, label, (x_offset + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        x_offset += largura_img

    cv2.imwrite(os.path.join(pasta_saida, nome_saida), comparativo)

def main():
    if len(sys.argv) == 2 and sys.argv[1] == "clear":
        limpar_resultados()
        sys.exit()

    if len(sys.argv) < 2:
        print("Uso: python3 script.py nome_da_imagem.jpg")
        print("Ou:  python3 script.py clear")
        sys.exit()

    caminho_imagem = sys.argv[1]
    imagem = carregar_imagem(caminho_imagem)
    pasta_resultados, nome_base = preparar_pasta_resultados(caminho_imagem)

    respostas_por_escala = processar_escalas(imagem, pasta_resultados)

    segmentacoes = []
    for escala, (respostas, imagem_base) in enumerate(respostas_por_escala):
        print(f"\nSegmentando textura - Escala {escala}...")
        rotulos, posicoes, dimensoes = segmentar_textura(respostas)
        segmentacao = salvar_segmentacao(rotulos, posicoes, dimensoes, janela=3, pasta_saida=pasta_resultados, escala=escala)
        segmentacoes.append(segmentacao)
        gerar_heatmap_final(respostas, pasta_resultados, escala)

    referencia_shape = respostas_por_escala[0][1].shape[:2][::-1]  # largura, altura
    print("\nGerando comparativo de heatmaps...")
    heatmaps_finais = [(cv2.resize(cv2.imread(os.path.join(pasta_resultados, f"escala_{i}", "heatmap_dog.png")), referencia_shape), f"Escala {i}") for i in range(len(respostas_por_escala))]
    heatmaps_finais.insert(0, (cv2.resize(imagem, referencia_shape), "Original"))
    gerar_comparativo(heatmaps_finais, pasta_resultados, "comparativo_heatmaps.png", referencia_shape)

    print("Gerando comparativo de segmentações...")
    segmentacoes_resized = [cv2.resize(seg, referencia_shape) for seg in segmentacoes]
    segmentacoes_com_original = [(cv2.resize(imagem, referencia_shape), "Original")] + [(seg, f"Escala {i}") for i, seg in enumerate(segmentacoes_resized)]
    gerar_comparativo(segmentacoes_com_original, pasta_resultados, "comparativo_segmentacoes.png", referencia_shape)

    print("\nProcessamento finalizado.")

if __name__ == "__main__":
    main()
