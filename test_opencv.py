import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar imagem em tons de cinza
imagem = cv2.imread('anjo.jpeg', cv2.IMREAD_GRAYSCALE)

if imagem is None:
    print("Erro ao carregar a imagem. Certifique-se de que 'anjo.jpeg' está no mesmo diretório.")
    exit()

# Mostrar imagem com OpenCV
cv2.imshow('Imagem original', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Mostrar imagem com matplotlib
plt.imshow(imagem, cmap='gray')
plt.title("Imagem com matplotlib")
plt.axis('off')
plt.show()
