import cv2

imagem = cv2.imread('./Visao/joker.jpg')
#cv2.imshow("minha umagem", imagem)
cv2.waitKey(0)

for y in range(0, imagem.shape[0]):
    for x in range(0, imagem.shape[1]):
        imagem[y, x] = (255,0,0)

cv2.imshow("Imagem modificada", imagem)

imagem = cv2.imread('joker.jpg')
cv2.waitKey(0)

