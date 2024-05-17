import cv2
import numpy as np

def dimencaoQuadrado(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2]
    y2 = boxes[:,1] + boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        suppress = [last]
        for pos in range(0, last):
            j = idxs[pos]
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            overlap = float(w * h) / area[j]

            if overlap > overlapThresh:
                suppress.append(pos)

        idxs = np.delete(idxs, suppress)

    return boxes[pick].astype("int")

# Carregar o classificador treinado
cascade_classifier = cv2.CascadeClassifier('TrabalhoReconhecerGalinha2/myhaar.xml')

# Carregar imagem
imagem = cv2.imread('TrabalhoReconhecerGalinha2/galinhaTeste3.jpg')

imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

detectarGalinha = cascade_classifier.detectMultiScale(imagem_gray, scaleFactor=1.05, minNeighbors=7, minSize=(30, 30))

detectarGalinha = dimencaoQuadrado(detectarGalinha, overlapThresh=0.3)

for (x, y, w, h) in detectarGalinha:
    cv2.rectangle(imagem, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('Detecção de Galinha', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()