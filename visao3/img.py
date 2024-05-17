import cv2 as tela

imagem = tela.imread('visao2/img.jpg')

# Front face
df = tela.CascadeClassifier(tela.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces = df.detectMultiScale(imagem, scaleFactor=1.05, minNeighbors=7, minSize=(30,30), flags=tela.CASCADE_SCALE_IMAGE)

for (x, y, w, h) in faces:
    tela.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 255), 7)

tela.imshow(str(len(faces)) + ' face(s) encontrada(s).', imagem)
tela.waitKey(0)
tela.destroyAllWindows()
