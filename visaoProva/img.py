import cv2 as tela
# Nome do aluno: Leonardo Tavares Cortarelli


# Resposta para COLOR_BGR2GRAY: Achou a pessoa?
    # Achou a imagem porem a cor do quadrado mudou de verde para preto

# Resposta para COLOR_BGR2HSV: Achou a pessoa?
    # Achou a pessoa com a imagem vermelha

# Resposta para COLOR_BGR2LAB: Achou a pessoa?
    # Achou a pessoa com a imagem roxa 

# Mudar algum parâmetro na rede neural mas manter ela funcionando.
    # Foi mudado scaleFactor=1.1 , minNeighbors=5 , minSize=(20,20)


imagem = tela.imread('visaoProva/img.jpg')

imagemGray = tela.cvtColor(imagem, tela.COLOR_BGR2GRAY)
imagemHsv = tela.cvtColor(imagem, tela.COLOR_BGR2HSV)
imagemLab = tela.cvtColor(imagem, tela.COLOR_BGR2LAB)

# Front face
df = tela.CascadeClassifier(tela.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces = df.detectMultiScale(imagem, scaleFactor=1.05, minNeighbors=7, minSize=(30,30), flags=tela.CASCADE_SCALE_IMAGE)

for (x, y, w, h) in faces:
    tela.rectangle(imagem, (x, y), (x + w, y + h), (0, 128, 0), 7)

tela.imshow(str(len(faces)) + ' face(s) encontrada(s).', imagem)
tela.waitKey(0)
tela.destroyAllWindows()
