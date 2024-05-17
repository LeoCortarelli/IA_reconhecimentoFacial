import cv2 as tela
imagem = tela.imread('visao2/img.jpg')

verde = (0,255,0)
vermelho = (0,0,255)
tela.line(imagem, (0,0), (200,200), verde)
tela.rectangle(imagem, (0,2), (200,300), vermelho)

imageCinza = tela.cvtColor(imagem, tela.COLOR_BGR2GRAY) # DEIXA A IMAGEM CINZA

tela.imshow("Minha imagem", imageCinza)
tela.waitKey(0)