import numpy as np
import cv2 as cv


def transformToPoints(bboxes):
    bboxesPuntos = []

    for b in bboxes:
        bboxesPuntos.append(getFormattedPoint(b))

    return bboxesPuntos

def getFormattedPoint(oldFormatRectangle):
    extraespace = 1.5

    # ampliamos area del cuadrado (dismunuimos la x e y del primer punto y aumentamos ancho y alto)
    oldFormatRectangle[0] = oldFormatRectangle[0]*0.99
    oldFormatRectangle[1] = oldFormatRectangle[1]*0.99
    oldFormatRectangle[2] = oldFormatRectangle[2]*1.4
    oldFormatRectangle[3] = oldFormatRectangle[3]*1.4

    point1 = [oldFormatRectangle[0], oldFormatRectangle[1]]
    point2 = [oldFormatRectangle[0] + oldFormatRectangle[2], oldFormatRectangle[1]]
    point3 = [oldFormatRectangle[0] + oldFormatRectangle[2], oldFormatRectangle[1] + oldFormatRectangle[3]]
    point4 = [oldFormatRectangle[0], oldFormatRectangle[1] + oldFormatRectangle[3]]

    return np.array([point1, point2, point3, point4], dtype=np.int32)

def filterSquares(bboxes):
    # Filtramos las bboxes que menos se parezcan a un cuadrado, porque el ratio en un cuadrado ancho/alto es 1
    minRatioSquare = 0.75
    bboxesPuntosFiltered = []
    for b in bboxes:
        if b[2] / b[3] > minRatioSquare and b[3] / b[2] > minRatioSquare:
            bboxesPuntosFiltered.append(getFormattedPoint(b))
    return bboxesPuntosFiltered


mser = cv.MSER_create()


while True:

    # Cargamos la imagen en escala de grises --> 0
    img = cv.imread('train_10_ejemplos/00001.ppm', 1)

    imgCopy = img.copy()

    # Guardamos las Bounding Boxes de la imagen -> (x1,y1) y (x2, y2)
    regions, bboxes = mser.detectRegions(img)

    # hulls son todos los contornos, y convexHull nos da el contorno cerrado de una region convexa
    # con reshape, cambiamos las dimensiones de la matriz sin alterar su contenido
    # hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in filterSquares(bboxes)]

    """
    for h in hulls:
        print (h)
        print ("End hull-----------------")
    """

    cv.polylines(imgCopy, hulls, 1, (0, 255, 0))
    print("Fin rect-------------------")

    cv.imshow('img', imgCopy)

    """
    for b in bboxes:
        print (b)
        print ("Fin rect-------------------")
    """



    # Cerramos la ventana presionando escape
    if cv.waitKey(5) == 27:
        break

cv.destroyAllWindows()









