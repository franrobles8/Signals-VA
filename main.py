import numpy as np
import cv2 as cv


def transform_to_points(bboxes):
    bboxesPuntos = []

    for b in bboxes:
        bboxesPuntos.append(get_formatted_point(b))

    return bboxesPuntos

def get_formatted_point(oldFormatRectangle):

    # Creamos los puntos con los datos obtenidos de los bboxes en formato [x ,y , ancho, alto]

    coord_x = oldFormatRectangle[0]
    coord_y = oldFormatRectangle[1]
    ancho = oldFormatRectangle[2]
    alto = oldFormatRectangle[3]

    point1 = [coord_x, coord_y]
    point2 = [coord_x + ancho, coord_y]
    point3 = [coord_x + ancho, coord_y + alto]
    point4 = [coord_x, coord_y + alto]

    # ampliamos area del cuadrado (disminuimos la x e y del primer punto y aumentamos ancho y alto)
    # la coordenada y es positiva hacia abajo (al reves de como suele ser normalmente)

    ampliacion = ancho/5 # expandimos el bbox 1/5 de su tamanio original

    # punto superior izquierdo
    point1[0] = point1[0] - ampliacion
    point1[1] = point1[1] - ampliacion

    # punto superior derecho
    point2[0] = point2[0] + ampliacion
    point2[1] = point2[1] - ampliacion

    # punto inferior derecho
    point3[0] = point3[0] + ampliacion
    point3[1] = point3[1] + ampliacion

    # punto inferior izquierdo
    point4[0] = point4[0] - ampliacion
    point4[1] = point4[1] + ampliacion

    return np.array([point1, point2, point3, point4], dtype=np.int32)

def filter_squares(bboxes):

    # Filtramos las bboxes que menos se parezcan a un cuadrado, porque el ratio en un cuadrado ancho/alto es 1

    min_ratio_square = 0.75
    bboxes_puntos_filtered = []

    for b in bboxes:
        if b[2] / b[3] > min_ratio_square and b[3] / b[2] > min_ratio_square:
            bboxes_puntos_filtered.append(get_formatted_point(b))

    return bboxes_puntos_filtered


# mser = cv.MSER_create()

# Orden de los parametros del constructor
#       _delta,_min_area,_max_area,
#       _max_variation, _min_diversity,
#       _max_evolution, _area_threshold,
#       _min_margin, _edge_blur_size
mser = cv.MSER_create(3, 100, 2000, 0.3, 1.0, 200, 1.01, 0.003, 0)

while True:

    # Cargamos la imagen en escala de grises
    img = cv.imread('train_10_ejemplos/00001.ppm', cv.IMREAD_GRAYSCALE)

    imgCopy = img.copy()

    # Guardamos las Bounding Boxes de la imagen -> (x1,y1) y ancho y alto
    regions, bboxes = mser.detectRegions(imgCopy)

    # hulls son todos los contornos, y convexHull nos da el contorno cerrado de una region convexa
    # con reshape, cambiamos las dimensiones de la matriz sin alterar su contenido
    # hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in filter_squares(bboxes)]

    cv.polylines(imgCopy, hulls, 1, (0, 255, 0))

    cv.imshow('img', imgCopy)

    # Cerramos la ventana presionando escape
    if cv.waitKey(5) == 27:
        break

cv.destroyAllWindows()









