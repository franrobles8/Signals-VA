import numpy as np
import cv2 as cv
import math
import os


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

    ampliacion = ancho/8 # expandimos el bbox 1/8 de su tamanio original

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


def distance_between_points(p1_x, p1_y, p2_x, p2_y):
    return math.sqrt((p2_x - p1_x) ** 2 + (p2_y - p1_y) ** 2)


def save_rois(bboxes, img):

    for b in bboxes:
        roi = img[b[0, 1]:b[3, 1], b[1, 0]:b[2, 0]]
        # cv.rectangle(img, (b[0, 0], b[0, 1]), (b[2, 0], b[2, 1]), (0, 255, 0), 1)

    # cv.imshow('marked areas', img)


def resize_img_25_25(img):
    return cv.resize(img, (25, 25))


def create_mask(img, lower_bound, upper_bound):
    # return cv.bitwise_not(cv.inRange(img, lower_bound, upper_bound))
    return cv.inRange(img, lower_bound, upper_bound)

def correlate_masks(mask1, mask2):
    # multiplicamos todos los pixeles de las dos matrices y sumamos
    correlation_matrix = np.multiply(mask1, mask2)
    correlation = 0
    for row in correlation_matrix:
        for elem in row:
            correlation += elem
    return correlation
def train():
    """
     Punto 2: 
    Utilizar el espacio de color HSV para localizar los píxeles que sean de color rojo. Creacion de mascaras
    para cada tipo de señal (prohibicion, peligro y stop)
    """
    directory_path_directory = "./train_recortadas/"
    file_names = [file for file in os.listdir(directory_path_directory) ]
    prohibiciones=[ "00", "01", "02", "03", "04", "05", "07", "08", "09", "10", "15", "16"]
    peligros=["11", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"]
    stops= ["14"]
    for fi in file_names:
        img_path_directory = "./train_recortadas/"+fi
        file_images= [file for file in os.listdir(img_path_directory) ]
        mask_prohibicion=np.zeros( (25,25) )

        mask_peligro=np.zeros( (25,25) )

        mask_stop=np.zeros( (25,25) )
        masks=[]
        for im in file_images:
            # Creamos una mascara para señales con una imagen intermedia
            
           
           
        
            # Pasamos espacio de colores a HSV para detectar rojos
            
           
            
        
        
            if (fi in prohibiciones):
                # Creamos la mascara de señal de prohibicion (detectamos el rojo)
                img_intermedia_prohibicion = cv.imread(img_path_directory +"/"+im)
                img_intermedia_prohibicion_copy = cv.cvtColor(img_intermedia_prohibicion, cv.COLOR_BGR2HSV)
                lower_red_prohibicion = np.array([0, 114, 0], np.uint8)
                upper_red_prohibicion = np.array([255, 255, 255], np.uint8)
                img_intermedia_prohibicion_copy = resize_img_25_25(img_intermedia_prohibicion_copy)
                mask_prohibicion = np.add(mask_prohibicion,create_mask(img_intermedia_prohibicion_copy, lower_red_prohibicion, upper_red_prohibicion))
                
        
            if (fi in peligros):
                # Creamos la mascara de señal de peligro (detectamos el rojo)
                # HSV [Hue, Sat, Value]
                img_intermedia_peligro = cv.imread(img_path_directory +"/"+im)
                img_intermedia_peligro_copy = cv.cvtColor(img_intermedia_peligro, cv.COLOR_BGR2HSV) 
                lower_red_peligro = np.array([0, 130, 0], np.uint8)
                upper_red_peligro = np.array([255, 255, 255], np.uint8)
                img_intermedia_peligro_copy = resize_img_25_25(img_intermedia_peligro_copy)
                mask_peligro =np.add(mask_prohibicion, create_mask(img_intermedia_peligro_copy, lower_red_peligro, upper_red_peligro))
            if (fi in stops):   
                # Creamos la mascara de señal de stop (detectamos el blanco)
                img_intermedia_stop = cv.imread(img_path_directory +"/"+im)
                img_intermedia_stop_copy = cv.cvtColor(img_intermedia_stop, cv.COLOR_BGR2HSV)
                lower_red_stop = np.array([55, 59, 0], np.uint8)
                upper_red_stop = np.array([255, 255, 255], np.uint8)
                img_intermedia_stop_copy = resize_img_25_25(img_intermedia_stop_copy)
                mask_stop =np.add(mask_prohibicion, create_mask(img_intermedia_stop_copy, lower_red_stop, upper_red_stop))
    masks.append(mask_prohibicion)
    masks.append(mask_peligro)
    masks.append(mask_stop)
    return masks
# mser = cv.MSER_create()

# Orden de los parametros del constructor
#       _delta,_min_area,_max_area,
#       _max_variation, _min_diversity,
#       _max_evolution, _area_threshold,
#       _min_margin, _edge_blur_size
mser = cv.MSER_create(2, 100, 2000, 0.05, 1.0, 200, 1.01, 0.003, 0)

while True:

    # Cargamos la imagen en color
    img = cv.imread('train/00000.ppm', 1)

    imgCopy = img.copy()
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


    # Guardamos las Bounding Boxes de la imagen -> (x1,y1) y ancho y alto
    regions, bboxes = mser.detectRegions(imgGray)

    # hulls son todos los contornos, y convexHull nos da el contorno cerrado de una region convexa
    # con reshape, cambiamos las dimensiones de la matriz sin alterar su contenido

    filtered_bboxes = filter_squares(bboxes)
    hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in filtered_bboxes]

    cv.polylines(imgGray, hulls, 1, (0, 255, 0))

    # Guardamos los ROIs detectados, para su posterior filtrado
    # save_rois(filtered_bboxes, imgCopy)

   
   

    # result = cv.bitwise_and(img_intermedia_stop_copy, img_intermedia_stop_copy, mask=mask_stop)
    # cv.imshow('mask', mask_stop)
    # cv.imshow('result', result)

    """
    Punto 3: 
    Deteccion mediante correlacion de máscaras
    """
    upper_red_prohibicion = np.array([255, 255, 255], np.uint8)
    lower_red_prohibicion = np.array([0, 114, 0], np.uint8)
    upper_red_peligro = np.array([255, 255, 255], np.uint8)
    lower_red_peligro = np.array([0, 130, 0], np.uint8)
    upper_red_stop = np.array([255, 255, 255], np.uint8)
    lower_red_stop = np.array([55, 59, 0], np.uint8)
    img_path_directory = "./recortes_prueba/"
    extensions = ['jpg', 'png', 'bmp', 'jpeg', 'ppm']
    file_names = [file for file in os.listdir(img_path_directory) if any(file.endswith(extension) for extension in extensions)]
    print(file_names)
    masks=train()
    # correlamos cada recorte
    for im in file_names:
        print(img_path_directory + im)
        signal_img = cv.imread(img_path_directory + im, 1)
        signal_img = cv.cvtColor(signal_img, cv.COLOR_BGR2HSV)
        signal_img = resize_img_25_25(signal_img)
        signal_mask_prohibicion = create_mask(signal_img, lower_red_prohibicion, upper_red_prohibicion)
        signal_mask_peligro = create_mask(signal_img, lower_red_peligro, upper_red_peligro)
        signal_mask_stop = create_mask(signal_img, lower_red_stop, upper_red_stop)

        cv.imshow('real', signal_img)
        cv.imshow('default_peligro', masks[1])
        cv.imshow('signal_mask_prohibicion', signal_mask_prohibicion)
        cv.imshow('signal_mask_peligro', signal_mask_peligro)
        cv.imshow('signal_mask_stop', signal_mask_stop)


        # cv.imshow('signal_mask_prohib', signal_mask_prohibicion)

        # Correlamos con señal de prohibicion
        corr_prohibicion = correlate_masks(masks[0], signal_mask_prohibicion)
        corr_peligro = correlate_masks(masks[1], signal_mask_peligro)
        corr_stop = correlate_masks(masks[2], signal_mask_stop)
        print(im + "(prohibicion): " + str(corr_prohibicion))
        print(im + "(peligro): " + str(corr_peligro))
        print(im + "(stop): " + str(corr_stop))
        print("----------")



    # cv.imshow('img', imgGray)

    # Cerramos la ventana presionando escape
    if cv.waitKey(5) == 27:
        break

cv.destroyAllWindows()









