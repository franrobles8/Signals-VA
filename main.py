import numpy as np
import cv2 as cv
import math
import os
import sys
import getopt
import shutil

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

    ampliacion = ancho*(1/3)# expandimos el bbox 1/3 de su tamanio original

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



def save_rois(bboxes, img, image_route):

    image_route_arr = image_route.split("/")
    image_route = image_route_arr[len(image_route_arr) - 1]

    for b in bboxes:
        if (b[0, 0]) >= 0 & (b[0, 1] >= 0) and (b[1, 0] >= 0) and (b[1, 1] >= 0) and (b[2, 0] >= 0) & (b[2, 1] >= 0) and (b[3, 0] >= 0) & (b[3, 1] >= 0):
            roi = img[b[0, 1]:b[3, 1], b[0, 0]:b[2, 0]]
            cv.imwrite("./rois/" + image_route + ";" + str(b[0, 0]) + ";" + str(b[0, 1]) + ";" + str(b[2, 0]) + ";" + str(b[2, 1]) + "__.jpg", roi)


def resize_img_25_25(img):
    return cv.resize(img, (25, 25))


def create_mask(img, lower_bound, upper_bound,lower_bound2,upper_bound2):
    mask =cv.add(cv.inRange(img, lower_bound, upper_bound),(cv.inRange(img, lower_bound2, upper_bound2)))
    
    return mask

               
def correlate_masks(mask1, mask2):
    # multiplicamos todos los pixeles de las dos matrices
    
    correlation_matrix = np.multiply(mask1, mask2)
    

    # Hallamos el numero de pixeles blancos en la mascara media

    n_pixeles_blancos_m_media = 0

    for idx, row in enumerate(mask2):
        for idy, elem in enumerate(row):
            if mask2[idx, idy] > 0:
                n_pixeles_blancos_m_media += 1

    # Hallamos el numero de pixeles que coinciden en las dos mascaras

    correlation = 0
    
    for idx, row in enumerate(correlation_matrix):
        for idy, elem in enumerate(row):
            if (correlation_matrix[idx][idy]>0):
                correlation += 1
            
    
    
    
    return correlation/n_pixeles_blancos_m_media*100
    

def filter_black_white(old_mask, umbral):
    #filtra los valores de la mascara para que sean o 0 o 255
    new_mask = []

    for row in old_mask:
        new_row = []
        for pixel in row:
            if pixel <= umbral:
                new_row.append(0)
            else:
                new_row.append(255)
        new_mask.append(new_row)
    return np.array(new_mask, np.float32)


def write_signal_to_results(str_signal):
    #escribe el resultado 
    f = open("resultado.txt", "a+")
    f.write(str_signal + "\n")
    f.close()

def train(train_directory):
    
    """
     Punto 2: 
    Utilizar el espacio de color HSV para localizar los píxeles que sean de color rojo. Creacion de mascaras
    para cada tipo de señal (prohibicion, peligro y stop)
    """
    mask_prohibicion = np.zeros((25, 25), dtype=np.int32)
            
    mask_peligro = np.zeros((25, 25), dtype=np.int32)
            
    mask_stop = np.zeros((25, 25), dtype=np.int32)
    n_prohibidos = 0
    n_peligros = 0
    n_stops = 0
    masks = []
    
    

    lower_red = np.array([0, 180, 90])
    upper_red = np.array([12, 200, 150])
    lower_red2 = np.array([240, 180, 90])
    upper_red2 = np.array([256, 200, 150])

    
    prohibiciones =[ "0", "1", "2", "3", "4", "5", "7", "8", "9", "10", "15", "16"]
    peligros =["11", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"]
    stops = ["14"]
    with open(train_directory + "/gt.txt", "r") as ObjFichero:        
        for line in ObjFichero:
            if(line!="\n"):
                linea=line.split(";")
                imagen=linea[0]
                x1=int(linea[1])
                y1=int(linea[2])
                x2=int(linea[3])
                y2=int(linea[4])
                tipo=str(int(linea[5]))
               
                
                img_path_directory = train_directory + "/" + imagen
               
                if (tipo in prohibiciones):
                    # Creamos la mascara de señal de prohibicion (detectamos el rojo)
                    img_intermedia_prohibicion = cv.imread(img_path_directory, 1)
                    img_intermedia_prohibicion_copy=img_intermedia_prohibicion[y1:y2, x1:x2]
                    img_intermedia_prohibicion_copy = cv.cvtColor(img_intermedia_prohibicion_copy, cv.COLOR_BGR2HSV)
                    img_intermedia_prohibicion_copy = resize_img_25_25(img_intermedia_prohibicion_copy)
                    mask_prohibicion = np.add(mask_prohibicion, create_mask(img_intermedia_prohibicion_copy, lower_red, upper_red,lower_red2,upper_red2))               
                    n_prohibidos += 1
                if (tipo in peligros):
                    # Creamos la mascara de señal de peligro (detectamos el rojo)
                    # HSV [Hue, Sat, Value]
                    img_intermedia_peligro = cv.imread(img_path_directory, 1)
                    
                    img_intermedia_peligro_copy=img_intermedia_peligro[y1:y2, x1:x2] 
                    
                    img_intermedia_peligro_copy = cv.cvtColor(img_intermedia_peligro_copy, cv.COLOR_BGR2HSV)
                    img_intermedia_peligro_copy = resize_img_25_25(img_intermedia_peligro_copy)
                    mask_peligro = np.add(mask_peligro, create_mask(img_intermedia_peligro_copy, lower_red, upper_red,lower_red2,upper_red2))
                    n_peligros += 1
                if (tipo in stops):   
                    # Creamos la mascara de señal de stop (detectamos el rojo)
                    img_intermedia_stop = cv.imread(img_path_directory, 1)
                    img_intermedia_stop_copy=img_intermedia_stop[y1:y2, x1:x2] 
                    img_intermedia_stop_copy = cv.cvtColor(img_intermedia_stop_copy, cv.COLOR_BGR2HSV)
                    img_intermedia_stop_copy = resize_img_25_25(img_intermedia_stop_copy)
                    mask_stop = np.add(mask_stop, create_mask(img_intermedia_stop_copy, lower_red, upper_red,lower_red2,upper_red2))
                    n_stops += 1
           
        mask_prohibicion = (np.divide(mask_prohibicion, n_prohibidos))  
        
        mask_prohibicion = filter_black_white(mask_prohibicion, 2)
        mask_peligro = np.divide(mask_peligro, n_peligros)
        mask_peligro = filter_black_white(mask_peligro, 2)
    
        mask_stop = np.divide(mask_stop, n_stops)
        mask_stop = filter_black_white(mask_stop, 2)
        
        masks.append(mask_prohibicion)
        masks.append(mask_peligro)
        masks.append(mask_stop)
        return masks


def crear_fichero_restultado():
    # Creamos un nuevo fichero resultado.txt
    f = open("resultado.txt", "w")
    f.close()


def main():

    """
    Referencias: https://www.kerneldev.com/2018/09/05/getopt-command-line-arguments-in-python/
    """
    try:
        lista_opciones, args = getopt.getopt(sys.argv[1:], '', ['train_path=', 'test_path=', 'detector='])
    except getopt.GetoptError as error:
        print(error)
        sys.exit(2)

    if len(lista_opciones) != 3:
        print("Error en el numero de argumentos")
        print("Los argumentos deben ser de la forma: python main.py --train_path /home/usuario/train --test_path /home/usuario/test --detector detector ")
        sys.exit(2)

    for opcion, valor_opcion in lista_opciones:
        if opcion == '--train_path':
            TRAIN_PATH = valor_opcion
        elif opcion == '--test_path':
            TEST_PATH = valor_opcion
        elif opcion == '--detector':
            DETECTOR = valor_opcion


    
    # Creamos el fichero resultado siempre, para borrar si existiera una version anterior
    crear_fichero_restultado()

    # Eliminamos el directorio rois si existe
    if os.path.exists('rois'):
        shutil.rmtree('rois')

    # Creamos un directorio temporal rois
    if not os.path.exists('rois'):
        os.makedirs('rois')

    
    # Orden de los parametros del constructor
    #       _delta,_min_area,_max_area,
    #       _max_variation, _min_diversity,
    #       _max_evolution, _area_threshold,
    #       _min_margin, _edge_blur_size
    mser = cv.MSER_create(5, 100, 2000, 0.02, 0.5, 200, 1.01, 0.003, 0)
    masks = train(TRAIN_PATH)
    image_clasification_finished = False
    
    while not image_clasification_finished:

        extensions = ['jpg', 'png', 'bmp', 'jpeg', 'ppm']
        file_names = [file for file in os.listdir(TEST_PATH) if not file.endswith(".DS_Store") and any(file.endswith(extension) for extension in extensions)]
    
        img_list = []
    
        for fi in file_names:
            img_list.append(TEST_PATH + "/" + fi)
        
    
        for image in img_list:
            print(image)
            # Cargamos la imagen en color
            img = cv.imread(image, 1)

            # Preprocesamos la imagen y restamos al canal rojo, el verde y el azul, para que evite bboxes que no tengan rojo

            canal_azul = np.array(img[:, :, 0])
            canal_verde = np.array(img[:, :, 1])
            canal_rojo = np.array(img[:, :, 2])

            imagen_sin_azul = cv.absdiff(canal_rojo, canal_azul)
            imagen_rojos = cv.absdiff(imagen_sin_azul, canal_verde)
            
    
            
    
            # Guardamos las Bounding Boxes de la imagen -> (x1,y1) y ancho y alto
            regions, bboxes = mser.detectRegions(imagen_rojos)
    
            # hulls son todos los contornos, y convexHull nos da el contorno cerrado de una region convexa
            # con reshape, cambiamos las dimensiones de la matriz sin alterar su contenido
    
            filtered_bboxes = filter_squares(bboxes)
    
            # guardamos los rois (recortes) detectados en una carpeta
            save_rois(filtered_bboxes, img, image)
    
        """
        Punto 3: 
        Deteccion mediante correlacion de máscaras
        """
    
       
    
        lower_red = np.array([0, 75, 50])
        upper_red = np.array([12, 255, 255])
        lower_red2 = np.array([240, 75, 50])
        upper_red2 = np.array([256, 255, 255])
    
        
    
        img_path_directory = "./rois/"
        extensions = ['jpg', 'png', 'bmp', 'jpeg', 'ppm']
        file_names = [file for file in os.listdir(img_path_directory) if any(file.endswith(extension) for extension in extensions)]
    
        # correlamos cada recorte
    
        for im in file_names:
            print(img_path_directory + im)
            nombre_arr = im.split("__")
            nombre_img = nombre_arr[0]
    
    
            signal_img = cv.imread(img_path_directory + im, 1)
    
            signal_img = cv.cvtColor(signal_img, cv.COLOR_BGR2HSV)
            signal_img = resize_img_25_25(signal_img)
            signal_mask_prohibicion = create_mask(signal_img, lower_red, upper_red,lower_red2,upper_red2)
    
            signal_mask_peligro = create_mask(signal_img, lower_red, upper_red,lower_red2,upper_red2)
            signal_mask_stop = create_mask(signal_img, lower_red, upper_red,lower_red2,upper_red2)

            # Correlamos con señal de prohibicion
            corr_prohibicion = correlate_masks(signal_mask_prohibicion, masks[0])
            corr_peligro = correlate_masks(signal_mask_peligro, masks[1])
            corr_stop = correlate_masks(signal_mask_stop, masks[2])


            # tipos de señal: 1 => prohibicion, 2 => peligro, 3 => stop
            if (corr_prohibicion > corr_peligro) & (corr_prohibicion > corr_stop):
                if corr_prohibicion > 24:
                    write_signal_to_results(str(nombre_img) + ";" + "1" + ";" + str(corr_prohibicion))
            elif (corr_peligro > corr_prohibicion) & (corr_peligro > corr_stop):
                if corr_peligro > 24:
                    write_signal_to_results(str(nombre_img) + ";" + "2" + ";" + str(corr_peligro))
            elif (corr_stop > corr_prohibicion) & (corr_stop > corr_peligro):
                if corr_stop > 24:
                    write_signal_to_results(str(nombre_img) + ";" + "3" + ";" + str(corr_stop))
    
            print(nombre_img + "(prohibicion): " + str(corr_prohibicion))
            print(nombre_img + "(peligro): " + str(corr_peligro))
            print(nombre_img + "(stop): " + str(corr_stop))
            print("----------")

        # Eliminamos el directorio rois si existe
        if os.path.exists('rois'):
            shutil.rmtree('rois')

        print("Resultados obtenidos -> resultado.txt")
           
    
        # Cerramos la ventana presionando escape
        
        if cv.waitKey(5) == 27:
            break
        
        image_clasification_finished = True





       
    cv.destroyAllWindows()

 


if __name__ == "__main__":
    main()









