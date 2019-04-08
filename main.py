import numpy as np
import cv2 as cv
import math
import os
import sys
import getopt
import shutil


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

    ampliacion = ancho*(1/6)# expandimos el bbox 1/8 de su tamanio original

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

def save_rois(bboxes, img, image_route):

    image_route_arr = image_route.split("/")
    image_route = image_route_arr[len(image_route_arr) - 1]

    for b in bboxes:
        if (b[0, 0]) >= 0 & (b[0, 1] >= 0) and (b[1, 0] >= 0) and (b[1, 1] >= 0) and (b[2, 0] >= 0) & (b[2, 1] >= 0) and (b[3, 0] >= 0) & (b[3, 1] >= 0):
            roi = img[b[0, 1]:b[3, 1], b[0, 0]:b[2, 0]]
            cv.imwrite("./rois/" + image_route + ";" + str(b[0, 0]) + ";" + str(b[0, 1]) + ";" + str(b[2, 0]) + ";" + str(b[2, 1]) + "__.jpg", roi)


def resize_img_25_25(img):
    return cv.resize(img, (25, 25))


def create_mask(img, lower_bound, upper_bound):
    # return cv.bitwise_not(cv.inRange(img, lower_bound, upper_bound))
    mask = cv.inRange(img, lower_bound, upper_bound)
    return mask

               
def correlate_masks(mask1, mask2):
    # size = mask_size(mask1)
    # print(size)
    # multiplicamos todos los pixeles de las dos matrices y sumamos
    """"""
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
            
    print(correlation)
    # print(size)
    
    return correlation/n_pixeles_blancos_m_media#(25*25)*100
    #return correlation/n_pixeles_blancos_m_media

def filter_black_white(old_mask, umbral):
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

def get_mean_image_masks():
    directory_path_directory = "./train_recortadas/"
    file_names = [file for file in os.listdir(directory_path_directory) if not file.endswith(".DS_Store")]
    prohibiciones = ["00", "01", "02", "03", "04", "05", "07", "08", "09", "10", "15", "16"]
    peligros = ["11", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"]
    stops = ["14"]

    mean_image_prohibicion = np.zeros((25, 25, 3), dtype=np.int32)
    mean_image_peligro = np.zeros((25, 25, 3), dtype=np.int32)
    mean_image_stop = np.zeros((25, 25, 3), dtype=np.int32)

    n_prohibiciones = 0
    n_peligros = 0
    n_stops = 0

    for fi in file_names:
        img_path_directory = "./train_recortadas/"+fi
        file_images = [file for file in os.listdir(img_path_directory)]

        for im in file_images:

            if (fi in prohibiciones):
                img_prohibicion = cv.imread(img_path_directory + "/" + im)
                img_prohibicion = resize_img_25_25(img_prohibicion)
                mean_image_prohibicion = np.add(mean_image_prohibicion, img_prohibicion)
                n_prohibiciones += 1

            if (fi in peligros):
                img_peligro = cv.imread(img_path_directory + "/" + im)
                img_peligro = resize_img_25_25(img_peligro)
                mean_image_peligro = np.add(mean_image_peligro, img_peligro)
                n_peligros += 1
            if (fi in stops):
                img_stop = cv.imread(img_path_directory + "/" + im)
                img_stop = resize_img_25_25(img_stop)
                mean_image_stop = np.add(mean_image_stop, img_stop)
                n_stops += 1

    cv.imshow("mean prohibicion", mean_image_prohibicion)
    cv.imshow("mean peligro", mean_image_peligro)
    cv.imshow("mean stop", mean_image_stop)

def write_signal_to_results(str_signal):
    f = open("resultado.txt", "a+")
    f.write(str_signal + "\n")
    f.close()

def train(train_directory):
    """
    f = open ('./train/gt.txt','r')
    mensaje = f.read()
    print(mensaje)
    f.close()
    splited=mensaje.split()
    """
    """
    with open("./train/gt.txt", "r") as ObjFichero:



        for line in ObjFichero:
            print(line)
    """
    """
     Punto 2: 
    Utilizar el espacio de color HSV para localizar los píxeles que sean de color rojo. Creacion de mascaras
    para cada tipo de señal (prohibicion, peligro y stop)
   
    directory_path_directory = "./train_recortadas/"
    file_names = [file for file in os.listdir(directory_path_directory) if not file.endswith(".DS_Store")] """
    mask_prohibicion = np.zeros((25, 25), dtype=np.int32)
            
    mask_peligro = np.zeros((25, 25), dtype=np.int32)
            
    mask_stop = np.zeros((25, 25), dtype=np.int32)
    n_prohibidos = 0
    n_peligros = 0
    n_stops = 0
    masks = []
    """"""
    lower_red_prohibicion = np.array([0, 190, 100])
    upper_red_prohibicion = np.array([10, 255, 255])

    lower_red_peligro = np.array([0, 180, 0])
    upper_red_peligro = np.array([10, 255, 255])

    lower_red_stop = np.array([0, 132, 0])
    upper_red_stop = np.array([11, 255, 255])
    prohibiciones =[ "0", "1", "2", "3", "4", "5", "7", "8", "9", "10", "15", "16"]
    peligros =["11", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"]
    stops = ["14"]
    with open(train_directory + "/gt.txt", "r") as ObjFichero:

        

        for line in ObjFichero:
            print(line)
            if((line=="")):
                print("sdsjddj")
            if(line!="\n"):
                linea=line.split(";")
                imagen=linea[0]
                x1=int(linea[1])
                y1=int(linea[2])
                x2=int(linea[3])
                y2=int(linea[4])
                tipo=str(int(linea[5]))
                prohibiciones =[ "0", "1", "2", "3", "4", "5", "7", "8", "9", "10", "15", "16"]
                peligros =["11", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"]
                stops = ["14"]
                
                
                
                
                
            
                #for fi in file_names:
                img_path_directory = train_directory + "/" + imagen
                #file_images = [file for file in os.listdir(img_path_directory)]
        
                #for im in file_images:
                
                e1=peligros[0]
                esta=(tipo ==peligros[0])
                if (tipo in prohibiciones):
                    # Creamos la mascara de señal de prohibicion (detectamos el rojo)
                    img_intermedia_prohibicion = cv.imread(img_path_directory )
                    
                    img_intermedia_prohibicion_copy=img_intermedia_prohibicion[y1:y2, x1:x2] 
                    
                    img_intermedia_prohibicion_copy = cv.cvtColor(img_intermedia_prohibicion_copy, cv.COLOR_BGR2HSV)
                    img_intermedia_prohibicion_copy = resize_img_25_25(img_intermedia_prohibicion_copy)
                    mask_prohibicion = np.add(mask_prohibicion, create_mask(img_intermedia_prohibicion_copy, lower_red_prohibicion, upper_red_prohibicion))               
                    n_prohibidos += 1
                if (tipo in peligros):
                    # Creamos la mascara de señal de peligro (detectamos el rojo)
                    # HSV [Hue, Sat, Value]
                    img_intermedia_peligro = cv.imread(img_path_directory)
                    
                    img_intermedia_peligro_copy=img_intermedia_peligro[y1:y2, x1:x2] 
                    
                    img_intermedia_peligro_copy = cv.cvtColor(img_intermedia_peligro_copy, cv.COLOR_BGR2HSV)
                    img_intermedia_peligro_copy = resize_img_25_25(img_intermedia_peligro_copy)
                    mask_peligro = np.add(mask_peligro, create_mask(img_intermedia_peligro_copy, lower_red_peligro, upper_red_peligro))
                    if(tipo=="11"):
                        patata=create_mask(img_intermedia_peligro_copy, lower_red_peligro, upper_red_peligro)
                    n_peligros += 1
                if (tipo in stops):   
                    # Creamos la mascara de señal de stop (detectamos el rojo)
                    img_intermedia_stop = cv.imread(img_path_directory )
                    img_intermedia_stop_copy=img_intermedia_stop[y1:y2, x1:x2] 
                    img_intermedia_stop_copy = cv.cvtColor(img_intermedia_stop_copy, cv.COLOR_BGR2HSV)
                    img_intermedia_stop_copy = resize_img_25_25(img_intermedia_stop_copy)
                    mask_stop = np.add(mask_stop, create_mask(img_intermedia_stop_copy, lower_red_stop, upper_red_stop))
                    n_stops += 1
           
        mask_prohibicion = (np.divide(mask_prohibicion, n_prohibidos))  
        
        mask_prohibicion = filter_black_white(mask_prohibicion, 2)
        """
        cv.imshow("patata", patata.astype(np.uint8))
        
        
        cv.imshow("zadasdaddass",mask_peligro)
        """
        mask_peligro = np.divide(mask_peligro, n_peligros)
        mask_peligro = filter_black_white(mask_peligro, 2)
    
        mask_stop = np.divide(mask_stop, n_stops)
        mask_stop = filter_black_white(mask_stop, 2)
    
        """
        mask_stop = filter_black_white(mask_stop, 2).astype(np.uint8)"""
        cv.imshow("mask intermedia prohibicion", mask_prohibicion.astype(np.uint8))
        cv.imshow("mask intermedia peligro", mask_peligro.astype(np.uint8))
        cv.imshow("mask intermedia stop", mask_stop.astype(np.uint8))
        
        
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

    # Eliminamos la carpeta rois si existe
    if os.path.exists('rois'):
        shutil.rmtree('rois')

    # Creamos un fichero temporal rois
    if not os.path.exists('rois'):
        os.makedirs('rois')
     """
    TRAIN_PATH="./train"
    TEST_PATH="./test"
    # Orden de los parametros del constructor
    #       _delta,_min_area,_max_area,
    #       _max_variation, _min_diversity,
    #       _max_evolution, _area_threshold,
    #       _min_margin, _edge_blur_size
    mser = cv.MSER_create(0, 100, 2000, 0.05, 1.0, 200, 1.01, 0.003, 0)
    masks = train(TRAIN_PATH)
    image_clasification_finished = False
    """"""
    while not image_clasification_finished:
    
        # get_mean_image_masks()
    
        file_names = [file for file in os.listdir(TEST_PATH) if not file.endswith(".DS_Store")]
    
        img_list = []
    
        for fi in file_names:
            img_list.append(TEST_PATH + "/" + fi)
        
    
        for image in img_list:
            print(image)
            # Cargamos la imagen en color
            img = cv.imread(image, 1)
    
            imgGray =img #cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
            # Guardamos las Bounding Boxes de la imagen -> (x1,y1) y ancho y alto
            regions, bboxes = mser.detectRegions(imgGray)
    
            # hulls son todos los contornos, y convexHull nos da el contorno cerrado de una region convexa
            # con reshape, cambiamos las dimensiones de la matriz sin alterar su contenido
    
            filtered_bboxes = filter_squares(bboxes)
            hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in filtered_bboxes]
            cv.polylines(imgGray, hulls, 1, (0, 255, 0))
            
            cv.imwrite("./poli/" + image.split("/")[2] , imgGray)
            cv.imshow("imfff",imgGray)
            
    
            # guardamos los rois (recortes) detectados en una carpeta
            save_rois(filtered_bboxes, img, image)
    
            """
            Punto 3: 
            Deteccion mediante correlacion de máscaras
            """
    
        lower_red_prohibicion = np.array([0, 110, 0])
        upper_red_prohibicion = np.array([10, 255, 255])
    
        lower_red_peligro = np.array([0, 0, 0])
        upper_red_peligro = np.array([10, 255, 255])
    
        lower_red_stop = np.array([0, 132, 0])
        upper_red_stop = np.array([11, 255, 255])
    
        img_path_directory = "./rois/"
        extensions = ['jpg', 'png', 'bmp', 'jpeg', 'ppm']
        file_names = [file for file in os.listdir(img_path_directory) if
                      any(file.endswith(extension) for extension in extensions)]
    
        # correlamos cada recorte
    
        for im in file_names:
            print(img_path_directory + im)
            nombre_arr = im.split("__")
            nombre_img = nombre_arr[0]
    
    
            signal_img = cv.imread(img_path_directory + im, 1)
    
            signal_img = cv.cvtColor(signal_img, cv.COLOR_BGR2HSV)
            signal_img = resize_img_25_25(signal_img)
    
            signal_mask_prohibicion = create_mask(signal_img, lower_red_prohibicion, upper_red_prohibicion)
    
            signal_mask_peligro = create_mask(signal_img, lower_red_peligro, upper_red_peligro)
            signal_mask_stop = create_mask(signal_img, lower_red_stop, upper_red_stop)
    
    
            cv.imshow('signal_mask_peligro', signal_mask_peligro.astype(np.uint8))
            cv.imshow('signal_mask_stop', signal_mask_stop.astype(np.uint8))
    
            # cv.imshow('signal_mask_prohib', signal_mask_prohibicion)
    
            # Correlamos con señal de prohibicion
            corr_prohibicion = correlate_masks(signal_mask_prohibicion, masks[0])
            corr_peligro = correlate_masks(signal_mask_peligro, masks[1])
            corr_stop = correlate_masks(signal_mask_stop, masks[2])
            if(im=="00400.jpg;1241;480;1301;542__.jpg"):
                print("yaaa")
            if (corr_prohibicion > corr_peligro) & (corr_prohibicion > corr_stop):
                write_signal_to_results(str(nombre_img) + ";" + "0" + ";" + str(corr_prohibicion))
            elif (corr_peligro > corr_prohibicion) & (corr_peligro > corr_stop):
                write_signal_to_results(str(nombre_img) + ";" + "1" + ";" + str(corr_prohibicion))
            elif (corr_stop > corr_prohibicion) & (corr_stop > corr_peligro):
                write_signal_to_results(str(nombre_img) + ";" + "2" + ";" + str(corr_prohibicion))
    
            print(nombre_img + "(prohibicion): " + str(corr_prohibicion))
            print(nombre_img + "(peligro): " + str(corr_peligro))
            print(nombre_img + "(stop): " + str(corr_stop))
            print("----------")
    
            # cv.imshow('img', imgGray)
    
            # Cerramos la ventana presionando escape
        """"""
        if cv.waitKey(5) == 27:
            break
        
        image_clasification_finished = True





        """
        # Cargamos la imagen en color
        img = cv.imread('train/00000.ppm', 1)

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
        """
        Punto 3: 
        Deteccion mediante correlacion de máscaras
        """
        """
        lower_red_prohibicion = np.array([0, 110, 0])
        upper_red_prohibicion = np.array([10, 255, 255])

        lower_red_peligro = np.array([0, 180, 0])
        upper_red_peligro = np.array([4, 255, 255])

        lower_red_stop = np.array([0, 132, 0])
        upper_red_stop = np.array([11, 255, 255])
        img_path_directory = "./recortes_prueba/"
        extensions = ['jpg', 'png', 'bmp', 'jpeg', 'ppm']
        file_names = [file for file in os.listdir(img_path_directory) if
                      any(file.endswith(extension) for extension in extensions)]
        print(file_names)
        masks = train(TRAIN_PATH)
        signal_img1 = cv.imread(img_path_directory + "00000.ppm", 1)

        # correlamos cada recorte

        for im in file_names:
            print(img_path_directory + im)

            signal_img = cv.imread(img_path_directory + im, 1)

            signal_img = cv.cvtColor(signal_img, cv.COLOR_BGR2HSV)
            signal_img = resize_img_25_25(signal_img)

            signal_mask_prohibicion = create_mask(signal_img, lower_red_prohibicion, upper_red_prohibicion)

            signal_mask_peligro = create_mask(signal_img, lower_red_peligro, upper_red_peligro)
            signal_mask_stop = create_mask(signal_img, lower_red_stop, upper_red_stop)
            if (im == "00016.ppm"):
                cv.imshow('signal_mask_prohibicion', signal_mask_prohibicion.astype(np.uint8))
            
            cv.imshow('real', signal_img)

            cv.imshow('default_peligro', masks[1].astype(np.uint8))
            cv.imshow('default_peligro1', masks[0].astype(np.uint8))
            cv.imshow('default_peligro2', masks[2].astype(np.uint8))
            cv.imshow('signal_mask_prohibicion', signal_mask_prohibicion.astype(np.uint8)) 
            

            cv.imshow('signal_mask_peligro', signal_mask_peligro.astype(np.uint8))
            cv.imshow('signal_mask_stop', signal_mask_stop.astype(np.uint8))

            """"""

            # cv.imshow('signal_mask_prohib', signal_mask_prohibicion)

            # Correlamos con señal de prohibicion
            corr_prohibicion = correlate_masks(signal_mask_prohibicion, masks[0])
            corr_peligro = correlate_masks(signal_mask_peligro, masks[1])
            corr_stop = correlate_masks(signal_mask_stop, masks[2])

            if (corr_prohibicion > corr_peligro) & (corr_prohibicion > corr_stop):
                write_signal_to_results(str(im) + ";" + "0" + ";" + str(corr_prohibicion))
            elif (corr_peligro > corr_prohibicion) & (corr_peligro > corr_stop):
                write_signal_to_results(str(im) + ";" + "1" + ";" + str(corr_prohibicion))
            elif (corr_stop > corr_prohibicion) & (corr_stop > corr_peligro):
                write_signal_to_results(str(im) + ";" + "2" + ";" + str(corr_prohibicion))

            print(im + "(prohibicion): " + str(corr_prohibicion))
            print(im + "(peligro): " + str(corr_peligro))
            print(im + "(stop): " + str(corr_stop))
            print("----------")

        # cv.imshow('img', imgGray)

        # Cerramos la ventana presionando escape

        if cv.waitKey(5) == 27:
            break
        """
    cv.destroyAllWindows()

    """"""


if __name__ == "__main__":
    main()









