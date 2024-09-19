import heapq
from collections import Counter
import numpy as np
import imageio.v2 as io
import matplotlib.pyplot as plt


class huffmanNode:
    def __init__(self, value, freq):
        self.value = value
        self.freq = freq
        self.izq = None
        self.der = None

    # método especial que define el comportamiento del operador menor que (<) para objetos de una clase. Su propósito es
    # permitir la comparación entre dos objetos de dicha clase en función de un atributo específico, en este caso, el atributo frecuencia.
    # Compara el valor del atributo frecuencia de la instancia actual (self.freq) con el valor del mismo atributo en otra
    # instancia (other.freq). Si la frecuencia de la primera instancia es menor, retorna True, de lo contrario, retorna False.
    def __lt__(self, other):
        return self.freq < other.freq


def rgb2gray(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    return 0.2989*R + 0.5870*G + 0.1140*B

def lz77Coding(data, windowSize = 1000):
    compressData = [] 
    i = 0 #indice para recorrer los caracteres de los datos a comprimir
    while i < len(data):
        match = -1 #indica el indice de la primera coincidencia en la ventana
        matchLenght = 0 #almacena la longitud de la coincidencia más larga encontrada

        startWindow = max(0, i - windowSize)

        #el siguiente ciclo recorre desde el inicio de la ventana hasta la posición actual.
        #con este buscamos coincidencias en el historial de datos
        for j in range(startWindow, i):
            k = 0 #contador para comprobar la longitud de la coincidencia

            #comparamos los caracteres de la ventana con los caracteres a partir de i
            #data[j + k] : caracteres siguientes
            #data[i + k] : caracteres de la ventana de busqueda (caracteres pasados)
            while i + k < len(data) and data[j + k] == data[i + k]:
                k += 1
                if k > matchLenght:
                    match = j
                    matchLenght = k

        #si hay coincidencia
        if matchLenght >= 1:
            offset = i - match #desplazamiento entre la posición actual y el inicio de la coincidencia 
            
            if i + matchLenght < len(data): #condición para evitar un error de fuera de rango al terminar de recorrer la secuencia a codificar
                compressData.append((offset, matchLenght, data[i + matchLenght])) 
            else:
                compressData.append((offset, matchLenght, data[i + matchLenght - 1])) #se almacena el desplazamiento, la longitud de la coincidencia y el caracter siguiente a esta 
            
            i += matchLenght + 1 #avanzamos hasta el termino de la coincidencia
            
        else:
            compressData.append((0, 0, data[i])) #si no hay coincidencia unicamente almacenamos el cacterer en curso
            i += 1 
            
    return compressData

def huffmanCoding(lz77List):

    def huffmanTree(frequencies):
        heap = [] #creamos una lista para alamacenar las frecuencias de los simbolos
        #recorremos cada par (clave, valor) del diccionario de frecuencias
        for val, freq in frequencies.items():
            node = huffmanNode(val, freq) #creamos un nuevo nodo de huffman
            heap.append(node) #añadimos el nodo a la lista

        #convertimos la lista en una nueva estructura de dato
        heapq.heapify(heap)

        #hasta que el heap solo tenga un elemento:
        while len(heap) > 1:
            izq = heapq.heappop(heap) #desencolamos el elemento más pequeño del heap
            der = heapq.heappop(heap)
            father = huffmanNode(None, izq.freq + der.freq)
            father.izq = izq
            father.der = der
            heapq.heappush(heap, father) #incorporamos el nuevo arbol creado como "un nuevo valor" en el heap

        return heap[0] #retornamos el unico valor que queda en el heap

    def huffmanCodes(node, actualCode = "", codes = {}):
        if node is None:
            return

        if node.value is not None:
            codes[node.value] = actualCode

        # caso especial, donde solo tenemos la raíz del arbol
        if node.izq is None and node.der is None:
            codes[node.value] = actualCode if actualCode != "" else "0"
            return codes

        huffmanCodes(node.izq, actualCode + "0", codes)
        huffmanCodes(node.der, actualCode + "1", codes)

        return codes

    frequencies = Counter(lz77List)

    tree = huffmanTree(frequencies)
    codes = huffmanCodes(tree)

    encodedHuffman =  ""
    for (ofsset, lenght, value) in lz77List:
        encodedHuffman += codes[(ofsset, lenght, value)]

    return encodedHuffman, codes, frequencies

#proceso de decodificación
def huffmanDecoding(encodedHuffman, codes, frequencies, length = 64):
    reverseCodes ={} #diccionario que almacenara los codigos invertidos, este será util para buscar los símbolos originales a partir de los códigos de huffman

    #iteramos sobre cada clave (que en este caso es una tupla) y valor del diccionario original (codes)
    for k, v in codes.items():
        reverseCodes[v] = k #asignamos v (valor) como clave y k (simbolo original) como su valor asociado

    actualCode = "" #ayudara a acumular los bits de la cadena codificada hasta formar el código de huffman completo
    decoded = [] #lista para almacenar los simbolos decodificados

    #iteramos sobre cada bit de la secuencia resultante de la codificación de huffman
    for bit in encodedHuffman:
        actualCode += bit #en cada iteración se añade un bit al codigo actual, así se va formando de manera incremental un codigo de huffman
        #verificamos si el conjunto de bits representa un simbolo codificado
        if actualCode in reverseCodes:
            symbolTuple = reverseCodes[actualCode]
            offset = int(symbolTuple[0])
            length = int(symbolTuple[1])
            value = float(symbolTuple[2])

            decoded.append((offset, length, value))  #almacenamos cada valor con su respectiva distancia y longitud
            actualCode = "" #vaciamos el codigo actual para empezar a acumular los bits siguientes del código de huffman

    return decoded

def lz77Decoding(compressData):
    output = []

    for offset, length, value in compressData:
        #si el offset es mayor a 0, es decir tenemos una coincidencia
        if offset > 0:
            start = len(output) - offset #copiamos la secuencia desde la ventana previa
            for _ in range(length):
                output.append(output[start])
                start += 1
        output.append(value) #añadimos el codigo nuevo

    #para evitar que se repita el ultimo valor si este no tiene coincidencia previa
    if output[-1] == output[-2]:
        output.pop()

    return output

if __name__ == "__main__":
    img_RGB = io.imread('lenna.png') #carga de la imagen

    # Codificación
    # Paso 1: Preprocesamiento
    imgGray = rgb2gray(img_RGB)
    imgGrayList = imgGray.flatten().tolist() #convertimos la imagen en una lista

    #Paso 2. Codficación
    # lz77
    encodedLz77 = lz77Coding(imgGrayList)

    #Huffman
    encodedHuffman, codes, frequencies = huffmanCoding(encodedLz77)

    # Decodificación
    # Paso 2 inverso: Decodificación
    # Huffman
    decodedHuffman = huffmanDecoding(encodedHuffman, codes, frequencies)

    #LZ77
    decodedLz77 = np.array(lz77Decoding(decodedHuffman))

    constructedImage = decodedLz77.reshape(imgGray.shape[0], imgGray.shape[1]) #convertimos la lista en una matriz nuevamente

    #cv2.imwrite('fotoCompress.jpg', constructedImage)

    # cv2.imshow("Imagen Original", imgGray.astype(np.uint8))
    # cv2.imshow("Imagen Reconstruida", constructedImage.astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    plt.figure()
    plt.imshow(imgGray, cmap = 'gray')
    plt.axis('off')
    plt.show()
    #
    plt.figure()
    plt.imshow(constructedImage, cmap = 'gray')
    plt.axis('off')
    plt.show()