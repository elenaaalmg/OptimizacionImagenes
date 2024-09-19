import heapq
import numpy as np
import imageio.v2 as io
import cv2
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

#proceso de codificación
def preProcess(img):

    def rgb2gray(img):
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        return 0.2989*R + 0.5870*G + 0.1140*B

    def centrality(img):
        imgGray = rgb2gray(img)

        return imgGray - 127

    def split(img):
        imgCentral = centrality(img)
        bolcks = [] #inicializamos una lista vacia para almacenar los bloques de 8x8

        #recorremos la imagen filaxfila en saltos de 8 pixeles
        for i in range(0, imgCentral.shape[0], 8):
            #recorremos la imagen columnaxcolumna en saltos de 8 pixeles
            for j in range(0, imgCentral.shape[1], 8):
                block = imgCentral[i:i+8, j:j+8] #extraemos un bloque de 8x8 pixeles comenzando en la posición [i,j]
                bolcks.append(block) #agregamos el bloque a la lista

        return bolcks

    return rgb2gray(img), split(img)

def dct2D(block):
    return cv2.dct(block.astype(np.float32))

def quantize(blockDCT, quantizationMatrix):
    quantizedBlock = np.round(blockDCT / quantizationMatrix)
    quantizedBlock[quantizedBlock == -0.0] = 0.0
    return quantizedBlock

def zigzagEncodeDecode(quantizedBlock, rle = None):
    zigzagList = [] #lista que almacena los elementos ordenados en zigzag
    zigzagExtend = [] #lista auxiliar que contiene los elementos ordenados en zigzag pero con repetidos

    #para la decodificación
    if rle is not None:
        #reconstruyendo la secuencia original a partir de la lista codificada. Se toma cada par (value, cont) de la lista
        #rle y utiliza esa información para expandir el valor repetido el número de veces que especifica cont
        for value, cont in rle:
            zigzagExtend.extend([value] * cont)

    block = np.zeros_like(quantizedBlock) #matriz para acomodar los valores de la lista zigzag en forma de bloque
    rows, columns = quantizedBlock.shape
    index = 0
    totalElements = rows * columns
    #recorremos las diagonales de la matriz
    for i in range((rows + columns) - 1): #recorrido por flas
        #si el indice es par, recorremos la diagonal de abajo hacia arriba
        if i % 2 == 0:
            for j in range(i + 1):
                if j < rows and (i - j) < columns:
                    zigzagList.append(quantizedBlock[i - j, j])
                    if rle is not None:
                        block[i - j, j] = zigzagExtend[index]
                        index += 1
        #si el indice es impar, recorremos la diagonal de arriba hacia abajo
        else:
            for j in range(i + 1):
                if (i - j) < rows and j < columns:
                    zigzagList.append(quantizedBlock[j, i - j])
                    if rle is not None:
                        block[j, i - j] = zigzagExtend[index]
                        index += 1

    return zigzagList, block

def rleCoding(quantizedBlock):
    rleList = []
    rleDic = {}
    zigzagList, _= zigzagEncodeDecode(quantizedBlock)
    cont = 1

    for i in range(1, len(zigzagList)):
        if zigzagList[i] == zigzagList[i - 1]:
            cont += 1
        else:
            # diferenciamos los ceros por frecuencia en la lista RLE
            if zigzagList[i - 1] == 0:
                rleList.append((f"0_{cont}", cont))

            elif rleDic.get(zigzagList[i - 1]):
                rleList.append((f"{zigzagList[i - 1]}_{cont}", cont))

            else:
                rleList.append((zigzagList[i - 1], cont))
                rleDic[zigzagList[i - 1]] = cont

            cont = 1

    rleList.append((zigzagList[-1], cont))

    return rleList

def huffmanCoding(quantizedBlock):
    rleList = rleCoding(quantizedBlock)

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

    listAux = rleList #creamos una lista auxiliar para evitar error de fuera de rango
    if len(rleList) > 1:
        listAux = rleList[:-1]

    frequencies = {}

    for value, freq in listAux:
        frequencies[value] = freq

    tree = huffmanTree(frequencies)
    codes = huffmanCodes(tree)

    encodedHuffman =  ""
    for value, _ in listAux:
        encodedHuffman += codes[value]

    return encodedHuffman, codes, frequencies

#proceso de decodificación
def huffmanDecoding(encodedHuffman, codes, frequencies, length = 64):
    reverseCodes ={} #diccionario que almacenara los codigos invertidos, este será util para buscar los símbolos originales a partir de los códigos de huffman

    #iteramos sobre cada clave y valor del diccionario original (codes)
    for k, v in codes.items():
        reverseCodes[v] = k #asignamos v (valor) como clave y k (simbolo original) como su valor asociado

    actualCode = "" #ayudara a acumular los bits de la cadena codificada hasta formar el código de huffman completo
    decoded = [] #lista para almacenar los simbolos decodificados y su frecuencia
    freqAux = 0

    #iteramos sobre cada bit de la secuencia resultante de la codificación de huffman
    for bit in encodedHuffman:
        actualCode += bit #en cada iteración se añade un bit al codigo actual, así se va formando de maneta incremental un codigo de huffman
        #verificamos si el conjunto de bits representa un simbolo codificado
        if actualCode in reverseCodes:
            symbol = str(reverseCodes[actualCode])
            if "_" in symbol: #detectamos los ceros intermedios
                value = float(symbol.split("_")[0])
                freq = frequencies[symbol]
            else:
                value = float(symbol)
                freq = frequencies[value]

            freqAux += freq
            decoded.append((value, freq))  #almacenamos cada valor con su frecuencia
            actualCode = "" #vaciamos el codigo actual para empezar a acumular los bits siguientes del código de huffman

    # rrellenamos la lista con los ceros sobrantes y su respectiva frecuencia
    decoded.append((0.0, length - freqAux))

    return decoded

def dequantize(quantizedBlock, quantizationMatrix):
    return quantizedBlock * quantizationMatrix

def idct2D(block):
    return cv2.idct(block.astype(np.float32))

def rebuildImage(blocks, imageForm):
    reconstructed = np.zeros_like(imageForm)

    index = 0
    for i in range(0, imageForm.shape[0], 8):
        for j in range(0, imageForm.shape[1], 8):
            reconstructed[i:i + 8, j:j + 8] = blocks[index]
            index += 1
    return reconstructed


if __name__ == "__main__":
    img_RGB = io.imread('lenna.png') #carga de la imagen

    #matriz de cuantización estándar
    quantizationMatrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    # quantizationMatrix = np.array([
    #     [2, 2, 2, 2, 3, 4, 5, 6],
    #     [2, 2, 2, 2, 3, 4, 5, 6],
    #     [2, 2, 2, 2, 4, 5, 7, 9],
    #     [2, 2, 2, 4, 5, 7, 9, 12],
    #     [3, 3, 4, 5, 8, 10, 12, 12],
    #     [4, 4, 5, 7, 10, 12, 12, 12],
    #     [5, 5, 7, 9, 12, 12, 12, 12],
    #     [6, 6, 9, 12, 12, 12, 12, 12]
    #     ])

    # quantizationMatrix = np.array([
    #     [3, 3, 5, 9, 13, 15, 15, 15],
    #     [3, 4, 6, 11, 14, 12, 12, 12],
    #     [5, 6, 9, 14, 12, 12, 12, 12],
    #     [9, 11, 14, 12, 12, 12, 12, 12],
    #     [13, 14, 12, 12, 12, 12, 12, 12],
    #     [15, 12, 12, 12, 12, 12, 12, 12],
    #     [15, 12, 12, 12, 12, 12, 12, 12],
    #     [15, 12, 12, 12, 12, 12, 12, 12]
    # ])

    # Codificación
    # Paso 1: Preprocesamiento
    imgGray, blocks = preProcess(img_RGB)
    dequantizedBlocks = []

    for block in blocks:
        # Paso 2: Transformación DCT
        dctBlock = dct2D(block)

        # Paso 3: Cuantificación
        quantizedBlock = quantize(dctBlock, quantizationMatrix)

        # Paso 4: Codificación
        # Codificación RLE
        encodedRle = rleCoding(quantizedBlock)

        # Codificación Huffman
        encodedHuffman, codes, frequencies = huffmanCoding(quantizedBlock)

        # Decodificacióm
        # Paso 4 inverso: Decodificación
        # Huffman
        decodedHuffman = huffmanDecoding(encodedHuffman, codes, frequencies)

        #RLE
        zigzagList, decodeQuantizedBlock = zigzagEncodeDecode(quantizedBlock, decodedHuffman)

        # Paso 3 inverso: Descuantificación
        dequantizedBlock = dequantize(decodeQuantizedBlock, quantizationMatrix)

        # Paso 2 inverso: IDCT
        idctBlock = idct2D(dequantizedBlock)

        #almacenamos el bloque procesado
        dequantizedBlocks.append(idctBlock)

    # Reconstruir la imagen a partir de los bloques
    constructedImage = rebuildImage(dequantizedBlocks, imgGray)

    # Finalmente añadimo el valor 127 para revertir el centrado
    constructedImage += 127

    cv2.imwrite('fotoCompress.jpg', constructedImage)

    cv2.imshow("Imagen Original", imgGray.astype(np.uint8))
    cv2.imshow("Imagen Reconstruida", constructedImage.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # plt.figure()
    # plt.imshow(imgGray, cmap = 'gray')
    # plt.axis('off')
    # plt.show()
    #
    plt.figure()
    plt.imshow(constructedImage, cmap = 'gray')
    plt.axis('off')
    plt.show()