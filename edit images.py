from PIL import Image
import glob

image_list = []


def load_images_level0():
    cont = 1
    j = 0
    print('\nCarregando imagens das retinas saudáveis (PASTA 1)\n')

    for i in glob.glob('/home/deusimar/Imagens/bases/ADCIS/levels/level0/selecionadas/*.tif'):
        im2 = Image.open(i)
        im2.save('data_set_png/train/normal/q6%d.png' % cont)
        im2 = im2.rotate(35)
        im2.save('data_set_png/train/normal/q6q%d.png' % cont)
        im2 = im2.rotate(55)
        im2.save('data_set_png/train/normal/q6qq%d.png' % cont)
        im2 = im2.rotate(75)
        im2.save('data_set_png/train/normal/q6qw%d.png' % cont)
        im2 = im2.rotate(125)
        im2.save('data_set_png/train/normal/e6%d.png' % cont)
        im2 = im2.rotate(180)
        im2.save('data_set_png/train/normal/r6%d.png' % cont)
        im2 = im2.rotate(15)
        im2.save('data_set_png/train/normal/t6%d.png' % cont)
        im2 = im2.rotate(25)
        im2.save('data_set_png/train/normal/y6%d.png' % cont)
        im2 = im2.rotate(25)
        im2.save('data_set_png/train/normal/u6%d.png' % cont)
        im2 = im2.rotate(24)
        im2.save('data_set_png/train/normal/i6%d.png' % cont)
        cont = cont+1
        j = j+1

    print('A memória está para ser liberada...\n')

    image_list.clear()

    print('Memória liberada pela primeira vez...\n')


def load_images_DR():

    print('\nCarregando imagens das retinas com DR (LEVEL 3)\n')

    cont = 1
    j = 0
    for i in glob.glob('/home/deusimar/Imagens/bases/ADCIS/levels/level3/selecionadas/*.tif'):
        im2 = Image.open(i)
        im2.save('data_set_png/train/avancado/i61%d.png' % cont)
        im2 = im2.rotate(9)
        im2.save('data_set_png/train/avancado/61%d.png' % cont)
        im2 = im2.rotate(5)
        im2.save('data_set_png/train/avancado/61%d.png' % cont)
        im2 = im2.rotate(5)
        im2.save('data_set_png/train/avancado/61%d.png' % cont)
        im2 = im2.rotate(5)
        im2.save('data_set_png/train/avancado/61%d.png' % cont)
        im2 = im2.rotate(5)
        im2.save('data_set_png/train/avancado/61%d.png' % cont)
        im2 = im2.rotate(5)
        im2.save('data_set_png/train/avancado/61%d.png' % cont)
        im2 = im2.rotate(5)
        im2.save('data_set_png/train/avancado/61%d.png' % cont)
        im2 = im2.rotate(5)
        im2.save('data_set_png/train/avancado/61%d.png' % cont)
        im2 = im2.rotate(5)
        im2.save('data_set_png/train/avancado/61%d.png' % cont)
        cont = cont+1
        j = j+1

    print('A memória está para ser liberada...\n')

    image_list.clear()

    print('Memória liberada pela terceira vez...\n')


def change():
    import glob as g
    import cv2

    file = open('train.csv', 'w')

    cont = 0
    list_normal = g.glob('/home/deusimar/Pictures/crop/crop-train/normal/*.png')
    list_advanced = g.glob('/home/deusimar/Pictures/crop/crop-train/avancado/*.png')
    array = []

    for i, value in enumerate(list_advanced):
        print(value)
        img = cv2.imread(value, 0)
        img = cv2.resize(img, None, fx=0.2, fy=0.2)
        cv2.imwrite('/home/deusimar/Pictures/data-set_train/image%d.png' % cont, img)

        for line in range(img.shape[0]):
            for column in range(img.shape[1]):
                array.append(img[line][column])

        file.write(str('0,'))
        file.write(','.join(str(b) for b in array) + "\n")
        print('image%d completed' % cont)

        array.clear()

        cont = cont + 1

        img = cv2.imread(list_normal[i], 0)
        print(list_normal[i])
        img = cv2.resize(img, None, fx=0.2, fy=0.2)
        cv2.imwrite('/home/deusimar/Pictures/data-set_train/image%d.png' % cont, img)

        for line in range(img.shape[0]):
            for column in range(img.shape[1]):
                array.append(img[line][column])

        file.write(str('1,'))
        file.write(','.join(str(b) for b in array) + "\n")
        print('image%d completed' % cont)

        array.clear()

        cont = cont + 1

    file.close()


def green_image():

    import cv2
    import glob as g
    list_path = g.glob('crop/train/*.png')
    cont = 0
    cont1 = 0
    for i, value in enumerate(list_path):
        # img = cv2.imread(value)
        # img = cv2.resize(img, None, fx=0.2, fy=0.2)
        # for i in range(img.shape[0]):
        #     for j in range(img.shape[1]):
        #         cores = img[i][j]
        #         img[i][j] = [0, cores[1], 0]
        # cv2.imwrite('/home/deusimar/Pictures/data-set_train/image%d.png' % cont, img)
        # cont = cont + 1
        print(value)

    print('train completed')

    list_path.clear()

    list_path = g.glob('crop/test/*.png')

    for i, value in enumerate(list_path):
        # img = cv2.imread(value)
        # img = cv2.resize(img, None, fx=0.2, fy=0.2)
        # for i in range(img.shape[0]):
        #     for j in range(img.shape[1]):
        #         cores = img[i][j]
        #         img[i][j] = [0, cores[1], 0]
        # cv2.imwrite('/home/deusimar/Pictures/data-set_test/image%d.png' % cont1, img)
        # cont1 = cont1 + 1
        print(value)

    print('test completed')


def test():

    import cv2

    img = cv2.imread('crop/train/image0.png', 0)
    cv2.imshow('red gray', img)

    img = cv2.imread('crop/train/image0.png')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            cores = img[i][j]
            img[i][j] = [0, cores[1], 0]

    cv2.imwrite('image1.png', img)

    img = cv2.imread('image1.png', 0)
    cv2.imshow('green gray', img)
    cv2.waitKey(0)


def limiar_image():

    import cv2
    from matplotlib import pyplot as plt

    img = cv2.imread('/home/deusimar/Pictures/crop/crop-train/avancado/image6.png', 0)

    img = cv2.medianBlur(img, 5)
    cv2.imshow('sss', img)
    cv2.waitKey(0)

    ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    for i in range(th2.shape[0]):
        for j in range(th2.shape[1]):
            if th3[i][j] < 100:
                img[i][j] = 117

    cv2.imshow('img', img)
    cv2.waitKey(0)

    titles = ['Original Image', 'Global Thresholding (v = 127)',
              'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]
    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


def quantization():
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    img = cv2.imread('/home/deusimar/Pictures/crop/crop-train/avancado/image0.png', 0)
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == '__main__':
    # load_images_level0()
    # load_images_DR()
    # change()
    # green_image()
    limiar_image()
    # quantization()