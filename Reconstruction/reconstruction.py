import numpy as np
import cv2
from scipy.sparse import csr_matrix, linalg
import matplotlib as plt


if __name__ == '__main__':
    ##read source image
    img_path = '../Reconstruction/large1.jpg'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow("image", img)
    cv2.waitKey(0)

    ##forming matrix A and vector b
    H, W = img.shape

    A = np.zeros((H * W, H * W))
    b = np.zeros(H * W)

    ##loop
    counter = 0
    const = 215
    for i in range(H):
        for j in range(W):
            if (i * W + j) == 0 or (i * W + j) == H-1 or (i * W + j) == W-1 or (i * W + j) == (H*W)-1:
                A[counter, i * W + j] = 1
                b[counter] = const
                counter += 1
                continue

            #vertical derivatives
            if i == 0 or i == H-1:
                A[counter, i * W + j] = 2
                A[counter, i * W + (j+1)] = -1
                A[counter, i * W + (j-1)] = -1
                b[counter] = (2*img[i, j]) - img[i, j-1] - img[i, j+1]

            # horizontal derivatives
            elif j == 0 or j == W-1:
                A[counter, i * W + j] = 2
                A[counter, (i + 1) * W + j] = -1
                A[counter, (i - 1) * W + j] = -1
                b[counter] = (2*img[i, j]) - img[i-1, j] - img[i+1, j]

            else:
                A[counter, i * W + j] = 4
                A[counter, i * W + j + 1] = -1
                A[counter, i * W + j - 1] = -1
                A[counter, (i + 1) * W + j] = -1
                A[counter, (i - 1) * W + j] = -1

                b[counter] = (4*img[i, j]) - img[i-1, j] - img[i+1, j] - img[i, j-1] - img[i, j+1]

            counter += 1


    ##Add constraint that controls brightness
    b[0] = const
    b[W - 1] = const
    b[H - 1] = const
    b[(H*W) - 1] = const

    #spare matrix
    A = csr_matrix(A)
    b = csr_matrix(b)
    image_hat = linalg.spsolve(A, b.T)
    error = np.linalg.norm(A * image_hat - b)
    print("Error: ", error)
    image_hat = image_hat.reshape(H, W)

    ##solve system of equations
    #image_hat = np.linalg.pinv(A) @ b
    #
    # ##plot image
    #image_hat = image_hat.reshape(H, W)
    image_hat = image_hat.astype(np.float64)/255.



    cv2.imshow("Reconstructed image", image_hat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
