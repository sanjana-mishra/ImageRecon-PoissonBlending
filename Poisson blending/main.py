import cv2
import numpy as np
from scipy import sparse

from align_target import align_target
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

def poisson_blend(source_image, target_image, target_mask):

    H, W, C = target_image.shape

    #num of pixels inside patch and their indices
    patch_px = np.argwhere(target_mask)
    num = len(patch_px)

    ##initalizing A and b
    A = np.zeros((num, num))
    b = np.zeros((C, num))

    #matrix for index count:
    counter = 0
    count = np.zeros((H,W), np.int32)
    for k in range(num):
        i = patch_px[k][0]
        j = patch_px[k][1]
        count[i, j] = counter
        counter+=1

    for i in range(H):
        for j in range(W):
            if target_mask[i,j] == 0:
                #pixel not in mask
                continue

            #for channels
            for k in range(C):
                ##setting A and b
                A[count[i,j], count[i,j]] = 4

                v_up = 0
                if target_mask[i-1, j] == 0:
                    b[k, count[i,j]] += target_image[i-1, j, k]

                else:
                    A[count[i,j], count[i-1,j]] = -1
                    b[k, count[i,j]] += int(source_image[i,j,k]) - int(source_image[i-1, j, k])

                v_left = 0
                if target_mask[i, j - 1] == 0:
                    b[k, count[i, j]] += target_image[i, j - 1, k]

                else:
                    A[count[i, j], count[i, j - 1]] = -1
                    b[k, count[i, j]] += int(source_image[i, j, k]) - int(source_image[i, j-1, k])

                v_down = 0
                if target_mask[i+1, j] == 0:
                    b[k, count[i, j]] += target_image[i+1, j, k]

                else:
                    A[count[i, j], count[i+1, j]] = -1
                    b[k, count[i, j]] += int(source_image[i, j, k]) - int(source_image[i+1, j, k])

                v_right= 0
                if target_mask[i, j + 1] == 0:
                    b[k, count[i, j]] += target_image[i, j + 1, k]

                else:
                    A[count[i, j], count[i, j + 1]] = -1
                    b[k, count[i, j]] += int(source_image[i, j, k]) - int(source_image[i, j + 1, k])

    # spare matrix
    A = csr_matrix(A)
    b = csr_matrix(b)
    # in BGR format
    channel_red = sparse.linalg.spsolve(A, b[2].T)
    channel_green = sparse.linalg.spsolve(A, b[1].T)
    channel_blue = sparse.linalg.spsolve(A, b[0].T)

    # error calculation
    error_red = np.linalg.norm(A * channel_red - b[2])
    print("Error in the Red channel: ", error_red)
    error_green = np.linalg.norm(A * channel_green - b[1])
    print("Error in the Green channel: ", error_green)
    error_blue = np.linalg.norm(A * channel_blue - b[0])
    print("Error in the Blue channel: ", error_blue)

    for c in range(num):
        i = patch_px[c][0]
        j = patch_px[c][1]
        target_image[i,j,0] = abs((channel_blue[c]))
        target_image[i,j,1] = abs((channel_green[c]))
        target_image[i,j,2] = abs((channel_red[c]))


    blended_image = target_image
    cv2.imshow("Final image after Poisson blend", blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #read source and target images
    source_path = 'source1.jpg'
    target_path = 'target.jpg'
    source_image = cv2.imread(source_path)
    target_image = cv2.imread(target_path)

    #align target image
    im_source, mask = align_target(source_image, target_image)

    ##poisson blend
    blended_image = poisson_blend(im_source, target_image, mask)