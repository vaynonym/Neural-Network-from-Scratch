# to downscale array use downscale_array_by_half

import imageio
import numpy as np

def downscale_array_by_half(array, mode="rgb"):
    # list of two adjacent column elements:
    
    column_array = np.array([list(zip(array[j], array[j+1])) if j+1 < array.shape[0] else list(zip(array[j], array[j]))
                            for j in range(0, array.shape[0], 2)])
    # list of the two adjacest row elements in the column_array:
    four_tuple_list = [(column_array[i][j][0], column_array[i][j][1], column_array[i][j+1][0], column_array[i][j+1][1]) 
                        if j+1 < column_array.shape[1] 
                        else (column_array[i][j][0], column_array[i][j][1], column_array[i][j][0], column_array[i][j][1]) 
                        for i in range(0, column_array.shape[0], 1) for j in range(0, column_array.shape[1], 2)]
    if (mode=="bw"):
        four_tuple_array = np.array(four_tuple_list).reshape(int(array.shape[0]/2 + 0.5), int(array.shape[1]/2 + 0.5), 4)
    if (mode=="rgb"):
        four_tuple_array = np.array(four_tuple_list).reshape(int(array.shape[0]/2 + 0.5), int(array.shape[1]/2 + 0.5), 4, 3)
    four_tuple_list = list(four_tuple_array)
    # replacing the 2x2 matrix with a value of the mean
    down_scaled_array_list = list(np.zeros([four_tuple_array.shape[0], four_tuple_array.shape[1]]))

    if (mode=="bw"):
        for i in range(len(four_tuple_list)):
            down_scaled_array_list[i] = [(list(x)[0] + list(x)[1] + list(x)[2] + list(x)[3] )/4 for x in list(four_tuple_list[i])]
    if (mode=="rgb"):
        for i in range(len(four_tuple_list)):
            # necessary to convert to normal int so we don't overflow when we go over 256 before dividing by 4
            down_scaled_array_list[i] = [((int(x[0][0]) + x[1][0] + x[2][0] + x[3][0])/4,
                                          (int(x[0][1]) + x[1][1] + x[2][1] + x[3][1])/4,
                                          (int(x[0][2]) + x[1][2] + x[2][2] + x[3][2])/4) for x in four_tuple_list[i]]

    return np.array(down_scaled_array_list)

akane_array = imageio.imread("testimages/Akane.jpg")
print(type(akane_array))
print(akane_array.shape)

for i in range(3):
    akane_array = downscale_array_by_half(akane_array, mode="rgb")

print(type(akane_array))
print(akane_array.shape)
print(akane_array)


imageio.imwrite("pixelAkane.jpg", akane_array)