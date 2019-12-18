# to downscale array use downscale_array_by_half

import imageio
import numpy as np



def downscale_image_by_half( inputpath, outputpath, extent ):
    filetype = (inputpath[::-1]).split('.')[0][::-1]
    print(filetype)
    array = imageio.imread(inputpath).astype(int)
    if(filetype == "jpg"):
        mode = "rgb"
    elif(filetype == "png"):
        mode = "png"
    else:
        mode = "bw"
    
    for i in range(extent):
        array = downscale_array_by_half(array, mode)
    imageio.imwrite(outputpath, array.astype("uint8"))



    
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
    if (mode=="png"):
        four_tuple_array = np.array(four_tuple_list).reshape(int(array.shape[0]/2 + 0.5), int(array.shape[1]/2 + 0.5), 4, 4)
    four_tuple_list = list(four_tuple_array)
    # replacing the 2x2 matrix with a value of the mean
    down_scaled_array_list = list(np.zeros([four_tuple_array.shape[0], four_tuple_array.shape[1]]))

    if (mode=="bw"):
        for i in range(len(four_tuple_list)):
            down_scaled_array_list[i] = [( x[0] + x[1] + x[2] + x[3])/4 for x in four_tuple_list[i]]
    if (mode=="rgb"):
        for i in range(len(four_tuple_list)):
            down_scaled_array_list[i] = [((x[0][0] + x[1][0] + x[2][0] + x[3][0])/4,
                                          (x[0][1] + x[1][1] + x[2][1] + x[3][1])/4,
                                          (x[0][2] + x[1][2] + x[2][2] + x[3][2])/4) for x in four_tuple_list[i]]
    if (mode=="png"):
        for i in range(len(four_tuple_list)):
            down_scaled_array_list[i] = [((x[0][0] + x[1][0] + x[2][0] + x[3][0])/4,
                                          (x[0][1] + x[1][1] + x[2][1] + x[3][1])/4,
                                          (x[0][2] + x[1][2] + x[2][2] + x[3][2])/4,
                                          (x[0][3] + x[1][3] + x[2][3] + x[3][3])/4) for x in four_tuple_list[i]]
    return np.array(down_scaled_array_list)

