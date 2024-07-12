import cv2
import glob
show = False
save_names = True

main = str('./Synthetic/Final')
# Clean the processing folder
import os
for f in glob.glob(main+"/Cropouts/S*.jpg"):
    os.remove(f)

main = str('./Synthetic/Final')
# Load the files
def get_file_names(path):
    ext = ['jpg']    # Add formats here
    files = []
    [files.extend(glob.glob(path + 'S*.' + e)) for e in ext]
    return files
files = get_file_names(main+'/Original/')

path_dia2 = main + str('/Cropouts/Names') + str('.txt')
if save_names==True:
    f2 = open(path_dia2, 'w')

##############################################################################
# Define the image to be used
for idx, file in enumerate(files):
    image = cv2.imread(files[idx])

    # Size of the image
    len_y = len(image)
    len_x = len(image[0])
    # Slide window
    size_y = 640
    size_x = 640
    max_y = len_y//size_y
    max_x = len_x//size_x

    for i in range(max_y):
        ##############################################################################
        # Define the size of the output pictures
        # Height of the picture
        line = i                    # first line is zero
        y2 = size_y + (line*size_y) # y2 maximum height
        y1 = y2 - size_y            # y1 minimum height

        # width of the pictures
        x2 = size_x                 # x2 maximum width
        x1 = x2 - size_x            # x1 minimum width
        yy = 0                      # step y (only if I wanna make with angle)
        xx = size_x                 # As I'm not interesed in over-writing
                                    # the second starts where the first ends
        maxi = max_x                # number of results (for each line)

        ##############################################################################
        # Present and save the results
        ins=0
        for j in range(0,maxi):
            # image output
            ins = image[(y1+(i*yy)):y2+(j*yy), x1+(j*xx):x2+(j*xx)]
            if show==True:
                cv2.imshow("FRI", ins)
                cv2.waitKey(500)

            # name the output
            name_j = str(j) # Variation of x
            name_i = str(i) # Variation of y
            filename = 'Synthetic/Final/Cropouts/'+ file[-20:-4] + '_R_'+name_i+'_'+name_j+'.jpg'

            #'Synthetic/Final/Cropouts/' + file_name[0:-4] + '_' + str(line[-36]) + '.jpg'

            print(filename)
            # save the results
            cv2.imwrite(filename, ins)

            if save_names==True:
                # Save the image names
                f2.write(filename)
                f2.write('\n')

    if show==True:
        cv2.destroyAllWindows()

if save_names==True:
    f2.close()
     
