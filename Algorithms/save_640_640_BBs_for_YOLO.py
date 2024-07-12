'''
Algorithm wrote by Stefano Frizzo Stefenon
Fondazione Bruno Kessler
Trento, Italy, June 06, 2024.
'''
import glob

main = str('./Synthetic/Final')
# Clean the processing folder
import os
for f in glob.glob(main+"/Cropouts/S*.txt"):
    os.remove(f)

# Load the files
def get_file_names(path):
    ext = ['txt']    # Add formats here
    files = []
    [files.extend(glob.glob(path + 'S*.' + e)) for e in ext]
    return files
files = get_file_names(main+'/Original/')

for file in files:
    with open(file=file) as f:
        lines = f.readlines()
    for line in lines:
        for slide_ln in range(0,9):
            new_lines = []

            if int(line[-27])==0:
                if int(slide_ln)==int(line[-36]):
                    final_line = line.replace((' ' + str(slide_ln)),' 0', 1)
                    new_lines.append(final_line)
                    new_file = 'Synthetic/Final/Cropouts/' + file[-20:-4] + '_R_0_' + str(int(line[-36])) + '.txt'
                    g = open(new_file, "a")
                    g.writelines(new_lines)
                    print(new_file)

            if int(line[-27])==1:
                if int(slide_ln)==int(line[-36]):
                    final_line = line.replace((' ' + str(1)),' 0', 2)
                    new_lines.append(final_line)
                    new_file = 'Synthetic/Final/Cropouts/' + file[-20:-4] + '_R_1_' + str(int(line[-36])) + '.txt'
                    g = open(new_file, "a")
                    g.writelines(new_lines)
                    print(new_file)

g.close()
     
