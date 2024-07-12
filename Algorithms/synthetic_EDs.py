'''
Algorithm wrote by Stefano Frizzo Stefenon
Fondazione Bruno Kessler
Trento, Italy, June 06, 2024.
'''
import cv2
import glob
from numpy import asarray
from PIL import Image
import numpy as np
import random
from random import randint
import sys
import os
import shutil
import math

###############################################################################
################################### Setup #####################################
###############################################################################
# Show step by step amd saving
save = True
save_cropped = True
save_names = True
include_noise = False
save_sp = False
img_show = False

# Line definition
line_setup_c = (0,0,0)
line_setup_width = 2
line_lenght_s = 105

# Probabilities
sy_or_dot = 50 # % of being a dot (electrical connection)
parallel_drawing_p = 50 # % of having a parallel circuit
skip_symbol_p = 1 # % of skipping a symbol
change_dir_p = 20 # % of changing direction

# Drawing Setup
n_symbols = 15 # Number of symbols
n_drawings = 8000 # Number of drawings
size_p = (2*640, 2*640) # Size of the drawing (640 or 1280 pixels)
symbol_size = 35
label_size = (15, 30)
specifier_size = (20, 50)

###############################################################################
#################################### Rules ####################################
###############################################################################
# Symbols (x >= 76) / Letters (x <= 61) / Arrows (x >= 74 and x <= 75) / Dots (x >= 62 and x <= 73)

# Resuts of YOLO (YOLOv8n)
symbols = list(range(76, 133+1))
labels = list(range(0, 61+1))

# Case 1C
weights_s = [abs(math.log10(0.930)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # C00-C21
             0, 0, abs(math.log10(0.975)), 0, abs(math.log10(0.992)), abs(math.log10(0.970)), abs(math.log10(0.916)), 0, 0, 0, 0, 0, 0, 0, # C22-C35
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, abs(math.log10(0.669)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # case 1A
weights_l = [abs(math.log10(0.910)), abs(math.log10(0.914)), abs(math.log10(0.977)), abs(math.log10(0.911)), abs(math.log10(0.847)), # 0-4
             abs(math.log10(0.912)), abs(math.log10(0.703)), abs(math.log10(0.898)), abs(math.log10(0.674)), abs(math.log10(0.705)), # 5-9
             abs(math.log10(0.100)), abs(math.log10(0.966)), abs(math.log10(0.987)), abs(math.log10(0.972)), abs(math.log10(0.957)), # a-e
             abs(math.log10(0.100)), abs(math.log10(0.589)), 0, abs(math.log10(0.100)), 0, 0, abs(math.log10(0.100)), # f-l
             abs(math.log10(0.984)), abs(math.log10(0.100)), 0, 0, 0, abs(math.log10(0.596)), abs(math.log10(0.891)), # m-s
             0, abs(math.log10(0.660)), 0, 0, 0, 0, 0, abs(math.log10(0.333)), abs(math.log10(0.455)), 0, # t-C
             abs(math.log10(0.971)), abs(math.log10(0.753)), abs(math.log10(0.100)), abs(math.log10(0.100)), abs(math.log10(0.617)),# D-H
             abs(math.log10(0.763)), abs(math.log10(0.945)),  abs(math.log10(0.820)), abs(math.log10(0.680)), abs(math.log10(0.787)), # I-M
             abs(math.log10(0.148)), 0, abs(math.log10(0.958)), 0, abs(math.log10(0.988)), abs(math.log10(0.764)), abs(math.log10(0.923)), # N-T
             abs(math.log10(0.971)), abs(math.log10(0.785)),  0, 0, 0, abs(math.log10(0.100))]

'''
# Case 1A
weights_s = [(1/0.930), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # C00-C21
             0, 0, (1/0.975), 0, (1/0.992), (1/0.970), (1/0.916), 0, 0, 0, 0, 0, 0, 0, # C22-C35
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (1/0.669), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # case 1A
weights_l = [(1/0.910), (1/0.914), (1/0.977), (1/0.911), (1/0.847), # 0-4
             (1/0.912), (1/0.703), (1/0.898), (1/0.674), (1/0.705), # 5-9
             (1/0.100), (1/0.966), (1/0.987), (1/0.972), (1/0.957), # a-e
             (1/0.100), (1/0.589), 0, (1/0.100), 0, 0, (1/0.100), # f-l
             (1/0.984), (1/0.100), 0, 0, 0, (1/0.596), (1/0.891), # m-s
             0, (1/0.660), 0, 0, 0, 0, 0, (1/0.333), (1/0.455), 0, # t-C
             (1/0.971), (1/0.753), (1/0.100), (1/0.100), (1/0.617),# D-H
             (1/0.763), (1/0.945),  (1/0.820), (1/0.680), (1/0.787), # I-M
             (1/0.148), 0, (1/0.958), 0, (1/0.988), (1/0.764), (1/0.923), # N-T
             (1/0.971), (1/0.785),  0, 0, 0, (1/0.100)]

# Case 1B
weights_s = [(1-0.930), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, (1-0.975), 0, (1-0.992), (1-0.970), (1-0.916), 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (1-0.669), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # case 1B
weights_l = [(1-0.910), (1-0.914), (1-0.977), (1-0.911), (1-0.847), # 0-4
             (1-0.912), (1-0.703), (1-0.898), (1-0.674), (1-0.705), # 5-9
             (1-0.100), (1-0.966), (1-0.987), (1-0.972), (1-0.957), # a-e
             (1-0.100), (1-0.589), 0, (1-0.100), 0, 0, (1-0.100), # f-l
             (1-0.984), (1-0.100), 0, 0, 0, (1-0.596), (1-0.891), # m-s
             0, (1-0.660), 0, 0, 0, 0, 0, (1-0.333), (1-0.455), 0, # t-C
             (1-0.971), (1-0.753), (1-0.100), (1-0.100), (1-0.617),# D-H
             (1-0.763), (1-0.945),  (1-0.820), (1-0.680), (1-0.787), # I-M
             (1-0.148), 0, (1-0.958), 0, (1-0.988), (1-0.764), (1-0.923), # N-T
             (1-0.971), (1-0.785),  0, 0, 0, (1-0.100)]

# Standard P_class
weights_s = [(1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, (1), 0, (1), (1), (1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, (1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # case 1A
weights_l = [(1), (1), (1), (1), (1), (1), (1), (1), (1), (1), # 0-9
             (1), (1), (1), (1), (1), (1), (1), 0, (1), 0, 0, (1), # a-l
             (1), (1), 0, 0, 0, (1), (1), 0, (1), 0, 0, 0, 0, 0, (1), (1), 0, # m-C
             (1), (1), (1), (1), (1), (1), (1),  (1), (1), (1), # D-M
             (1), 0, (1), 0, (1), (1), (1), (1), (1),  0, 0, 0, (1)]'''

# Start of end of the circuit / End = (76, 81, 87, 92, 125); Start = (122)
# Circuit with one or 3 connections (89, 101, 104)
not_used = [76, 81, 87, 92, 125, 122, 101, 89, 104]
weights_c = weights_s
for i in range(0,len(not_used)):
    weights_c = np.where(np.array(symbols)==not_used[i], 0, weights_c)

# Regular symbols are not considered in the end of the circuit
not_end = [100, 102, 103, 104, 122]
weights_e = weights_s
for i in range(0,len(not_end)):
    weights_e = np.where(np.array(symbols)==not_end[i], 0, weights_e)

# Rotated objects to keep horizontal
symbol_rt = [83, 119, 120, 121, 127]

###############################################################################
################ Delete existing files and load annotations ###################
###############################################################################
# Main path
main = str('./Synthetic/')
# Clean the processing folder
for f in glob.glob(main+"/Final/Output/P*.jpg"):
    os.remove(f)
for f in glob.glob(main+"/Final/Original/S*.jpg"):
    os.remove(f)
for f in glob.glob(main+"/Final/Original/S*.txt"):
    os.remove(f)
for f in glob.glob(main+"/Final/Normal/S*.jpg"):
    os.remove(f)
for f in glob.glob(main+"/Final/Normal/S*.txt"):
    os.remove(f)
for f in glob.glob(main+"/Final/Background/S*.jpg"):
    os.remove(f)
for f in glob.glob(main+"/Final/Background/S*.txt"):
    os.remove(f)
for f in glob.glob(main+"/Final/Cropouts/S*.txt"):
    os.remove(f)
for f in glob.glob(main+"/Final/Cropouts/*.jpg"):
    os.remove(f)

print('Cleaned Folders')

# Import the images (Objects)
path = str('./Input_images/Difussion_just_train/') #diffusion

###############################################################################
############################## Functions to draw ##############################
###############################################################################
# Function to draw object
def draw_obj(symbol, label1, label2, label3, specifier, r_symbol, random_line, m_1, n_1):
    updated_p = update_position(m_1, n_1, r_symbol) # Rules to update the position
    if random_line==0:
        m_1 = m_1 + updated_p
    background.paste(specifier, ((m_1-int(symbol.size[0]/2)-(specifier.size[0])), (n_1-specifier.size[1]-5)))
    background.paste(label1, ((m_1+int(symbol.size[0]/2)), (n_1-(int(symbol.size[1]/2))-20)))
    background.paste(label2, ((m_1+int(symbol.size[0]/2)+label1.size[0]+1), (n_1-(int(symbol.size[1]/2))-20)))
    background.paste(label3, ((m_1+int(symbol.size[0]/2)+label1.size[0]+label2.size[0]+1), (n_1-(int(symbol.size[1]/2))-20)))
    if random_line==1:
        n_1 = n_1 - updated_p
    background.paste(symbol, ((m_1-int(symbol.size[0]/2)), (n_1-int(symbol.size[1]/2))))
    return

# Function to draw dots
def draw_dot(symbol, m_1, n_1):
    updated_p = update_position(m_1, n_1, r_symbol) # Rules to update the position
    if random_line==0 and special==0:
        m_1 = m_1 + updated_p
    if random_line==1 and special==0:
        n_1 = n_1 - updated_p
    background.paste(symbol, ((m_1-int(symbol.size[0]/2)), (n_1-int(symbol.size[1]/2))))
    return updated_p

# Function to draw lines to complete the drawing
def draw_lines(current_position, updated_position):
    p0, p1 = [current_position, updated_position]
    cv2.line(background, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), line_setup_c, line_setup_width)
    return

# Rules to update the position
def update_position(m_1, n_1, r_symbol):
    symbol = rotation(r_symbol, sb)
    for j in range(0, int(symbol.size[0]*0.1)): # Evaluate the first pixels
        horizontal = np.where(asarray(symbol)[j]<200); updated_p=0 # threshold = 100
        try:
            updated_p = int(symbol.size[0]/2)-horizontal[0][0]; break
        except:
            pass # If there is no line to match don' do it
    if r_symbol==76: # and dot==True: # When the symbol is at the end doesn't update it
        updated_p=0
    return updated_p

# Rotate the symbols if the line are vertical or don't do it if they are already rotated
def sym_rot(r_symbol, random_line, sb):
    if r_symbol in symbol_rt:
        rotated_sym = True
    else:
        rotated_sym = False
    if random_line!=0 and rotated_sym==False: # Check horizontal lines and rotation of the symbol
        symbol = sb
    elif random_line!=1 and rotated_sym==True:
        symbol = sb
    else: # Vertical lines
        symbol = rotation(r_symbol, sb)
    return symbol

def rotation(r_symbol, sb):
    if len(asarray(sb).shape)==3: # Consider RGB images
        color = str('RGB')
    else: # Consider gray scale images
        color = str('L')
    # Define the rotation for the dots
    if r_symbol == 63:
        rotation = cv2.ROTATE_180
    elif r_symbol == 67:
        rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
    else:
        rotation = cv2.ROTATE_90_CLOCKWISE
    if r_symbol == 62 or r_symbol == 65 or r_symbol == 64:
        symbol = sb # These dot are already on the right direction
    else: # For all other symbols
        symbol = Image.fromarray(cv2.rotate(asarray(sb), rotation), color)
    return symbol

###############################################################################
##################### Functions to load the annotations #######################
###############################################################################
def load_annotations(dot, parallel_d, conclude_d, special):
    while True: # Draw a dot only if it was not used before
        if (randint(1, 100) <= sy_or_dot) and dot == False and parallel_d == False and conclude_d == False and special == 0:
            if random_line==0: # Dots in vertical
                r_symbol = 65; dt = True
            else: # Dots in horizontal or rotated
                r_symbol = (random.choices([62, 63, 64, 67, 68], weights = [1, 1, 1, 1, 1], k = 1))[0]; dt = True
        elif conclude_d == False and special == 1:
            r_symbol = 70; dt = True
        elif conclude_d == False and special == 2:
            r_symbol = 69; dt = True
        elif conclude_d == False and special == 3:
                r_symbol = 71; dt = True
        elif conclude_d == False and special == 4:
                r_symbol = 104; dt = False
        elif conclude_d == True:
            r_symbol = (random.choices(symbols, weights = weights_e, k = 1))[0]; dt = False
        else: # Draw std symbols
            r_symbol = (random.choices(symbols, weights = weights_c, k = 1))[0]; dt = False
        try: # Load only the image that have annotations
            sb_names = glob.glob(path + str(r_symbol) + str('/*.jpg'), recursive=True)
            sb = Image.open(sb_names[randint(0,len(sb_names)-1)])
            if dt == False: # To fix the aspect ratio of symbols
                if sb.size[0] >= sb.size[1]:
                    sb = sb.resize((symbol_size, (int(symbol_size*sb.size[1]/sb.size[0]))))
                else:
                    sb = sb.resize((int(symbol_size*sb.size[0]/sb.size[1]), symbol_size))
            else: # To define the size of dots
                sb = sb.resize((int(symbol_size*0.7), int(symbol_size*0.7)))
            break
        except:
            pass
    a1 = int(sb.size[0]/2); a2 = int(sb.size[1]/2) # x and y of the symbol
    if dt==True: # To reduce the chance of having two dots in sequence
        dot = True
    else:
        dot = False
    while True: # Random initialization of drawing labels and specifiers
        r_label1 = (random.choices(labels, weights = weights_l, k = 1))[0]
        r_label2 = (random.choices(labels, weights = weights_l, k = 1))[0]
        r_label3 = (random.choices(labels, weights = weights_l, k = 1))[0]
        r_specifier = randint(74, 75)
        try: # Load only the image that have annotations
            label1_names = glob.glob(path + str(r_label1) + str('/*.jpg'), recursive=True)
            label1 = (Image.open(label1_names[randint(0,len(label1_names)-1)])).resize(label_size)
            label2_names = glob.glob(path + str(r_label2) + str('/*.jpg'), recursive=True)
            label2 = (Image.open(label2_names[randint(0,len(label2_names)-1)])).resize(label_size)
            label3_names = glob.glob(path + str(r_label3) + str('/*.jpg'), recursive=True)
            label3 = (Image.open(label3_names[randint(0,len(label3_names)-1)])).resize(label_size)
            specifier_names = glob.glob(path + str(r_specifier) + str('/*.jpg'), recursive=True)
            specifier = (Image.open(specifier_names[randint(0, len(specifier_names)-1)])).resize(specifier_size)
            break
        except: # When there is no folder
            pass
    return r_symbol, sb, r_label1, label1, r_label2, label2, r_label3, label3, r_specifier, specifier, dot, a1, a2

###############################################################################
######################### Functions to work with YOLO #########################
###############################################################################

# Function to convert cartesian coordinates to BB for YOLO
def conv_to_YOLO(m_1, n_1, a1, a2, Type, random_line):
    updated_p = update_position(m_1, n_1, r_symbol) # Rules to update the position
    if random_line==0:
        m_1 = m_1 + updated_p
    if Type==1: # Specifier
        x1=(m_1-int(sb.size[0]/2)-(specifier.size[0]))
        y1=(n_1-specifier.size[1]-5)
        x2=x1+specifier.size[0]; y2=y1+specifier.size[1]
    if Type==2: # Label1
        x1=(m_1+int(sb.size[0]/2))
        y1=(n_1-(int(sb.size[1]/2))-20)
        x2=x1+label1.size[0]; y2=y1+label1.size[1]
    if Type==3: # Label2
        x1=(m_1+int(sb.size[0]/2)+label1.size[0]+1)
        y1=(n_1-(int(sb.size[1]/2))-20)
        x2=x1+label2.size[0]; y2=y1+label2.size[1]
    if Type==4: # Label3
        x1=(m_1+int(sb.size[0]/2)+label1.size[0]+label2.size[0]+1)
        y1=(n_1-(int(sb.size[1]/2))-20)
        x2=x1+label3.size[0]; y2=y1+label3.size[1]
    if random_line==1:
        n_1 = n_1 - updated_p
    if Type==0 and random_line==1: # Symbol
        x1=m_1-a1; y1=n_1-a2; x2=m_1+a1; y2=n_1+a2
    if Type==0 and random_line==0: # Symbol
        x1=m_1-a2; y1=n_1-a1; x2=m_1+a2; y2=n_1+a1
    a=(0.5*x1+0.5*x2)/640; b=(0.5*y1+0.5*y2)/640
    c=(x2-x1)/640; d=(y2-y1)/640
    coord=[a, b, abs(c), d]
    return coord

# Save converted YOLO coordinates for YOLO print it
def save_YOLO(r_symbol, r_label1, r_label2, r_label3, r_specifier, m_1, n_1, random_line, dot):
    save_obj.append([r_symbol, r_label1, r_label2, r_label3, r_specifier, m_1, n_1, random_line, dot, special])
    Type=0; coord=conv_to_YOLO(m_1,n_1,a1,a2,Type,random_line) # Symbol
    print(f'{r_symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f} {coord[3]:.6f}')
    if dot==False:
        Type=1; coord=conv_to_YOLO(m_1,n_1,a1,a2,Type,random_line) # Specifier
        print(f'{r_specifier} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f} {coord[3]:.6f}')
        Type=2; coord=conv_to_YOLO(m_1,n_1,a1,a2,Type,random_line) # Label1
        print(f'{r_label1} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f} {coord[3]:.6f}')
        Type=3; coord=conv_to_YOLO(m_1,n_1,a1,a2,Type,random_line) # Label2
        print(f'{r_label2} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f} {coord[3]:.6f}')
        Type=4; coord=conv_to_YOLO(m_1,n_1,a1,a2,Type,random_line) # Label3
        print(f'{r_label3} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f} {coord[3]:.6f}')

###############################################################################
################################ Noise ########################################
###############################################################################
# Function to include noise
def noisy(noise_typ,image):
   image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   row,col = image.shape
   if noise_typ == "gauss_white": # Create gaussian noise
      Mean = 0.001; var = 1; sigma = np.sqrt(var)
      n = np.random.normal(loc=Mean, scale=sigma, size=(row,col))
      noisy = 255*(image + n) # Not save the background
      return noisy
   if noise_typ == "gauss": # Create gaussian noise
      image = image/255; Mean = 0; var = 0.01
      sigma = np.sqrt(var)
      n = np.random.normal(loc=Mean, scale=sigma, size=(row,col))
      noisy=255*(image + n) # Not save the background
      return noisy
   elif noise_typ == "s&p":
       output = np.zeros(image.shape,np.uint8)
       prob=0.005; thres = 1 - prob
       for i in range(image.shape[0]):
           for j in range(image.shape[1]):
               rdn = random.random()
               if rdn < prob:
                   output[i][j] = 0
               elif rdn > thres:
                    output[i][j] = 255
               else:
                    output[i][j] = image[i][j]
       return output
   elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 0.8 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
   elif noise_typ =="speckle":
        gauss = np.random.randn(row,col)
        gauss = gauss.reshape(row,col)
        noisy = image + image * gauss
        return noisy

###############################################################################
############################# Draws EDs #######################################
###############################################################################
path_dia2 = main + str('/Final/Normal/Names') + str('.txt')
if save_names==True:
    f2 = open(path_dia2, 'w')

for syn in range(0,n_drawings):
    # Raw Image Background
    background  = Image.new(mode = "RGB", size = size_p, color = (255, 255, 255))
    # Starting point
    start_position = [100, 70]; right=0; down=0; left=0; dot = True; dt = True
    label1=0; label2=0; label3=0; specifier=0; fst_skip=0
    current_position = start_position
    border_l = start_position
    line_lenght = line_lenght_s
    save_obj=[]; save_lin=[]; symbol=[]

    # Save output as a text file
    if save==True:
        orig_stdout = sys.stdout
        path_dia = main + str('/Final/Original/Synthetic_a_') + str(syn).zfill(3) + str('.txt')
        f = open(path_dia, 'w')
        sys.stdout = f

    for sym_i in range(0,n_symbols):
        conclude_d = False; first = False; special=0
        m_1, n_1 = current_position
        # Skip symbol in regular lines
        skip_symbol = random.uniform(0, 1) <= skip_symbol_p / 100.0 # skip_percentage
        # Define were to go
        c_d = randint(0, 100) # Chance of change direction
        if right==1: # Coming from right
            if c_d>=change_dir_p: # % of chance change_dir going right
                random_line=1
            else:
                random_line=0 # change direction go down
                skip_symbol=True # Skip the object
        if right==0:
            if c_d>=change_dir_p: # % of chance of change_dir going down
                random_line=0
            else:
                random_line=1 # change direction go right
                skip_symbol=True # Skip the object
        if sym_i==0: # Don't change the direction if it is the first
            random_line=0

        # If it is getting to close to the borders go in other direction
        if m_1>(size_p[0]*0.8):
            random_line=0;
            if fst_skip==0:
                skip_symbol=True
            else:
                skip_symbol=False
            fst_skip+=1
        if size_p[0]>640:
            rate=0.7
        else:
            rate=0.45
        if n_1>(size_p[0]*rate):
            random_line=1;
            if fst_skip==0:
                skip_symbol=True
            else:
                skip_symbol=False
            fst_skip+=1

        ###############################################################################
        ############################ Draw objects #####################################
        ###############################################################################
        # Random initialization of drawing symbols
        parallel_d = False
        r_symbol, sb, r_label1, label1, r_label2, label2, r_label3, label3, r_specifier, specifier, dot, a1, a2 = load_annotations(dot, parallel_d, conclude_d, special)

        # For the first symbol (if it is vertical)
        if sym_i == 0 and random_line==0:
            r_symbol = 122; first = True; dot = True
            sb_names = glob.glob(path + str(r_symbol) + str('/*.jpg'), recursive=True)
            sb = Image.open(sb_names[randint(0,len(sb_names)-1)])
            a1 = int(sb.size[0]/2); a2 = int(sb.size[1]/2) # x and y of the symbol
            draw_dot(sb, m_1, n_1)
            if save==True:
                save_YOLO(r_symbol, r_label1, r_label2, r_label3, r_specifier, m_1, n_1, random_line, dot)

        # Draw the symbol, label, and specifier
        if skip_symbol==False and dot==False and first==False:
            symbol = sym_rot(r_symbol, random_line, sb)
            draw_obj(symbol, label1, label2, label3, specifier, r_symbol, random_line, m_1, n_1); line_lenght = line_lenght_s
            if save==True:
                save_YOLO(r_symbol, r_label1, r_label2, r_label3, r_specifier, m_1, n_1, random_line, dot)

        ##############################################################################
        ############################## Draw Dots #####################################
        ##############################################################################
        if skip_symbol==False and dot==True and first==False: # Draw the dots
            symbol = rotation(r_symbol, sb)
            updated_p = draw_dot(symbol, m_1, n_1)
            if save==True:
                save_YOLO(r_symbol, r_label1, r_label2, r_label3, r_specifier, m_1, n_1, random_line, dot)

        ##############################################################################
        ########################## Parallel drawing ##################################
        ##############################################################################
        # Chance of having parallel_drawing
        if (randint(0, 100) <= parallel_drawing_p) and skip_symbol==False and dot==False and random_line!=0 and sym_i!=0:
            parallel_d = True
            r_symbol, sb, r_label1, label1, r_label2, label2, r_label3, label3, r_specifier, specifier, dot, a1, a2 = load_annotations(dot, parallel_d, conclude_d, special)

            # Lines of parallel drawing
            background = asarray(background)
            draw_lines([m_1-(line_lenght/2), n_1], [m_1-(line_lenght/2), n_1+line_lenght-2]) # Left vertical
            draw_lines([m_1-(line_lenght/2), n_1+line_lenght-2], [m_1-a1, n_1+line_lenght-2]) # Left horizontal
            draw_lines([m_1+a1, n_1+line_lenght-2], [m_1+a1+(line_lenght/2), n_1+line_lenght-2])
            draw_lines([m_1+a1+(line_lenght/2), n_1+line_lenght-2], [m_1+a1+(line_lenght/2), n_1])
            background = Image.fromarray(background)

            # Update the position to draw the parallel symbol
            n_1 = n_1 + line_lenght - line_setup_width
            symbol = sym_rot(r_symbol, random_line, sb)
            draw_obj(symbol, label1, label2, label3, specifier, r_symbol, random_line, m_1, n_1);
            if save==True:
                save_YOLO(r_symbol, r_label1, r_label2, r_label3, r_specifier, m_1, n_1, random_line, dot)
            n_1 = n_1 - line_lenght + line_setup_width

        ##############################################################################
        ########################## Special Conditions ################################
        ##############################################################################
        # Chance of having parallel_drawing
        if r_symbol != 62 and skip_symbol==False and dot==True and random_line!=0 and sym_i!=0 and conclude_d==False:
            background = asarray(background); a1b=a1; a2b=a2 # keep the coordinates
            draw_lines([m_1, n_1+a1-updated_p], [m_1, n_1+line_lenght]) # Vertical down
            background = Image.fromarray(background)
            r_symbol, sb, r_label1, label1, r_label2, label2, r_label3, label3, r_specifier, specifier, dot, a1, a2 = load_annotations(dot, parallel_d, conclude_d, special)
            n_1 = n_1 + line_lenght - line_setup_width; random_line=0
            symbol = sym_rot(r_symbol, random_line, sb)
            draw_obj(symbol, label1, label2, label3, specifier, r_symbol, random_line, m_1, n_1);
            if save==True:
                save_YOLO(r_symbol, r_label1, r_label2, r_label3, r_specifier, m_1, n_1, random_line, dot)
            n_1 = n_1 - line_lenght + line_setup_width
            random_line=1;
            special = random.randint(1,3);
            background = asarray(background)
            draw_lines([m_1, n_1+line_lenght+a2-line_setup_width], [m_1, n_1+1.5*line_lenght]) # Vertical down
            r_symbol, sb, r_label1, label1, r_label2, label2, r_label3, label3, r_specifier, specifier, dot, a1, a2 = load_annotations(dot, parallel_d, conclude_d, special)

            if special==1:
                draw_lines([m_1, n_1+1.5*line_lenght], [int(m_1-line_lenght/4), int(n_1+1.5*line_lenght)]) # Horizotal left
                background = Image.fromarray(background)
                draw_dot(sb, int(m_1-line_lenght/4), int(n_1+1.5*line_lenght))
                if save==True:
                    save_YOLO(r_symbol, r_label1, r_label2, r_label3, r_specifier, int(m_1-line_lenght/4), int(n_1+1.5*line_lenght), random_line, dot)
            elif special==2:
                draw_lines([m_1, n_1+1.5*line_lenght], [int(m_1+line_lenght/4), int(n_1+1.5*line_lenght)]) # Horizotal right
                background = Image.fromarray(background)
                draw_dot(sb, int(m_1+line_lenght/4), int(n_1+1.5*line_lenght))
                if save==True:
                    save_YOLO(r_symbol, r_label1, r_label2, r_label3, r_specifier, int(m_1+line_lenght/4), int(n_1+1.5*line_lenght), random_line, dot)
            else:
                draw_lines([m_1, n_1+1.5*line_lenght], [int(m_1+line_lenght/4), int(n_1+1.5*line_lenght)]) # Horizotal right
                background = Image.fromarray(background)
                draw_dot(sb, int(m_1+line_lenght/4), int(n_1+1.5*line_lenght))
                if save==True:
                    save_YOLO(r_symbol, r_label1, r_label2, r_label3, r_specifier, int(m_1+line_lenght/4), int(n_1+1.5*line_lenght), random_line, dot)
                background = asarray(background);
                draw_lines([int(m_1+line_lenght/4)+2, int(n_1+1.5*line_lenght)+a2], [int(m_1+line_lenght/4)+2, int(n_1+2.1*line_lenght)]) # Horizotal right
                background = Image.fromarray(background); special = 4;
                r_symbol, sb, r_label1, label1, r_label2, label2, r_label3, label3, r_specifier, specifier, dot, a1, a2 = load_annotations(dot, parallel_d, conclude_d, special)
                draw_obj(sb, label1, label2, label3, specifier, r_symbol, random_line, int(m_1+line_lenght/4)-8, int(n_1+2.2*line_lenght));
                if save==True:
                    save_YOLO(r_symbol, r_label1, r_label2, r_label3, r_specifier, int(m_1+line_lenght/4)-8, int(n_1+2.2*line_lenght), random_line, dot)
            background = asarray(background); a1=a1b; a2=a2b # keep the coordinates

        ###############################################################################
        ##################### Show the image every new symbol #########################
        ###############################################################################
        # Draw rectangle considering two dots
        background = asarray(background)
        if skip_symbol==False and dot==True and sym_i!=0:
            border_r = current_position
            cv2.rectangle(background, [border_l[0]-90, border_l[1]-60], [border_r[0]+78, border_r[1]+30], color=(0,0,0), thickness=2)  #color=(i*10,255-(i*10),0)
            border_l=current_position[0]+45,current_position[1]

        # Show image updated per epoch
        if img_show == True:
            cv2.imshow("Img", background)
            cv2.waitKey(500)
            cv2.destroyAllWindows()

        # If there is no symbol the starting point is the center
        if skip_symbol==True and sym_i!=0:
            a1=0; a2=0
        ###############################################################################
        ############################## Where to go then ###############################
        ###############################################################################
        if random_line==0:
            current_position = [m_1, n_1+a1]
            updated_position = [m_1, n_1+a1+line_lenght]
            draw_lines(current_position, updated_position); right=0
        else: # horizontal
            current_position = [m_1+a1, n_1]
            updated_position = [m_1+a1+line_lenght, n_1]
            draw_lines(current_position, updated_position); right=1

        #if i==0: # Save the initial position of the symbol
            #start = ((m_1-a1), (n_1)); random_line=0
        #save_lin.append([current_position, updated_position, [m_1, n_1, a2, a2]])
        current_position = updated_position # Update the position and the image
        background = Image.fromarray(background, "RGB")

    ###############################################################################
    ########################### Conclude the drawing ##############################
    ###############################################################################
    background = asarray(background)
    if random_line==1: # If it is horizontal go down to finish
        updated_position = [updated_position[0], updated_position[1]+int(line_lenght)]
    draw_lines(current_position, updated_position)
    m_1, n_1 = updated_position
    current_position = updated_position
    background = Image.fromarray(background)

    # Random initialization of drawing objects
    conclude_d = True; random_line=1
    r_symbol, sb, r_label1, label1, r_label2, label2, r_label3, label3, r_specifier, specifier, dot, a1, a2 = load_annotations(dot, parallel_d, conclude_d, special)
    draw_obj(sb, label1, label2, label3, specifier, r_symbol, random_line, m_1, n_1); line_lenght = line_lenght_s

    if save==True:
        save_YOLO(r_symbol, r_label1, r_label2, r_label3, r_specifier, m_1, n_1, random_line, dot)

    ###############################################################################
    ##################### Show the updated figure and save it #####################
    ###############################################################################
    background = asarray(background)
    if img_show == True:
        cv2.imshow("Img", background)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    # Save it
    filename: str = main + '/Final/Original/Synthetic_a_' + str(syn).zfill(3) + '.jpg'
    cv2.imwrite(filename, background)
    background2 = Image.fromarray(background, "RGB")
    if save==True:
        sys.stdout = orig_stdout
        f.close()

    ###############################################################################
    ################### Include the background noise ##############################
    ###############################################################################
    if include_noise == True:
        if save==True: # Save output as a text file (save annotations)
            shutil.copyfile(path_dia, path_dia[:19]+str('Normal')+path_dia[27:])

        img0 = background
        noise_img = noisy("gauss_white", img0)
        filename: str = main + '/Final/Output/Processing_' + str(syn).zfill(3) + '.jpg'
        cv2.imwrite(filename, noise_img)

        # Rewrite the drawing with the backgournd noise
        img = cv2.imread(filename)
        background = img.copy()       # Create a image copy
        if img_show == True:
            cv2.imshow("Img", background)
            cv2.waitKey(500)
            cv2.destroyAllWindows()

        for j in range(0,len(save_obj)):
            m_1, n_1 = save_obj[j][5], save_obj[j][6]
            r_symbol = save_obj[j][0]
            r_label1 = save_obj[j][1]
            r_label2 = save_obj[j][2]
            r_label3 = save_obj[j][3]
            r_specifier = save_obj[j][4]
            random_line = save_obj[j][7]
            dot = save_obj[j][8]
            special = save_obj[j][9]

            # Load images for annotations
            label1_names = glob.glob(path + str(r_label1) + str('/*.jpg'), recursive=True)
            label1 = (Image.open(label1_names[randint(0,len(label1_names)-1)])).resize(label_size)
            label2_names = glob.glob(path + str(r_label1) + str('/*.jpg'), recursive=True)
            label2 = (Image.open(label2_names[randint(0,len(label2_names)-1)])).resize(label_size)
            label3_names = glob.glob(path + str(r_label3) + str('/*.jpg'), recursive=True)
            label3 = (Image.open(label3_names[randint(0,len(label3_names)-1)])).resize(label_size)
            specifier_names = glob.glob(path + str(r_specifier) + str('/*.jpg'), recursive=True)
            specifier = (Image.open(specifier_names[randint(0, len(specifier_names)-1)])).resize(specifier_size)
            sb_names = glob.glob(path + str(r_symbol) + str('/*.jpg'), recursive=True)
            sb = Image.open(sb_names[randint(0,len(sb_names)-1)])

            ###############################################################################
            ############################# Draw objects ####################################
            ###############################################################################
            # Load only the image that have annotations
            if dot == False: # To fix the aspect ratio of symbols
                if sb.size[0] >= sb.size[1]:
                    sb = sb.resize((symbol_size, (int(symbol_size*sb.size[1]/sb.size[0]))))
                else:
                    sb = sb.resize((int(symbol_size*sb.size[0]/sb.size[1]), symbol_size))
            else: # To define the size of dots
                sb = sb.resize((int(symbol_size*0.7), int(symbol_size*0.7)))

            updated_p = update_position(m_1, n_1, r_symbol) # Rules to update the position
            if random_line==0:
                m_1 = m_1 + updated_p

            # Rotate the symbols if the line are vertical
            if (random_line!=0 and dot==False) or special!=0 or j==0:
                sb = sb
            else:
                sb = rotation(r_symbol, sb)
            a1 = int(sb.size[0]/2); a2 = int(sb.size[1]/2)
            background = Image.fromarray(background, "RGB")

            if dot==True:
                draw_dot(sb, m_1, n_1);
            else:
                draw_obj(sb, label1, label2, label3, specifier, r_symbol, random_line, m_1, n_1)

            # Show image updated by epoch
            background = asarray(background)
            if img_show == True:
                cv2.imshow("Img", background)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()

        # Save the final image
        filename: str = main + '/Final/Normal/Synthetic_a_' + str(syn).zfill(3) + '.jpg'
        cv2.imwrite(filename, background)

        if save_sp == True:
            if save==True: # Save output as a text file (save annotations)
                shutil.copyfile(path_dia, path_dia[:19]+str('Background')+path_dia[27:])
            img = cv2.imread(filename) # Save the final image with backgorund noice
            background = img.copy()    # Create a image copy
            filename: str = main + '/Final/Background/Synthetic_a_' + str(syn).zfill(3) + '.jpg'
            background_sp = noisy("s&p", background)
            cv2.imwrite(filename, background_sp)

        if save_names==True:
            f2.write(filename) # Save the image names
            f2.write('\n')
            print(filename)
    ##############################################################################
    ##############################################################################
    ##############################################################################

if save_names==True:
    f2.close()

if save_cropped==True:
    if size_p[0]>640:
        print('Start coordinates convertion...')
        exec(open('Save_640_640_slide_window_for_YOLO.py').read())
        print('Slide window done!')
        exec(open('Save_640_640_BBs_for_YOLO.py').read())
        print('BB convertion done!')
