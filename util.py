import numpy as np

def color_dictionary():

    colors = dict()    

    ## define colors
    #blues  lightest to darkest
    blueVec1 = np.array([145,184,219]); colors['blue_1'] = blueVec1/256;
    blueVec2 = np.array([96,161,219]); colors['blue_2'] = blueVec2/256;
    blueVec3 = np.array([24,90,149]); colors['blue_3'] = blueVec3/256;
    blueVec4 = np.array([44,73,100]); colors['blue_4'] = blueVec4/256;
    blueVec5 = np.array([4,44,80]); colors['blue_5'] = blueVec5/256;
    #reds  lightest to darkest
    redVec1 = np.array([246,177,156]); colors['red_1'] = redVec1/256;
    redVec2 = np.array([246,131,98]); colors['red_2'] = redVec2/256;
    redVec3 = np.array([230,69,23]); colors['red_3'] = redVec3/256;
    redVec4 = np.array([154,82,61]); colors['red_4'] = redVec4/256;
    redVec5 = np.array([123,31,4]); colors['red_5'] = redVec5/256;
    #greens  lightest to darkest
    greenVec1 = np.array([142,223,180]); colors['green_1'] = greenVec1/256;
    greenVec2 = np.array([89,223,151]); colors['green_2'] = greenVec2/256;
    greenVec3 = np.array([16,162,84]); colors['green_3'] = greenVec3/256;
    greenVec4 = np.array([43,109,74]); colors['green_4'] = greenVec4/256;
    greenVec5 = np.array([3,87,42]); colors['green_5'] = greenVec5/256;
    #yellows  lightest to darkest
    yellowVec1 = np.array([246,204,156]); colors['yellow_1'] = yellowVec1/256;
    yellowVec2 = np.array([246,185,98]); colors['yellow_2'] = yellowVec2/256;
    yellowVec3 = np.array([230,144,23]); colors['yellow_3'] = yellowVec3/256;
    yellowVec4 = np.array([154,115,61]); colors['yellow_4'] = yellowVec4/256;
    yellowVec5 = np.array([123,74,4]); colors['yellow_5'] = yellowVec5/256;
    
    #blue grays
    gBlueVec1 = np.array([197,199,202]); colors['grey_blue_1'] = gBlueVec1/256;
    gBlueVec2 = np.array([195,198,202]); colors['grey_blue_2'] = gBlueVec2/256;
    gBlueVec3 = np.array([142,145,149]); colors['grey_blue_3'] = gBlueVec3/256;
    gBlueVec4 = np.array([108,110,111]); colors['grey_blue_4'] = gBlueVec4/256;
    gBlueVec5 = np.array([46,73,97]); colors['grey_blue_5'] = gBlueVec5/256;
    #red grays
    gRedVec1 = np.array([242,237,236]); colors['grey_red_1'] = gRedVec1/256;
    gRedVec2 = np.array([242,235,233]); colors['grey_red_2'] = gRedVec2/256;
    gRedVec3 = np.array([230,231,218]); colors['grey_red_3'] = gRedVec3/256;
    gRedVec4 = np.array([172,167,166]); colors['grey_red_4'] = gRedVec4/256;
    gRedVec5 = np.array([149,88,71]); colors['grey_red_5'] = gRedVec5/256;
    #green grays
    gGreenVec1 = np.array([203,209,206]); colors['grey_green_1'] = gGreenVec1/256;
    gGreenVec2 = np.array([201,209,204]); colors['grey_green_2'] = gGreenVec2/256;
    gGreenVec3 = np.array([154,162,158]); colors['grey_green_3'] = gGreenVec3/256;
    gGreenVec4 = np.array([117,122,119]); colors['grey_green_4'] = gGreenVec4/256;
    gGreenVec5 = np.array([50,105,76]); colors['grey_green_5'] = gGreenVec5/256;
    #yellow grays
    gYellowVec1 = np.array([242,240,236]); colors['grey_yellow_1'] = gYellowVec1/256;
    gYellowVec2 = np.array([242,239,233]); colors['grey_yellow_2'] = gYellowVec2/256;
    gYellowVec3 = np.array([230,225,218]); colors['grey_yellow_3'] = gYellowVec3/256;
    gYellowVec4 = np.array([172,169,166]); colors['grey_yellow_4'] = gYellowVec4/256;
    gYellowVec5 =np.array( [149,117,71]); colors['grey_yellow_5'] = gYellowVec5/256;
    
    #pure grays (white to black)
    gVec1 = np.array([256,256,256]); colors['grey_1'] = gVec1/256;
    gVec2 = np.array([242,242,242]); colors['grey_2'] = gVec2/256;
    gVec3 = np.array([230,230,230]); colors['grey_3'] = gVec3/256;
    gVec4 = np.array([204,204,204]); colors['grey_4'] = gVec4/256;
    gVec5 = np.array([179,179,179]); colors['grey_5'] = gVec5/256;
    gVec6 = np.array([153,153,153]); colors['grey_6'] = gVec6/256;
    gVec7 = np.array([128,128,128]); colors['grey_7'] = gVec7/256;
    gVec8 = np.array([102,102,102]); colors['grey_8'] = gVec8/256;
    gVec9 = np.array([77,77,77]); colors['grey_9'] = gVec9/256;
    gVec10 = np.array([51,51,51]); colors['grey_10'] = gVec10/256;
    gVec11 = np.array([26,26,26]); colors['grey_11'] = gVec11/256;
    gVec12 = np.array([0,0,0]); colors['grey_12'] = gVec12/256;
    
    return colors

