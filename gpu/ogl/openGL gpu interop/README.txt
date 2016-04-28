#####
#####               ===================================================
#####             '    ..                                         ..    `
#####            |    (  )           openGL gpu interop          (  )    |
#####             .                                                     ,
#####               ===================================================  
##### 
##### 
#####   This sample demonstrates the most efficient way to grab and display images and disparity map
#####   with the ZED SDK. No GPU->CPU readback is needed to display images, for RealTime monitoring.                    
##### 
#####   The GPU buffer is ingested directly into OpenGL texture for avoiding  GPU->CPU readback time.
##### 
#####   For the Left image, a GLSL shader is used for RGBA-->BGRA transformation , just as an example.
#####
#####   This sample needs freeGLUT (OpenGL Window) and GLEW (for GLSL).
#####
#####   =====================
#####   * Build the program *
#####   =====================
#####    
#####       Open a terminal in openGL gpu interop directory and execute the following command:
#####           $ mkdir build
#####           $ cd build
#####           $ cmake ..
#####           $ make
#####    
#####   ===================
#####   * Run the program *
#####   ===================
#####    
#####       Open a terminal in build directory and execute the following command:
#####           $ ./ZED\ openGL\ gpu\ interop 
#####     
##### 
##### 
#####        
#####                          ######### ######### ######## 
###################                   #  #         #       #            ###################
###################                #     # ######  #       #            ################### 
###################             #        #         #       #            ###################
                               #########  ######## ########