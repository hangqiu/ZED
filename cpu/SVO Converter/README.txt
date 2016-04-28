#####
#####               ===================================================
#####             '    ..                                         ..    `
#####            |    (  )             SVO Converter             (  )    |
#####             .                                                     ,
#####               =================================================== 
#####    
#####    
#####   This sample demonstrates how to grab images and depth with the ZED SDK from a SVO file,
#####   display aligned Left and Right images, and compute depth to record it in various formats (image, video...).   
#####    
#####   An SVO file must be specified in cmd arguments. (See Record sample or ZEDExplorer to save SVO files) 
#####    
#####   This sample need Boost, module progam_options.
#####   On windows it's advised to :
#####   - build it statically (with 'b2.exe link=static')
#####   - in visual studio project add '{YOUR_BOOST_DIR}/stage/lib' to the linker path
#####     and 'libboost_program_options-XXX.lib' to the dependency
#####   - in visual studio project also add the include dir ('{YOUR_BOOST_DIR}')
#####   -> That way boost is no longer a dependency
#####
#####   =====================
#####   * Build the program *
#####   =====================
#####    
#####       Open a terminal in SVO Converter directory and execute the following command:
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
#####           $ ./ZED\ SVO\ Converter [path to SVO file]    
#####    
#####   =====================
#####   * Available options *
#####   =====================
#####    
#####    ________________________________________________________________________________
#####   |                       |                                 |                      |
#####   |        Option         |           Description           |       Argument       |
#####   |=======================|=================================|======================|
#####   | --help                | Display help message            |                      |
#####   |_______________________|_________________________________|______________________|
#####   | -f,--filename         | SVO filename                    | path to an SVO file  |
#####   |_______________________|_________________________________|______________________|
#####   | -r,--record           | Record a sequence of images     |                      |
#####   |                       | Left+Disparity with -z option   |                      |
#####   |                       | Left+Right otherwise            |                      |
#####   |_______________________|_________________________________|______________________|
#####   | -v,--video            | Record a video file             | filename             |
#####   |                       | Left+Disparity with -z option   | WITH ".mp4" EXTENSION|
#####   |                       | Left+Right otherwise            |                      |
#####   |_______________________|_________________________________|______________________|
#####   | -s,--nodisplay        | Disable image display           |                      |
#####   |_______________________|_________________________________|______________________|
#####   | -z,--disparity        | Compute disparity               |                      |
#####   |_______________________|_________________________________|______________________|
#####   | -q,--quality          | Disparity Map quality           | '1': PERFORMANCE     |
#####   |                       | (-z needed)                     | '2': QUALITY         |
#####   |_______________________|_________________________________|______________________|
#####   | -d,--device           | CUDA device                     | int                  |
#####   |                       | (-z needed)                     |                      |
#####   |_______________________|_________________________________|______________________|
#####    
#####    
#####    
#####               
#####                          ######### ######### ######## 
###################                   #  #         #       #            ###################
###################                #     # ######  #       #            ###################
###################             #        #         #       #            ###################
                               #########  ######## ########   