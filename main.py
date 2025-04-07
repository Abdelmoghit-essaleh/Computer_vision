import cv2 
#from tkinter import * 
import tkinter as tk
from PIL import Image, ImageTk
from PIL import ImageFilter
from PIL import ImageEnhance #utiliser pour fair des amelioration sur l'image tel que limonositer d'image contarste
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pylab as pylab
from tkinter import filedialog
from skimage.morphology import binary_erosion, rectangle
from skimage.color import rgb2gray # convertir une image rgb en grey 
from skimage.io import imread, imsave
from skimage.util import img_as_float
from skimage.morphology import binary_dilation, disk

def beginwindow():
    window1= tk.Toplevel(root)
    window1.geometry("1000x500")
    window1.config(background='#41B77F')
    bopen= tk.Button(window1, text="operation_ponctuelle", command=operation_ponctuelle)
    bopen.pack()
    bopen= tk.Button(window1, text="filtes_pb", command=filtes_pb)
    bopen.pack()
    bopen= tk.Button(window1, text="filtes_ph", command=filtes_ph)
    bopen.pack()
    bopen= tk.Button(window1, text="Morphologie", command=Morphologie)
    bopen.pack()
    bopen=tk.Button(window1,text="Hought",command=hough_lines_detector)
    bopen.pack()
    bopen=tk.Button(window1,text="harris",command=harris)
    bopen.pack()
    bopen=tk.Button(window1,text="susan",command=susan)
    bopen.pack()
    bopen=tk.Button(window1,text="filtrage_frequentiel",command=filtrage_frequentiel)
    bopen.pack()


    
def operation_ponctuelle():
    window2= tk.Toplevel(root)
    window2.geometry("1000x500")
    window2.config(background='#41B77F')
    btn2 = tk.Button(window2, text="add",command=add)
    btn2.pack()
    btn3 = tk.Button (window2,text="histogramme",command=hist)
    btn3.pack()
    btn4=tk.Button(window2,text="(Luminosite)",command=ajuster_luminositer)
    btn4.pack()
    btn5=tk.Button(window2,text="ameliorer_contraste_linear",command=ameliorer_contraste_linear)
    btn5.pack()
    btn6=tk.Button(window2,text="ameliorer_contraste_avec_saturation",command=ameliorer_contraste_avec_saturation)
    btn6.pack()
    btn7=tk.Button(window2,text="histogram_equalization",command=histogram_equalization)
    btn7.pack()
    btn7=tk.Button(window2,text="decalage_additif",command=decalage_additif)
    btn7.pack()
    btn8=tk.Button(window2,text="decalage_multiplicatif",command=decalage_multiplicatif)
    btn8.pack()
    btn9=tk.Button(window2,text="inversion",command=inversion)
    btn9.pack()
    btn10=tk.Button(window2,text="ameliorer_contraste_decalage",command=ameliorer_contraste_decalage)
    btn10.pack()
    btn11=tk.Button(window2,text="seuillage",command=seuillage)
    btn11.pack()




def filtes_pb():
    window3= tk.Toplevel(root)
    window3.geometry("1000x500")
    window3.config(background='#41B77F')
    btn2 = tk.Button(window3, text="Filtre de moyenne ",command=moyenneur)
    btn2.pack()
    btn3 = tk.Button(window3, text="Filtre Gaussien ",command=gaussian)
    btn3.pack()
    btn3 = tk.Button(window3, text="Filtre pyramidal ",command=pyramidal)
    btn3.pack()
    btn4 = tk.Button(window3, text="Filtre conique ",command=conique)
    btn4.pack()
    btn5 = tk.Button(window3, text="Filtre median ",command=median)
    btn5.pack()
   

def filtes_ph():
    window4= tk.Toplevel(root)
    window4.geometry("1000x500")
    window4.config(background='#41B77F')
    btn2 = tk.Button(window4, text="Filtre par difference  ",command=ph_par_diff)
    btn2.pack()
    btn3 = tk.Button(window4, text="Filtre de moyenne passe haut  ",command=moyenne_ph)
    btn3.pack()
    btn4 = tk.Button(window4, text="Filtre de gradient sobel  ",command=gradient_sobel)
    btn4.pack()
    btn5 = tk.Button(window4, text="Filtre de gradient prewitt  ",command=gradient_prewitt)
    btn5.pack()
    btn6 = tk.Button(window4, text="Filtre de robert ",command=robert)
    btn6.pack()
    btn7 = tk.Button(window4, text="Filtre de laplacien ",command=laplacien)
    btn7.pack()
    btn8 = tk.Button(window4, text="Filtre de kirsch ",command=kirsch)
    btn8.pack()
    btn9 = tk.Button(window4, text="Filtre de kirsch_v2  ",command=kirsch_v2)
    btn9.pack()
    btn10 = tk.Button(window4, text="Filtre de marr_hildreth  ",command=marr_hildreth)
    btn10.pack()
    btn11 = tk.Button(window4, text="Filtre canny ",command=edg_canny)
    btn11.pack()

def Morphologie():
    window5= tk.Toplevel(root)
    window5.geometry("1000x500")
    window5.config(background='#41B77F')
    btn2 = tk.Button(window5, text="Erosion ",command=erosion)
    btn2.pack()
    btn3= tk.Button(window5, text="Dilatation",command=dilatation)
    btn3.pack()

def hough():
    rho_resolution = 1
    theta_resolution = np.pi/180
    threshold = 100
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Apply Canny edge detection
    edges = canny(image)
    
    # Initialize accumulator to store the votes for each line parameter combination
    height, width = edges.shape
    diagonal_length = np.sqrt(height ** 2 + width ** 2)
    rho_max = int(diagonal_length)  # Maximum possible rho value
    accumulator = np.zeros((2 * rho_max, int(np.pi / theta_resolution)), dtype=np.uint64)
    
    # Compute the sine and cosine of theta values
    theta_values = np.arange(0, np.pi, theta_resolution)
    cos_theta = np.cos(theta_values)
    sin_theta = np.sin(theta_values)
    
    # Find edge pixels and their coordinates
    edge_points = np.column_stack(np.nonzero(edges))
    
    # Vote in the accumulator for each edge point
    for rho_index in range(accumulator.shape[0]):
        rho = rho_index - rho_max
        for theta_index, (cos_val, sin_val) in enumerate(zip(cos_theta, sin_theta)):
            # Compute rho value for the given (rho, theta) pair
            rho_val = int(round(rho * cos_val + rho * sin_val))
            
            # Vote only if the rho value is positive
            if rho_val >= 0:
                accumulator[rho_val, theta_index] += 1
    
    # Find lines based on the accumulator
    lines = []
    for rho_index in range(accumulator.shape[0]):
        for theta_index in range(accumulator.shape[1]):
            if accumulator[rho_index, theta_index] > threshold:
                rho = rho_index - rho_max
                theta = theta_index * theta_resolution
                lines.append((rho, theta))
    
    return lines

def hough_lines_detector():
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    lines = hough()
    rho_resolution = 1
    theta_resolution =np.pi/180 
    threshold = 100
    min_line_length = 50
    max_line_gap = 10
    edges = canny(image)
    filtered_lines = cv2.HoughLinesP(edges, rho_resolution, theta_resolution, threshold,
                                     minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    # Draw detected lines on the original image
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if filtered_lines is not None:
        for line in filtered_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    
    # Display the original image and the result
    fig = plt.figure(figsize=(11, 6))
    plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray')
    plt.title('Original')
    plt.subplot(1, 3, 2), plt.imshow(result_image, cmap='gray')
    plt.title('Original')
    plt.show()

def harris():
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    dst = cv2.cornerHarris(image, blockSize=2, ksize=3, k=0.04)
    threshold = 0.01
    # Thresholding
    dst_thresh = np.zeros_like(dst)
    dst_thresh[dst > threshold * dst.max()] = 255
    
    # Convert thresholded image to uint8 for visualization
    dst_thresh = np.uint8(dst_thresh)
    
    # Mark detected corners on the original image
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    result_image[dst_thresh > 0] = [0, 0, 255]  # Red color for corners

    
    fig = plt.figure(figsize=(11, 6))
    plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray')
    plt.title('Original')
    plt.subplot(1, 3, 2), plt.imshow(result_image, cmap='gray')
    plt.title('result_image')
    plt.show()

def susan():
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    threshold=27 
    distance=3
    mask = np.array([[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 0, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]], dtype=np.uint8)
    
    # Compute number of similar pixels for each pixel using mask
    similar_pixels = cv2.filter2D(image, -1, mask, borderType=cv2.BORDER_CONSTANT)
    
    # Compute dissimilarity measure for each pixel
    dissimilarity = mask.size - similar_pixels
    
    # Threshold dissimilarity to identify corner points
    corner_mask = dissimilarity >= threshold
    
    # Get coordinates of corner points
    corner_points = np.argwhere(corner_mask)
    
    # Draw circles around corner points
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for point in corner_points:
        x, y = point[::-1]
        cv2.line(result_image, (x - distance, y), (x + distance, y), (0, 0, 255), 1)
        cv2.line(result_image, (x, y - distance), (x, y + distance), (0, 0, 255), 1)

    fig = plt.figure(figsize=(11, 6))
    plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray')
    plt.title('Original')
    plt.subplot(1, 3, 2), plt.imshow(result_image, cmap='gray')
    plt.title('result_image')
    plt.show()
def filtrage_frequentiel():
    window6= tk.Toplevel(root)
    window6.geometry("1000x500")
    window6.config(background='#41B77F')
    btn2 = tk.Button(window6, text="ideal_low_pass_filter_fourier ",command=ideal_low_pass_filter_fourier)
    btn2.pack()
    btn3= tk.Button(window6, text="butterworth",command=butterworth)
    btn3.pack()
    btn4= tk.Button(window6, text="high_pass_ideal_filter_fourier",command=high_pass_ideal_filter_fourier)
    btn4.pack()
def ideal_low_pass_filter_fourier():
    # Read the image as grayscale
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    cutoff_frequency = 70 
    # Compute the Fourier transform of the image
    f_transform = np.fft.fft2(image)
    
    # Shift the zero frequency component to the center
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20*np.log(np.abs(f_transform_shifted))
    
    # Get the dimensions of the image
    rows, cols = image.shape
    
    # Create a grid of frequencies
    crow, ccol = rows // 2, cols // 2
    x = np.arange(-crow, crow)
    y = np.arange(-ccol, ccol)
    x, y = np.meshgrid(x, y)
    distance = np.sqrt(x**2 + y**2)
    
    # Create the ideal low-pass filter mask
    mask = np.zeros((rows, cols), np.uint8)
    mask[distance <= cutoff_frequency] = 1
    
    # Apply the filter to the Fourier transformed image
    filtered_f_transform = f_transform_shifted * mask
    
    # Shift the spectrum back
    filtered_f_transform_shifted = np.fft.ifftshift(filtered_f_transform)
    
    # Apply the inverse Fourier transform
    filtered_image = np.fft.ifft2(filtered_f_transform_shifted)
    filtered_image = np.abs(filtered_image)  # Take the magnitude
    
    # Convert the filtered image to uint8 for display
    filtered_image = np.uint8(filtered_image)
    
    # Display the original and filtered images
    fig = plt.figure(figsize=(11, 6))
    plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray')
    plt.title('Original')
    plt.subplot(1, 3, 2), plt.imshow(magnitude_spectrum, cmap='gray'),plt.title('Spectre')
    plt.subplot(1, 3, 3), plt.imshow(filtered_image, cmap='gray')
    plt.title('result_image')
    plt.show()
    
def butterworth():
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
    # Compute the Fourier transform of the image
    f_transform = np.fft.fft2(image)
    
    # Shift the zero frequency component to the center
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20*np.log(np.abs(f_transform_shifted))
    # Get the dimensions of the image
    rows, cols = image.shape
    
    # Create a grid of frequencies
    crow, ccol = rows // 2, cols // 2
    x = np.arange(-crow, crow)
    y = np.arange(-ccol, ccol)
    x, y = np.meshgrid(x, y)
    distance = np.sqrt(x**2 + y**2)
    cutoff_frequency =20 
    order = 2
    # Create the Butterworth low-pass filter mask
    mask = 1 / (1 + (distance / cutoff_frequency) ** (2 * order))
    
    # Apply the filter to the Fourier transformed image
    filtered_f_transform = f_transform_shifted * mask
    
    # Shift the spectrum back
    filtered_f_transform_shifted = np.fft.ifftshift(filtered_f_transform)
    
    # Apply the inverse Fourier transform
    filtered_image = np.fft.ifft2(filtered_f_transform_shifted)
    filtered_image = np.abs(filtered_image)  # Take the magnitude
    
    # Convert the filtered image to uint8 for display
    filtered_image = np.uint8(filtered_image)
    
    fig = plt.figure(figsize=(11, 6))
    plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray')
    plt.title('Original')
    plt.subplot(1, 3, 2), plt.imshow(magnitude_spectrum, cmap='gray'),plt.title('Spectre')
    plt.subplot(1, 3, 3), plt.imshow(filtered_image, cmap='gray')
    plt.title('result_image')
    plt.show()


def high_pass_ideal_filter_fourier():
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
    # Perform Fourier Transform
    f_transform = np.fft.fft2(img)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20*np.log(np.abs(f_transform_shifted))
    cutoff_frequency = 30
    # Create a mask for high-pass ideal filter
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = 1
    
    # Apply the mask
    f_transform_shifted_filtered = f_transform_shifted * mask
    
    # Inverse Fourier Transform
    f_ishift = np.fft.ifftshift(f_transform_shifted_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    fig = plt.figure(figsize=(11, 6))
    plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.subplot(1, 3, 2), plt.imshow(magnitude_spectrum, cmap='gray'),plt.title('Spectre')
    plt.subplot(1, 3, 3), plt.imshow(img_back, cmap='gray')
    plt.title('result_image')
    plt.show()


def add():
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = Image.open(file_path)
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    plt.imshow(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
   
    plt.show()

def hist():
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = Image.open(file_path)
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Create a subplot with two columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the image in the first subplot
    ax1.imshow(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
    ax1.set_title("Image")
    ax1.axis("off")

    # Calculate and plot the histogram in the second subplot
    hist_color = ('b', 'g', 'r')
    for i, col in enumerate(hist_color):
        histogram = cv2.calcHist([image_cv2], [i], None, [256], [0, 256])
        ax2.plot(histogram, color=col)
    ax2.set_title("Histogram")
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Frequency")

    plt.show() 

def ajuster_luminositer():
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    gray_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if gray_image is None:
        print("Error: Unable to load image")
        return
    mean_value = np.mean(gray_image)
    std_dev = np.std(gray_image)
    print(f"Mean: {mean_value}")
    print(f"Standard Deviation: {std_dev}")
    print(gray_image.shape)

    brightness_factor=-50

    # Afficher l'image d'origine
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB))
    plt.title('Image Originale')
   

    # Afficher la moyenne de luminosité
    plt.subplot(1, 2, 2)
    bgr_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 2] = cv2.add(hsv_image[:, :, 2], brightness_factor)
    adjusted_bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    plt.imshow(cv2.cvtColor(adjusted_bgr_image, cv2.COLOR_BGR2RGB))
    plt.title('Moyenne Luminosité')
    plt.text(80, 350, f"Mean: {mean_value}\nStd Dev: {std_dev}\nShape: {gray_image.shape}", fontsize=10) 
    plt.show()

def ameliorer_contraste_linear():
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    
    if image is None:
        print("Error: Unable to load image")
        return

    
    min_val = np.min(image)
    max_val = np.max(image)

    
    enhanced_image = ((image - min_val) / (max_val - min_val)) * 255

    
    enhanced_image = np.uint8(enhanced_image)

    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image Originale')
   
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
    plt.title('Image amelioree')
    
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_cv3 = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)
    # Create a subplot with two columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    hist_color = ('b', 'g', 'r')
    for i, col in enumerate(hist_color):
        histogram1 = cv2.calcHist([image_cv2], [i], None, [256], [0, 256])
        histogram2 = cv2.calcHist([image_cv3], [i], None, [256], [0, 256])
        ax1.plot(histogram1, color=col)
        ax2.plot(histogram2, color=col)
    ax2.set_title("Histogram 2")
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Frequency")

    ax1.set_title("Histogram 1")
    ax1.set_xlabel("Pixel Value")
    ax1.set_ylabel("Frequency")
    plt.show()



def ameliorer_contraste_avec_saturation():
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Unable to load image")
        return

    
    stretched_image = cv2.normalize(image, None, alpha=1, beta=0, norm_type=cv2.NORM_MINMAX)

    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image Originale')
   
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(stretched_image, cv2.COLOR_BGR2RGB))
    plt.title('Image amelioree')
    plt.show()

def histogram_equalization():
    
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    
    if image is None:
        print("Error: Unable to load image")
        return

    
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    
    equalized_image = cdf[image]

    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image Originale')
   
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB))
    plt.title('Image amelioree')

    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_cv3 = cv2.cvtColor(np.array(equalized_image), cv2.COLOR_RGB2BGR)
    # Create a subplot with two columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    hist_color = ('b', 'g', 'r')
    for i, col in enumerate(hist_color):
        histogram1 = cv2.calcHist([image_cv2], [i], None, [256], [0, 256])
        histogram2 = cv2.calcHist([image_cv3], [i], None, [256], [0, 256])
        ax1.plot(histogram1, color=col)
        ax2.plot(histogram2, color=col)
    ax2.set_title("Histogram 2")
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Frequency")

    ax1.set_title("Histogram 1")
    ax1.set_xlabel("Pixel Value")
    ax1.set_ylabel("Frequency")

    
    plt.show()

def decalage_additif():
    
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    img_float = image.astype(np.float32)
    L=100
    img_adjusted = np.uint8(img_float + L)

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image Originale')
   
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_adjusted, cv2.COLOR_BGR2RGB))
    plt.title('Image amelioree')

    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_cv3 = cv2.cvtColor(np.array(img_adjusted), cv2.COLOR_RGB2BGR)
    # Create a subplot with two columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    hist_color = ('b', 'g', 'r')
    for i, col in enumerate(hist_color):
        histogram1 = cv2.calcHist([image_cv2], [i], None, [256], [0, 256])
        histogram2 = cv2.calcHist([image_cv3], [i], None, [256], [0, 256])
        ax1.plot(histogram1, color=col)
        ax2.plot(histogram2, color=col)
    ax2.set_title("Histogram 2")
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Frequency")

    ax1.set_title("Histogram 1")
    ax1.set_xlabel("Pixel Value")
    ax1.set_ylabel("Frequency")

    plt.show()

def decalage_multiplicatif():
    
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    img_float = image.astype(np.float32)
    L=1.5
    img_adjusted = np.uint8(img_float * L)

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image Originale')
   
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_adjusted, cv2.COLOR_BGR2RGB))
    plt.title('Image amelioree')

    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_cv3 = cv2.cvtColor(np.array(img_adjusted), cv2.COLOR_RGB2BGR)
    # Create a subplot with two columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    hist_color = ('b', 'g', 'r')
    for i, col in enumerate(hist_color):
        histogram1 = cv2.calcHist([image_cv2], [i], None, [256], [0, 256])
        histogram2 = cv2.calcHist([image_cv3], [i], None, [256], [0, 256])
        ax1.plot(histogram1, color=col)
        ax2.plot(histogram2, color=col)
    ax2.set_title("Histogram 2")
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Frequency")

    ax1.set_title("Histogram 1")
    ax1.set_xlabel("Pixel Value")
    ax1.set_ylabel("Frequency")

    plt.show()



def inversion():
    
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    img_float = image.astype(np.float32)
    img_adjusted = np.uint8(-1 * img_float + 255)

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image Originale')
   
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_adjusted, cv2.COLOR_BGR2RGB))
    plt.title('Image amelioree')

    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_cv3 = cv2.cvtColor(np.array(img_adjusted), cv2.COLOR_RGB2BGR)
    # Create a subplot with two columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    hist_color = ('b', 'g', 'r')
    for i, col in enumerate(hist_color):
        histogram1 = cv2.calcHist([image_cv2], [i], None, [256], [0, 256])
        histogram2 = cv2.calcHist([image_cv3], [i], None, [256], [0, 256])
        ax1.plot(histogram1, color=col)
        ax2.plot(histogram2, color=col)
    ax2.set_title("Histogram 2")
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Frequency")

    ax1.set_title("Histogram 1")
    ax1.set_xlabel("Pixel Value")
    ax1.set_ylabel("Frequency")

    plt.show()
    

def ameliorer_contraste_decalage():
    
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
    A = float(np.min(image))
    B = float(np.max(image))
    
    P = 255 / (B - A)
    L = -P * A

    img_float = image.astype(np.float32)
    img_adjusted = np.uint8(P * (img_float) + L)

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image Originale')
   
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_adjusted, cv2.COLOR_BGR2RGB))
    plt.title('Image amelioree')

    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_cv3 = cv2.cvtColor(np.array(img_adjusted), cv2.COLOR_RGB2BGR)
    # Create a subplot with two columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    hist_color = ('b', 'g', 'r')
    for i, col in enumerate(hist_color):
        histogram1 = cv2.calcHist([image_cv2], [i], None, [256], [0, 256])
        histogram2 = cv2.calcHist([image_cv3], [i], None, [256], [0, 256])
        ax1.plot(histogram1, color=col)
        ax2.plot(histogram2, color=col)
    ax2.set_title("Histogram 2")
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Frequency")

    ax1.set_title("Histogram 1")
    ax1.set_xlabel("Pixel Value")
    ax1.set_ylabel("Frequency")

    plt.show() 


def seuillage():

    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    threshold_value = 128 
    img_binary = image.copy()
    for i in range(img_binary.shape[0]):
        for j in range(img_binary.shape[1]):
            if image[i, j] > threshold_value:
                img_binary[i, j] = 255  # White (binary)
            else:
                img_binary[i, j] = 0    # Black (binary)

    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image Originale')
   
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_binary, cv2.COLOR_BGR2RGB))
    plt.title('Image amelioree')
    plt.show()


def moyenneur():
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    kernel_5 = np.array([
                        [1/25, 1/25, 1/25, 1/25, 1/25],
                        [1/25, 1/25, 1/25, 1/25, 1/25],
                        [1/25, 1/25, 1/25, 1/25, 1/25],
                        [1/25, 1/25, 1/25, 1/25, 1/25],
                        [1/25, 1/25, 1/25, 1/25, 1/25]
                                        ])
    kernel_3 = np.array([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ])

    kernel_size = 5

    if kernel_size == 3:
        img_filtered = cv2.filter2D(image, -1, kernel_3)
    elif kernel_size == 5:
        img_filtered = cv2.filter2D(image, -1, kernel_5)
    else :
        print("not available !!!")
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image Originale')
   
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_filtered, cv2.COLOR_BGR2RGB))
    plt.title('Image filtrer')
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_cv3 = cv2.cvtColor(np.array(img_filtered), cv2.COLOR_RGB2BGR)
    # Create a subplot with two columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    hist_color = ('b', 'g', 'r')
    for i, col in enumerate(hist_color):
        histogram1 = cv2.calcHist([image_cv2], [i], None, [256], [0, 256])
        histogram2 = cv2.calcHist([image_cv3], [i], None, [256], [0, 256])
        ax1.plot(histogram1, color=col)
        ax2.plot(histogram2, color=col)
    ax2.set_title("Histogram 2")
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Frequency")

    ax1.set_title("Histogram 1")
    ax1.set_xlabel("Pixel Value")
    ax1.set_ylabel("Frequency")
    plt.show()


def gaussian():
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    kernel_5 = np.array([
                        [1/256, 4/256, 6/256, 4/256, 1/256],
                        [4/256, 16/256, 24/256, 16/256, 4/256],
                        [6/256, 24/256, 36/256, 24/256, 6/256],
                        [4/256, 16/256, 24/256, 16/256, 4/256],
                        [1/256, 4/256, 6/256, 4/256, 1/256]
                                        ])
    kernel_3 = np.array([
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16]
    ])
    kernel_size = 5 

    if kernel_size == 3:
        img_filtered = cv2.filter2D(image, -1, kernel_3)
    elif kernel_size == 5:
        img_filtered = cv2.filter2D(image, -1, kernel_5)
    else :
        print("not available !!!")
    

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image Originale')
   
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_filtered, cv2.COLOR_BGR2RGB))
    plt.title('Image filtrer')
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_cv3 = cv2.cvtColor(np.array(img_filtered), cv2.COLOR_RGB2BGR)
    # Create a subplot with two columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    hist_color = ('b', 'g', 'r')
    for i, col in enumerate(hist_color):
        histogram1 = cv2.calcHist([image_cv2], [i], None, [256], [0, 256])
        histogram2 = cv2.calcHist([image_cv3], [i], None, [256], [0, 256])
        ax1.plot(histogram1, color=col)
        ax2.plot(histogram2, color=col)
    ax2.set_title("Histogram 2")
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Frequency")

    ax1.set_title("Histogram 1")
    ax1.set_xlabel("Pixel Value")
    ax1.set_ylabel("Frequency")
    plt.show()
    

def pyramidal ():
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    kernel = np.array([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ])
    
    kernel_pyramidal = cv2.filter2D(kernel, -1, kernel)
    
    img_filtered = cv2.filter2D(image, -1, kernel_pyramidal)
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image Originale')
   
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_filtered, cv2.COLOR_BGR2RGB))
    plt.title('Image filtrer')
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_cv3 = cv2.cvtColor(np.array(img_filtered), cv2.COLOR_RGB2BGR)
    # Create a subplot with two columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    hist_color = ('b', 'g', 'r')
    for i, col in enumerate(hist_color):
        histogram1 = cv2.calcHist([image_cv2], [i], None, [256], [0, 256])
        histogram2 = cv2.calcHist([image_cv3], [i], None, [256], [0, 256])
        ax1.plot(histogram1, color=col)
        ax2.plot(histogram2, color=col)
    ax2.set_title("Histogram 2")
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Frequency")

    ax1.set_title("Histogram 1")
    ax1.set_xlabel("Pixel Value")
    ax1.set_ylabel("Frequency")
    plt.show()


def conique ():
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    kernel = np.array([
        [0, 0, 1/25, 0, 0],
        [0, 2/25, 2/25, 2/25, 0],
        [1/25, 2/25, 5/25, 2/25, 1/25],
        [0, 2/25, 2/25, 2/25, 0],
        [0, 0, 1/25, 0, 0]
    ])
    
    img_filtered = cv2.filter2D(image, -1, kernel)

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image Originale')
   
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_filtered, cv2.COLOR_BGR2RGB))
    plt.title('Image filtrer')
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_cv3 = cv2.cvtColor(np.array(img_filtered), cv2.COLOR_RGB2BGR)
    # Create a subplot with two columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    hist_color = ('b', 'g', 'r')
    for i, col in enumerate(hist_color):
        histogram1 = cv2.calcHist([image_cv2], [i], None, [256], [0, 256])
        histogram2 = cv2.calcHist([image_cv3], [i], None, [256], [0, 256])
        ax1.plot(histogram1, color=col)
        ax2.plot(histogram2, color=col)
    ax2.set_title("Histogram 2")
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Frequency")

    ax1.set_title("Histogram 1")
    ax1.set_xlabel("Pixel Value")
    ax1.set_ylabel("Frequency")
    plt.show()


def  median():
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    height, width = image.shape
    output = np.zeros((height, width), dtype=np.uint8)

    kernel_size = 5 

    padded_image = np.pad(image, ((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2)), mode='constant')

    for i in range(height):
        for j in range(width):
            
            neighborhood = padded_image[i:i+kernel_size, j:j+kernel_size]

            
            flattened_neighborhood = neighborhood.flatten()

            
            sorted_neighborhood = np.sort(flattened_neighborhood)

            
            median_index = len(sorted_neighborhood) // 2

            
            median_value = sorted_neighborhood[median_index]

            
            output[i, j] = median_value
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image Originale')
   
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title('Image filtrer')
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_cv3 = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
    # Create a subplot with two columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    hist_color = ('b', 'g', 'r')
    for i, col in enumerate(hist_color):
        histogram1 = cv2.calcHist([image_cv2], [i], None, [256], [0, 256])
        histogram2 = cv2.calcHist([image_cv3], [i], None, [256], [0, 256])
        ax1.plot(histogram1, color=col)
        ax2.plot(histogram2, color=col)
    ax2.set_title("Histogram 2")
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Frequency")

    ax1.set_title("Histogram 1")
    ax1.set_xlabel("Pixel Value")
    ax1.set_ylabel("Frequency")
    plt.show()

def ph_par_diff():
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
    
    kernel_5 = np.array([
                        [1/25, 1/25, 1/25, 1/25, 1/25],
                        [1/25, 1/25, 1/25, 1/25, 1/25],
                        [1/25, 1/25, 1/25, 1/25, 1/25],
                        [1/25, 1/25, 1/25, 1/25, 1/25],
                        [1/25, 1/25, 1/25, 1/25, 1/25]
                                        ])
    kernel_3 = np.array([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ])
   
    kernel_size = 5

    if kernel_size == 3:
        to_substract = cv2.filter2D(image, -1, kernel_3)
        final = image.astype(np.float32) - to_substract.astype(np.float32)
        final = np.abs(final).astype(np.uint8)
    elif kernel_size == 5:
        to_substract = cv2.filter2D(image, -1, kernel_5)
        final = image.astype(np.float32) - to_substract.astype(np.float32)
        final = np.abs(final).astype(np.uint8)
    else:
        print("Not Available!!!")
    

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image Originale')
   
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    plt.title('Image filtrer')
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_cv3 = cv2.cvtColor(np.array(final), cv2.COLOR_RGB2BGR)
    # Create a subplot with two columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    hist_color = ('b', 'g', 'r')
    for i, col in enumerate(hist_color):
        histogram1 = cv2.calcHist([image_cv2], [i], None, [256], [0, 256])
        histogram2 = cv2.calcHist([image_cv3], [i], None, [256], [0, 256])
        ax1.plot(histogram1, color=col)
        ax2.plot(histogram2, color=col)
    ax2.set_title("Histogram 2")
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Frequency")

    ax1.set_title("Histogram 1")
    ax1.set_xlabel("Pixel Value")
    ax1.set_ylabel("Frequency")
    plt.show()


def moyenne_ph ():
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    kernel_5 = np.array([
                        [-1/25, -1/25, -1/25, -1/25, -1/25],
                        [-1/25, -1/25, -1/25, -1/25, -1/25],
                        [-1/25, -1/25, 24/25, -1/25, -1/25],
                        [-1/25, -1/25, -1/25, -1/25, -1/25],
                        [-1/25, -1/25, -1/25, -1/25, -1/25]
                                        ])
    kernel_3 = np.array([
        [-1/9, -1/9, -1/9],
        [-1/9, 8/9, -1/9],
        [-1/9, -1/9, -1/9]
    ])
    
    kernel_size = 5
    if kernel_size == 3:
        final = cv2.filter2D(image, -1, kernel_3)
        
    elif kernel_size == 5:
        final = cv2.filter2D(image, -1, kernel_5)
        
    else:
        print("Not Available!!!")
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image Originale')
   
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    plt.title('Image filtrer')
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_cv3 = cv2.cvtColor(np.array(final), cv2.COLOR_RGB2BGR)
    # Create a subplot with two columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    hist_color = ('b', 'g', 'r')
    for i, col in enumerate(hist_color):
        histogram1 = cv2.calcHist([image_cv2], [i], None, [256], [0, 256])
        histogram2 = cv2.calcHist([image_cv3], [i], None, [256], [0, 256])
        ax1.plot(histogram1, color=col)
        ax2.plot(histogram2, color=col)
    ax2.set_title("Histogram 2")
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Frequency")

    ax1.set_title("Histogram 1")
    ax1.set_xlabel("Pixel Value")
    ax1.set_ylabel("Frequency")
    plt.show()


def gradient_sobel():
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    kernel_X = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    kernel_Y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    img_x = cv2.filter2D(image, -1, kernel_X)
    final = cv2.filter2D(img_x, -1, kernel_Y)

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image Originale')
   
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    plt.title('Image filtrer')
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_cv3 = cv2.cvtColor(np.array(final), cv2.COLOR_RGB2BGR)
    # Create a subplot with two columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    hist_color = ('b', 'g', 'r')
    for i, col in enumerate(hist_color):
        histogram1 = cv2.calcHist([image_cv2], [i], None, [256], [0, 256])
        histogram2 = cv2.calcHist([image_cv3], [i], None, [256], [0, 256])
        ax1.plot(histogram1, color=col)
        ax2.plot(histogram2, color=col)
    ax2.set_title("Histogram 2")
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Frequency")

    ax1.set_title("Histogram 1")
    ax1.set_xlabel("Pixel Value")
    ax1.set_ylabel("Frequency")
    plt.show()

def gradient_prewitt ():
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    kernel_X = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])
    kernel_Y = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ])
    img_x = cv2.filter2D(image, -1, kernel_X)
    final = cv2.filter2D(img_x, -1, kernel_Y)

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image Originale')
   
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    plt.title('Image filtrer')
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_cv3 = cv2.cvtColor(np.array(final), cv2.COLOR_RGB2BGR)
    # Create a subplot with two columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    hist_color = ('b', 'g', 'r')
    for i, col in enumerate(hist_color):
        histogram1 = cv2.calcHist([image_cv2], [i], None, [256], [0, 256])
        histogram2 = cv2.calcHist([image_cv3], [i], None, [256], [0, 256])
        ax1.plot(histogram1, color=col)
        ax2.plot(histogram2, color=col)
    ax2.set_title("Histogram 2")
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Frequency")

    ax1.set_title("Histogram 1")
    ax1.set_xlabel("Pixel Value")
    ax1.set_ylabel("Frequency")
    plt.show()


def robert ():
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    kernel_X = np.array([
        [ 0, -1],
        [1, 0]
        
    ])
    kernel_Y = np.array([
        [-1, 0],
        [0, 1]
        
    ])
    img_x = cv2.filter2D(image, -1, kernel_X)
    final = cv2.filter2D(img_x, -1, kernel_Y)

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image Originale')
   
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    plt.title('Image filtrer')
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_cv3 = cv2.cvtColor(np.array(final), cv2.COLOR_RGB2BGR)
    # Create a subplot with two columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    hist_color = ('b', 'g', 'r')
    for i, col in enumerate(hist_color):
        histogram1 = cv2.calcHist([image_cv2], [i], None, [256], [0, 256])
        histogram2 = cv2.calcHist([image_cv3], [i], None, [256], [0, 256])
        ax1.plot(histogram1, color=col)
        ax2.plot(histogram2, color=col)
    ax2.set_title("Histogram 2")
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Frequency")

    ax1.set_title("Histogram 1")
    ax1.set_xlabel("Pixel Value")
    ax1.set_ylabel("Frequency")
    plt.show()

def laplacien ():

    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    kernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])

    final = cv2.filter2D(image, -1, kernel)

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image Originale')
   
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    plt.title('Image filtrer')
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_cv3 = cv2.cvtColor(np.array(final), cv2.COLOR_RGB2BGR)
    # Create a subplot with two columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    hist_color = ('b', 'g', 'r')
    for i, col in enumerate(hist_color):
        histogram1 = cv2.calcHist([image_cv2], [i], None, [256], [0, 256])
        histogram2 = cv2.calcHist([image_cv3], [i], None, [256], [0, 256])
        ax1.plot(histogram1, color=col)
        ax2.plot(histogram2, color=col)
    ax2.set_title("Histogram 2")
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Frequency")

    ax1.set_title("Histogram 1")
    ax1.set_xlabel("Pixel Value")
    ax1.set_ylabel("Frequency")
    plt.show()

def kirsch ():

    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    kernel = np.array([[ -3,  -3, -3],
                        [ 5,  0, -3],
                        [ 5, 5, -3]])

    final = cv2.filter2D(image, -1, kernel)

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image Originale')
   
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    plt.title('Image filtrer')
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_cv3 = cv2.cvtColor(np.array(final), cv2.COLOR_RGB2BGR)
    # Create a subplot with two columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    hist_color = ('b', 'g', 'r')
    for i, col in enumerate(hist_color):
        histogram1 = cv2.calcHist([image_cv2], [i], None, [256], [0, 256])
        histogram2 = cv2.calcHist([image_cv3], [i], None, [256], [0, 256])
        ax1.plot(histogram1, color=col)
        ax2.plot(histogram2, color=col)
    ax2.set_title("Histogram 2")
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Frequency")

    ax1.set_title("Histogram 1")
    ax1.set_xlabel("Pixel Value")
    ax1.set_ylabel("Frequency")
    plt.show()


def kirsch_v2():

    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    kernel_north = np.array([[-3, -3, -3],
                         [ 0,  0,  0],
                         [ 3,  3,  3]])

    kernel_northeast = np.array([[-3, -3,  0],
                             [-3,  0,  3],
                             [ 0,  3,  3]])

    kernel_east = np.array([[-3,  0,  3],
                        [-3,  0,  3],
                        [-3,  0,  3]])

    kernel_southeast = np.array([[ 0,  3,  3],
                             [-3,  0,  3],
                             [-3, -3,  0]])

    kernel_south = np.array([[ 3,  3,  3],
                         [ 0,  0,  0],
                         [-3, -3, -3]])

    kernel_southwest = np.array([[ 0,  3,  3],
                             [ 3,  0, -3],
                             [ 3, -3, -3]])

    kernel_west = np.array([[ 3,  0, -3],
                        [ 3,  0, -3],
                        [ 3,  0, -3]])

    kernel_northwest = np.array([[ 3,  3,  0],
                              [ 3,  0, -3],
                              [ 0, -3, -3]])
    
    final = cv2.filter2D(image, -1, kernel_north)
    final = cv2.filter2D(final, -1, kernel_northeast)
    final = cv2.filter2D(final, -1, kernel_east)
    final = cv2.filter2D(final, -1, kernel_southeast)
    final = cv2.filter2D(final, -1, kernel_south)
    final = cv2.filter2D(final, -1, kernel_southwest)
    final = cv2.filter2D(final, -1, kernel_west)
    final = cv2.filter2D(final, -1, kernel_northwest)

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image Originale')
   
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    plt.title('Image filtrer')
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_cv3 = cv2.cvtColor(np.array(final), cv2.COLOR_RGB2BGR)
    # Create a subplot with two columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    hist_color = ('b', 'g', 'r')
    for i, col in enumerate(hist_color):
        histogram1 = cv2.calcHist([image_cv2], [i], None, [256], [0, 256])
        histogram2 = cv2.calcHist([image_cv3], [i], None, [256], [0, 256])
        ax1.plot(histogram1, color=col)
        ax2.plot(histogram2, color=col)
    ax2.set_title("Histogram 2")
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Frequency")

    ax1.set_title("Histogram 1")
    ax1.set_xlabel("Pixel Value")
    ax1.set_ylabel("Frequency")
    plt.show()

def marr_hildreth ():

    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    kernel = np.array([[-1, -3, -4, -3, -1],
                        [-3, 0, 6, 0, -3],
                        [-4, 6, 20, 6, -4],
                        [-3, 0, 6, 0, -3],
                        [-1, -3, -4, -3, -1]])

    final = cv2.filter2D(image, -1, kernel)

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image Originale')
   
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    plt.title('Image filtrer')
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_cv3 = cv2.cvtColor(np.array(final), cv2.COLOR_RGB2BGR)
    # Create a subplot with two columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    hist_color = ('b', 'g', 'r')
    for i, col in enumerate(hist_color):
        histogram1 = cv2.calcHist([image_cv2], [i], None, [256], [0, 256])
        histogram2 = cv2.calcHist([image_cv3], [i], None, [256], [0, 256])
        ax1.plot(histogram1, color=col)
        ax2.plot(histogram2, color=col)
    ax2.set_title("Histogram 2")
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Frequency")

    ax1.set_title("Histogram 1")
    ax1.set_xlabel("Pixel Value")
    ax1.set_ylabel("Frequency")
    plt.show()

def canny(image):
    # Appliquer l'algorithme Canny à l'image
    canny_edges = cv2.Canny(image, 50, 150)

    return canny_edges
def edg_canny():

    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
    canny_ed = canny(image)

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image Originale')
   
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(canny_ed, cv2.COLOR_BGR2RGB))
    plt.title('Image filtrer')
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_cv3 = cv2.cvtColor(np.array(canny_ed), cv2.COLOR_RGB2BGR)
    # Create a subplot with two columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    hist_color = ('b', 'g', 'r')
    for i, col in enumerate(hist_color):
        histogram1 = cv2.calcHist([image_cv2], [i], None, [256], [0, 256])
        histogram2 = cv2.calcHist([image_cv3], [i], None, [256], [0, 256])
        ax1.plot(histogram1, color=col)
        ax2.plot(histogram2, color=col)
    ax2.set_title("Histogram 2")
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Frequency")

    ax1.set_title("Histogram 1")
    ax1.set_xlabel("Pixel Value")
    ax1.set_ylabel("Frequency")
    plt.show()



def erosion():
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    var = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

   
    var_rgb = cv2.cvtColor(var, cv2.COLOR_BGR2RGB)


    im = rgb2gray(var_rgb)

    im[im <= 0.5] = 0
    im[im > 0.5] = 1


    plt.subplot(1, 3, 1), plt.imshow(im, cmap='gray')
    plt.title('Original')


    im1 = binary_erosion(im, rectangle(1, 5))
    plt.subplot(1, 3, 2), plt.imshow(im1, cmap='gray')
    plt.title('Erosion (1, 5)')


    im2 = binary_erosion(im, rectangle(1, 15))
    plt.subplot(1, 3, 3), plt.imshow(im2, cmap='gray')
    plt.title('Erosion (1, 15)')


    plt.show()


def dilatation():
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures"
    )
    var = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Pas besoin de rgb2gray pour une image en niveaux de gris
    #im = 1 - var / 255.0  # Normaliser les valeurs entre 0 et 1
    im = img_as_float(var)
    im[im <= 0.5] = 0
    im[im > 0.5] = 1

    plt.gray()

    # Utilisation de plt.imshow() avec cmap='gray' pour une image en niveaux de gris
    plt.subplot(1, 3, 1), plt.imshow(im, cmap='gray')
    plt.title('Original')

    im1 = binary_dilation(im, disk(2))
    plt.subplot(1, 3, 2), plt.imshow(im1, cmap='gray')
    plt.title('Dilatation size 2')

    im2 = binary_dilation(im, disk(4))
    plt.subplot(1, 3, 3), plt.imshow(im2, cmap='gray')
    plt.title('Dilatation size 4')

    plt.show()






 

root = tk.Tk()
root.title("Image_processing")
root.geometry("1200x700")
root.config(background='#41B77F')
label_title = tk.Label(root,text='Welcome',bg='#41B77F', fg='white')
label_title.pack()

bopen= tk.Button(root, text="Ouvrir", command=beginwindow)
bopen.pack()

bclose=tk.Button(root,text="Sortir",command=root.destroy)
bclose.pack(padx=20,pady=20)

root.mainloop()

