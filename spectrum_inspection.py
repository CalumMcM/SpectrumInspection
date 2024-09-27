import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import math

CLASS_NAMES = {
    0: 'person-inwater',
    1: 'person-notinwater',
    2: 'boat',
    3: 'anomaly',
    4: 'wind_turbine',
    5: 'offshore_platform',
    6: 'dummy',
    7: 'dummy-inwater',
    8: 'dummy-notinwater',
    9: 'dummy-falling',
    10: 'person-falling',
    11: 'person',
    12: 'liferaft'
}

def get_data(directory='labels/') -> dict:

    data = {}

    # iterate over files in
    # that directory
    label_datasets = [ f.path for f in os.scandir(directory) if f.is_dir() ]

    print (f"Datasets Found: {label_datasets}")

    for dataset_lbl_dir in label_datasets:

        for label_filename in os.listdir(dataset_lbl_dir):

            label_file_path = os.path.join(dataset_lbl_dir, label_filename)

            image_filename = label_filename.replace('_yolo.txt', '.jpg')
            image_file_path = os.path.join(dataset_lbl_dir, image_filename)
            image_file_path = image_file_path.replace('labels', 'images')

            # checking if it is a file
            if os.path.isfile(image_file_path) and image_filename.endswith('.jpg'):
                id = image_filename.removesuffix('.jpg')
                if id not in data:
                    data[id] = {}
                data[id]['image'] = image_file_path

            if os.path.isfile(label_file_path) and label_filename.endswith('.txt'):
                id = label_filename.removesuffix('_yolo.txt')
                if id not in data:
                    data[id] = {}
                data[id]['label'] = label_file_path
    return data


def read_labels(label_file):

    labels = []
    bboxes = []

    with open(label_file) as f:
        lines = f.readlines()

    for line in lines:
        label, bbox = get_bbox(line)
        labels.append(label)
        bboxes.append(bbox)

    return labels, bboxes


def get_bbox(label_file_line: str):
    
    contents = label_file_line.strip().split(' ')

    label = contents[0]

    # format: x, y, w, h
    bbox = (contents[1], contents[2], contents[3], contents[4])
    
    return label, bbox


def unnormalise(img, x, y, w, h):

    img_height, img_width, _ = img.shape
    normed = int(img_width*x), int(img_height*y), int(img_width*w), int(img_height*h)
    return normed


def crop_image_to_bbox(image, cx, cy, width, height):
    # Calculate top-left corner from center coordinates and width/height
    x1 = int(cx - width / 2)
    y1 = int(cy - height / 2)
    
    # Ensure coordinates are within image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    
    x2 = int(cx + width / 2)
    y2 = int(cy + height / 2)
    # Ensure the coordinates are within the image dimensions
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    
    # Crop the image
    cropped_image = image[y1:y2, x1:x2]
    
    return cropped_image


# Function to compute and plot color histograms
def plot_color_histogram(image, title, save_dir):
    
    # Convert the image from BGR to RGB (since OpenCV loads images as BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Split the image into its R, G, and B channels
    channels = ['Red', 'Green', 'Blue']
    colors = ('r', 'g', 'b')
    
    # Initialize the plot
    plt.figure(figsize=(10, 6))
    
    # For each color channel
    for i, color in enumerate(colors):
        # Compute the histogram for the channel
        histogram = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
        
        # Plot the histogram
        plt.plot(histogram, color=color)
        plt.xlim([0, 256])
    
    plt.title('Color Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend(channels)
    
    # Display the histogram plot
    plt.savefig(f'{save_dir}/{title}.png')


# Function to compute and plot the average color histogram
def plot_average_color_histogram(images, title, save_dir):
    # Initialize the histogram for each channel
    hist_r = np.zeros((256, 1))
    hist_g = np.zeros((256, 1))
    hist_b = np.zeros((256, 1))
    
    num_images = len(images)
    
    # Loop over each image path
    for image in images:
        
        # Convert the image from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Compute the histograms for each channel
        hist_r += cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
        hist_g += cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
        hist_b += cv2.calcHist([image_rgb], [2], None, [256], [0, 256])
        
    # Calculate the average histogram for each channel
    hist_r /= num_images
    hist_g /= num_images
    hist_b /= num_images
    
    # Plot the average histograms
    plt.figure(figsize=(10, 6))
    
    # Plot Red histogram
    plt.plot(hist_r, color='r', label='Red Channel')
    
    # Plot Green histogram
    plt.plot(hist_g, color='g', label='Green Channel')
    
    # Plot Blue histogram
    plt.plot(hist_b, color='b', label='Blue Channel')
    
    # Set plot limits and labels
    plt.xlim([0, 256])
    plt.ylim([0, 100])
    plt.title('Average Color Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    
    # Display the plot
    plt.savefig(f'{save_dir}/{title}.png')
               

def compute_average_color_histogram(images, title, save_dir):
    # Initialize the histogram for each channel
    hist_r = np.zeros((256, 1))
    hist_g = np.zeros((256, 1))
    hist_b = np.zeros((256, 1))
    
    num_images = len(images)
    
    def bincount_app(a):
        a2D = a.reshape(-1,a.shape[-1])
        col_range = (256, 256, 256) # generically : a2D.max(0)+1
        a1D = np.ravel_multi_index(a2D.T, col_range)
        return np.unravel_index(np.bincount(a1D).argmax(), col_range)

    # Loop over each image path
    for image in images:
        
        # Convert the image from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Compute the histograms for each channel
        hist_r += cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
        hist_g += cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
        hist_b += cv2.calcHist([image_rgb], [2], None, [256], [0, 256])
        
        # avg_color_per_row = np.mean(image_rgb, axis=0)
        # avg_color = np.mean(avg_color_per_row, axis=0) / np.sum(avg_color_per_row)
        dominant_colour = bincount_app(image_rgb) / np.sum(bincount_app(image_rgb))
    
    r, g, b = int(dominant_colour[0]), int(dominant_colour[1]), int(dominant_colour[2]) 

    # Calculate the average histogram for each channel
    hist_r /= num_images
    hist_g /= num_images
    hist_b /= num_images
    
    # Plot the average histograms
    plt.figure(figsize=(10, 6))
    
    # Plot Red histogram
    plt.hist(hist_r, alpha=0.5, color=(r,g,b), label='Red Channel')
    
    # Plot Green histogram
    plt.hist(hist_g, alpha=0.5, color='g', label='Green Channel')
    
    # Plot Blue histogram
    plt.hist(hist_b, alpha=0.5, color='b', label='Blue Channel')
    
    # Set plot limits and labels
    # plt.xlim([0, 256])
    # plt.ylim([0, 100])
    plt.title('Average Color Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    
    # Display the plot
    plt.savefig(f'{save_dir}/{title}.png')


def threeDImage(images, title, save_dir):

    # Plot the average histograms   
    plt.figure(figsize=(10, 6))

    ax = plt.axes(projection = '3d')

    x = []
    y = []
    z = []
    c = []

    pbar_local = tqdm(total=len(images))
    step = max(1, math.ceil(len(images)*0.05))
    print (f"Using Step size: {step}")
    for idx in range(0, len(images), step):
        img = images[idx]
        pbar_local.update(step)
        for row in range(0,img.shape[1]):
            for col in range(0, img.shape[0]):
                pix = img[col,row]
                newCol = (pix[0] / 255, pix[1] / 255, pix[2] / 255)

                if(not newCol in c):
                    x.append(pix[0])
                    y.append(pix[1])
                    z.append(pix[2])
                    c.append(newCol)

    ax.scatter(x,y,z, c = c)
    plt.savefig(f'{save_dir}/{title}.png')


def remove_background(image):
    
    # Create an initial mask
    mask = np.zeros(image.shape[:2], np.uint8)

    # Define a rectangle around the foreground (initial guess)
    height, width = image.shape[:2]
    rect = (10, 10, width-30, height-30)  # You can adjust these values to your image

    # Create temporary arrays used by GrabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Apply GrabCut algorithm
    cv2.grabCut(
        img=image, 
        mask=mask, 
        rect=None,
        bgdModel=bgdModel, 
        fgdModel=fgdModel, 
        iterCount=5, 
        mode=cv2.GC_INIT_WITH_MASK)

    # Modify the mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Create a masked image by multiplying the mask with the original image
    image_no_bg = image * mask2[:, :, np.newaxis]

    # Save the result
    cv2.imwrite('no_background_img.png', image_no_bg)

    return image_no_bg


def avg_img(images, title):
    avg_img = np.mean(images)
    cv2.imwrite(f'{title}.png', avg_img)

if __name__ == "__main__":
    directory = 'labels/'

    data = get_data(directory=directory)

    cropped_bboxes = {}

    with tqdm(total=len(data.keys())) as pbar:
        # Open each image and crop to 
        for frame_idx, frame in enumerate(data.keys()):
            
            if frame_idx < 10:
                continue
            pbar.update(1)

            if 'label' not in list(data[frame].keys()) or \
                'image' not in list(data[frame].keys()):
                continue

            img = cv2.imread(data[frame]['image'])
            labels, bboxes = read_labels(
                label_file=data[frame]['label'],
                )

            cv2.imwrite('image.png', img)
            for bbox_idx, bbox in enumerate(bboxes):

                bbox = unnormalise(
                    img=img,
                    x=float(bbox[0]),
                    y=float(bbox[1]),
                    w=float(bbox[2]),
                    h=float(bbox[3]),
                )

                bbox_size = get_bbox_size(bbox)

                cropped_bbox = crop_image_to_bbox(
                    image=img,
                    cx=int(bbox[0]),
                    cy=int(bbox[1]),
                    width=int(bbox[2]),
                    height=int(bbox[3]),
                    )
                
                cv2.imwrite('background_image.png', cropped_bbox)
                no_bg_image = remove_background(cropped_bbox)

                break
            
                if labels[bbox_idx] not in cropped_bboxes.keys():
                    cropped_bboxes[labels[bbox_idx]] = [cropped_bbox]
                else:
                    cropped_bboxes[labels[bbox_idx]].append(cropped_bbox)

    
    print ("plotting histograms per class")
    prev_img = None
    class_strs = [CLASS_NAMES[int(i)] for i in cropped_bboxes.keys()]
    print(f"Classes found: {class_strs}")

    for label in cropped_bboxes.keys():
        print (f"Graphing: {CLASS_NAMES[int(label)]}")
        title = f"{CLASS_NAMES[int(label)]}_3d_pixel_colours"
        
        threeDImage(
            images=cropped_bboxes[label], 
            title=title, 
            save_dir='histograms'
        )

        title = f"{CLASS_NAMES[int(label)]}_histogram_avg"
        
        plot_average_color_histogram(
            images=cropped_bboxes[label], 
            title=title, 
            save_dir='histograms'
        )
        

