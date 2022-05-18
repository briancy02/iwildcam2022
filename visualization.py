from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from PIL import Image, ImageFile
from matplotlib import pyplot as plt



def visualize_seq(df, df_detection, seq_id, train_dir):
    img_rows = df.loc[df.seq_id == seq_id]
    images = []
    detections_list = []
    for index, img_row in img_rows.iterrows():
        detections = df_detection.loc[df_detection.file == ('train/'+img_row['file_name'])]['detections'].to_list()[0]
        image = Image.open(train_dir+img_row['file_name'])
        images.append(image)
        detections_list.append(detections)
    
    fig, ax = plt.subplots()
    anim = FuncAnimation(
        fig=fig,
        func=func_anim,
        fargs=(images,ax, detections_list),        
        frames=len(images),
        interval=4000,
        blit=False,
    )
    writergif = animation.PillowWriter(fps=3)
    anim.save(f"videos/{id}.gif", writer=writergif)

def func_anim(i, images, ax, detections):
    ax.imshow(images[i])
    [p.remove() for p in reversed(ax.patches)]
    for d in detections[i]:
        ax.add_patch(bbox_to_rect(d["bbox"], images[i].size, d["conf"]))
    return ax

def bbox_to_rect(bbox, im_size,conf):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (upper-left x, upper-left y, lower-right x,
    # lower-right y) format to the matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return Rectangle(
        xy=(bbox[0]*im_size[0], bbox[1]*im_size[1]), width=bbox[2]*im_size[0], height=bbox[3]*im_size[1],
        fill=False, edgecolor='red', linewidth=5*conf)