import argparse
from PIL import Image
parser = argparse.ArgumentParser(description='Overlap two images')
parser.add_argument('satimage', help='path to the satelite background image')
parser.add_argument('networkimage', help='path to the network front image')
parser.add_argument('outputimage', help='path to save the resulting image')
args = parser.parse_args()

sat_image_path = args.satimage
net_image_path = args.networkimage
out_image_path = args.outputimage

background = Image.open(sat_image_path)
overlay = Image.open(net_image_path)


background = background.convert("RGBA")
overlay = overlay.convert("RGBA")
overlay = overlay.resize(background.size)
print(background, overlay)

background.paste(overlay, (0,0), overlay)
background.save(out_image_path)