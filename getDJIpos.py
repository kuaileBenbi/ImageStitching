from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def get_exif_data(image_path):
    """Extract EXIF data from an image file."""
    image = Image.open(image_path)
    exif_data = {}

    # Extract EXIF data
    exif_info = image._getexif()
    if exif_info:
        for tag, value in exif_info.items():
            decoded_tag = TAGS.get(tag, tag)
            if decoded_tag == "GPSInfo":
                gps_data = {}
                for gps_tag in value:
                    decoded_gps_tag = GPSTAGS.get(gps_tag, gps_tag)
                    gps_data[decoded_gps_tag] = value[gps_tag]
                exif_data[decoded_tag] = gps_data
            else:
                exif_data[decoded_tag] = value
    return exif_data

# Use the function
image_path = '/Users/lixiwang/Desktop/二维-2D/5cm-正射影像-2D-53pic/DJI_20230403143804_0017_V.JPG'
exif_data = get_exif_data(image_path)
print(exif_data)


