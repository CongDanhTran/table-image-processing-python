import sys
sys.path.append("libs") 

import os
import cv2
import glob
import TableExtractor as te
from PIL import Image
from natsort import natsorted  # install via: pip install natsort

# Input and output folders
input_folder = "./image/"
output_folder = "./corrected/"
pdf_output_folder = "./pdfs/"

os.makedirs(pdf_output_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Loop through all jpg/png images
for img_path in glob.glob(os.path.join(input_folder, "*.*")):
    if img_path.lower().endswith((".jpg", ".jpeg", ".png")):
        print(f"Processing: {img_path}")

        # Extract file name only
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_folder, f"corrected_{filename}")

        # Run your table extractor
        table_extractor = te.TableExtractor(img_path)
        perspective_corrected_image = table_extractor.execute()

        # Save result
        cv2.imwrite(output_path, perspective_corrected_image)
        print(f"✅ Saved: {output_path}")


# Collect and naturally sort images
image_paths = natsorted(glob.glob(os.path.join(output_folder, "*.jpg")))

# Batch size
batch_size = 5

for i in range(0, len(image_paths), batch_size):
    batch = image_paths[i:i + batch_size]
    
    if len(batch) == 0:
        continue
    
    # Open and convert
    pil_images = [Image.open(img).convert("RGB") for img in batch]
    
    # Save PDF
    pdf_name = f"batch_{i//batch_size + 1}.pdf"
    pdf_path = os.path.join(pdf_output_folder, pdf_name)
    
    pil_images[0].save(pdf_path, save_all=True, append_images=pil_images[1:])
    print(f"✅ Saved: {pdf_path}")






# import OcrToTableTool as ottt
# import TableExtractor as te
# import TableLinesRemover as tlr
# import cv2

# path_to_image = "./image/202509010251_1.jpg"
# table_extractor = te.TableExtractor(path_to_image)
# perspective_corrected_image = table_extractor.execute()
# cv2.imwrite("perspective_corrected_image.jpg", perspective_corrected_image)


# # lines_remover = tlr.TableLinesRemover(perspective_corrected_image)
# # image_without_lines = lines_remover.execute()
# # cv2.imwrite("image_without_lines.jpg", image_without_lines)

# # ocr_tool = ottt.OcrToTableTool(image_without_lines, perspective_corrected_image)
# # ocr_tool.execute()

# cv2.waitKey(0)

