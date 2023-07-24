#@title Run object detection and show the detection results

INPUT_IMAGE_URL = "https://storage.googleapis.com/cloud-ml-data/img/openimage/3/2520/3916261642_0a504acd60_o.jpg" #@param {type:"string"}
DETECTION_THRESHOLD = 0.3 #@param {type:"number"} #! 如果小於DETECTION_THRESHOLD皆會不納入輸出

TEMP_FILE = '/tmp/image.png'

#* 將INPUT_IMAGE_URL下載到TEMP_FILE(指定的路徑=>如果沒有會在目前路徑)
!wget -q -O $TEMP_FILE $INPUT_IMAGE_URL #? -O == output
im = Image.open(TEMP_FILE)
im.thumbnail((512, 512), Image.ANTIALIAS) #? 改變大小(pixel)
im.save(TEMP_FILE, 'PNG')

# Load the TFLite model
#* 加載解釋、翻譯模型的工具 & 分配張量的內存空間
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Run inference and draw detection result on the local copy of the original file
#* 偵測照片 => bounding box
detection_result_image = run_odt_and_draw_results(
    TEMP_FILE,
    interpreter,
    threshold=DETECTION_THRESHOLD
)

# Show the detection result
#* 將一個NumPy array轉為image對象 & open image
img=Image.fromarray(detection_result_image)
img.show()