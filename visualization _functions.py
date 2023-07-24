# @title Load the trained TFLite model and define some visualization functions

import cv2

from PIL import Image

model_path = 'model.tflite'

# Load the labels into a list
# ! 將model的config中將class的數量拿出來*['???'] => 表示classes會有class數量的['???'] array
classes = ['???'] * model.model_spec.config.num_classes
label_map = model.model_spec.config.label_map
for label_id, label_name in label_map.as_dict().items():
    classes[label_id-1] = label_name

# Define a list of colors for visualization
#! 生成指定範圍內的隨機數(0, 255) => size=(len(classes), 3)表示行列式 => 3是RGB、len(classes)是表示每一類別有一種RGB
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)


def preprocess_image(image_path, input_size):
    """Preprocess the input image to feed to the TFLite model"""
    # *
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    original_image = img
    # ? input_size(turple) => (high, width)
    resized_img = tf.image.resize(img, input_size)
    # * 给调整后的图像添加一个新的维度，这样图像就变成了一个四维张量（[batch_size, height, width, channels]），其中 batch_size 为 1，表示这是单个图像
    resized_img = resized_img[tf.newaxis, :]
    # ? 因不改變大小，只強制轉換類型，所以用cast
    resized_img = tf.cast(resized_img, dtype=tf.uint8)
    return resized_img, original_image


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""

    # * 用於運行推斷過程並輸出結果的工具
    signature_fn = interpreter.get_signature_runner()

    # Feed the input image to the model
    # * 用預處理的照片去做推測 => 包含array的dic
    output = signature_fn(images=image)

    # Get all outputs from the model
    # * np.squeeze => 降維度 => 等於1的都去掉 => 只剩下數值的array
    count = int(np.squeeze(output['output_0']))
    scores = np.squeeze(output['output_1'])
    classes = np.squeeze(output['output_2'])
    boxes = np.squeeze(output['output_3'])

    # * 對輸出的結果數據append到results中 => return results
    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results


def run_odt_and_draw_results(image_path, interpreter, threshold=0.5):
    """Run object detection on the input image and draw the detection results"""
    # Load the input shape required by the model
    # * shape代表張量的形狀 => get_input_details會返回一組array組成的dic => model張量的適合大小、狀態
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    # Load the input image and preprocess it
    # *　把照片resize成模型可以計算的大小 & 轉換成uint8單位
    preprocessed_image, original_image = preprocess_image(
        image_path,
        (input_height, input_width)
    )

    # Run object detection on the input image
    # * 偵測照片並回傳標註結果 => bounding_box class_id score
    results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

    # Plot the detection results on the input image
    # * 將照片轉為np.uint8方便opencv從照片上輸出方框
    original_image_np = original_image.numpy().astype(np.uint8)
    for obj in results:
        # Convert the object bounding box from relative coordinates to absolute
        # coordinates based on the original image resolution
        # ? x, y為[0, 1]，表示匡列的百分比位置，所以後續才要*original_image_np.shape => 整張照片的寬高
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * original_image_np.shape[1])
        xmax = int(xmax * original_image_np.shape[1])
        ymin = int(ymin * original_image_np.shape[0])
        ymax = int(ymax * original_image_np.shape[0])

        # Find the class index of the current object
        class_id = int(obj['class_id'])

        # Draw the bounding box and label on the image
        # ! 因為選了class_id，所以遍歷RGB (R, G, B)
        color = [int(c) for c in COLORS[class_id]]
        cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax),color, 2)  # ! (目標圖像, 左上角座標, 右下角座標, 顏色, 線條粗細)
        # Make adjustments to make the label visible for all objects
        y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        label = "{}: {:.0f}%".format(classes[class_id], obj['score'] * 100)  # ! .0f為不帶小數點的浮點數
        cv2.putText(original_image_np, label, (xmin, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # ! cv2.FONT_HERSHEY_SIMPLEX為字體種類, 文字大小, color, 線寬

    # Return the final image
    original_uint8 = original_image_np.astype(np.uint8)  # ! 已經是np.uint8了，可刪除（待確認）
    return original_uint8
