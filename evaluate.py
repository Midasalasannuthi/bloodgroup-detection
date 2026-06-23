import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224,224)

model = load_model("model/blood_model.h5")

datagen = ImageDataGenerator(rescale=1./255)

test_data = datagen.flow_from_directory(
    "dataset",
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode="categorical"
)

predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)

print("Classification Report:")
print(classification_report(test_data.classes, y_pred))

cm = confusion_matrix(test_data.classes, y_pred)

plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.show()