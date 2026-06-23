import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

dataset="dataset"

train_datagen=ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2
)

train=train_datagen.flow_from_directory(
    dataset,
    target_size=(224,224),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

val=train_datagen.flow_from_directory(
    dataset,
    target_size=(224,224),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

base=MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

base.trainable=True

for layer in base.layers[:100]:
    layer.trainable=False

x=base.output
x=GlobalAveragePooling2D()(x)

x=Dense(
    256,
    activation='relu'
)(x)

x=BatchNormalization()(x)

x=Dropout(0.5)(x)

output=Dense(
    8,
    activation='softmax'
)(x)

model=Model(
    base.input,
    output
)

model.compile(
optimizer=Adam(1e-4),
loss='categorical_crossentropy',
metrics=['accuracy']
)

early=EarlyStopping(
monitor='val_accuracy',
patience=5,
restore_best_weights=True
)

reduce=ReduceLROnPlateau(
monitor='val_loss',
factor=0.2,
patience=3
)

history=model.fit(
train,
validation_data=val,
epochs=35,
callbacks=[early,reduce]
)

model.save(
"model/blood_model.h5"
)

print("Training complete")