# -*- coding: utf-8 -*-            
# @Author : BingYu Nan
# @Location : Wuxi
# @Time : 2024/9/13 12:08
import datetime

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight

# Model Building
NUM_CLASSES = 7
BATCH_SIZE = 16

TRAIN_EPOCH = 100
TRAIN_LR = 1e-3
TRAIN_ES_PATIENCE = 5
TRAIN_LR_PATIENCE = 3
TRAIN_MIN_LR = 1e-6
TRAIN_DROPOUT = 0.1

FT_EPOCH = 60
FT_LR = 1e-5
FT_LR_DECAY_STEP = 80.0
FT_LR_DECAY_RATE = 1
FT_ES_PATIENCE = 20
FT_DROPOUT = 0.2

ES_LR_MIN_DELTA = 0.003
IMG_SHAPE = (224, 224, 3)
TRAIN_DROPOUT = 0.1


with h5py.File(r'.\rafdb.h5', 'r') as hdf5_file:
    # 加载数据和标签
    X_train = np.array(hdf5_file.get('X_train'))
    y_train = np.array(hdf5_file.get('y_train'))
    X_test = np.array(hdf5_file.get('X_test'))
    y_test = np.array(hdf5_file.get('y_test'))
    X_valid = X_test
    y_valid = y_test
# Load your data here, PAtt-Lite was trained with h5py for shorter loading time
X_train, y_train = shuffle(X_train, y_train)

print("Shape of train_sample: {}".format(X_train.shape))
print("Shape of train_label: {}".format(y_train.shape))
print("Shape of valid_sample: {}".format(X_valid.shape))
print("Shape of valid_label: {}".format(y_valid.shape))
print("Shape of test_sample: {}".format(X_test.shape))
print("Shape of test_label: {}".format(y_test.shape))

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))


input_layer = tf.keras.Input(shape=IMG_SHAPE, name='universal_input')
sample_resizing = tf.keras.layers.experimental.preprocessing.Resizing(224, 224, name="resize")
data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip(mode='horizontal'),
                                        tf.keras.layers.RandomContrast(factor=0.3)], name="augmentation")
preprocess_input = tf.keras.applications.convnext.preprocess_input

backbone = tf.keras.applications.convnext.ConvNeXtBase(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
backbone.trainable = False
base_model = tf.keras.Model(backbone.input, backbone.layers[-57].output, name='base_model')#-126,-49,-28,-33,-65
print('model:{}'.format(base_model.summary()))

self_attention = tf.keras.layers.Attention(use_scale=True, name='attention')
patch_extraction = tf.keras.Sequential([
    tf.keras.layers.SeparableConv2D(256, kernel_size=4, strides=4, padding='same', activation='relu'),
    tf.keras.layers.SeparableConv2D(256, kernel_size=2, strides=2, padding='valid', activation='relu'),
], name='patch_extraction')

layernorm_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layernorm")
global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name='gap')
pre_classification = tf.keras.Sequential([tf.keras.layers.Dense(32, activation='relu'),
                                          tf.keras.layers.BatchNormalization()], name='pre_classification')
prediction_layer = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name='classification_head')

inputs = input_layer
x = sample_resizing(inputs)
x = data_augmentation(x)
x = preprocess_input(x)
x = base_model(x, training=False)
x = patch_extraction(x)
x = layernorm_layer(x)
x = tf.keras.layers.SpatialDropout2D(0.1)(x)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = pre_classification(x)
x = self_attention([x, x])
x = tf.keras.layers.Dropout(0.1)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs, name='train-head')
tf.debugging.set_log_device_placement(True)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=TRAIN_LR, global_clipnorm=3.0), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Training Procedure
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=TRAIN_ES_PATIENCE, min_delta=ES_LR_MIN_DELTA, restore_best_weights=True)
learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=TRAIN_LR_PATIENCE, verbose=0, min_delta=ES_LR_MIN_DELTA, min_lr=TRAIN_MIN_LR)
history = model.fit(X_train, y_train, epochs=TRAIN_EPOCH, batch_size=BATCH_SIZE, validation_data=(X_valid, y_valid), verbose=1,
                    class_weight=class_weights, callbacks=[early_stopping_callback, learning_rate_callback])
test_loss, test_acc = model.evaluate(X_test, y_test)

# Model Finetuning
print("\nFinetuning ...")
base_model.trainable = True
fine_tune_from = -1
for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False
for layer in base_model.layers[fine_tune_from:]:
    if isinstance(layer, tf.keras.layers.LayerNormalization):
        layer.trainable = False

inputs = input_layer
x = sample_resizing(inputs)
x = data_augmentation(x)
x = preprocess_input(x)
x = base_model(x, training=False)
x = patch_extraction(x)
#x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layernorm")(x)
x = tf.keras.layers.SpatialDropout2D(FT_DROPOUT)(x)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(FT_DROPOUT)(x)
x = pre_classification(x)
x = self_attention([x, x])
x = tf.keras.layers.Dropout(FT_DROPOUT)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs, name='finetune-backbone')
model.compile(optimizer=keras.optimizers.Adam(learning_rate=FT_LR, global_clipnorm=3.0), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training Procedure
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=ES_LR_MIN_DELTA, patience=FT_ES_PATIENCE, restore_best_weights=True)
scheduler = keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=FT_LR, decay_steps=FT_LR_DECAY_STEP, decay_rate=FT_LR_DECAY_RATE)
scheduler_callback = tf.keras.callbacks.LearningRateScheduler(schedule=scheduler)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    'best_raf_weights.h5',            # 保存权重的文件路径
    monitor='val_accuracy',       # 监控的指标是验证集的准确率
    verbose=1,                   # 日志显示模式，1表示输出详细日志
    save_best_only=True,         # 只保存在验证集上性能最好的模型
    mode='max',                  # 我们希望“监控指标”应该被最大化
    save_weights_only=True       # 只保存模型的权重
)

history_finetune = model.fit(X_train, y_train, epochs=FT_EPOCH, batch_size=BATCH_SIZE, validation_data=(X_valid, y_valid), verbose=1,
                             initial_epoch=history.epoch[-TRAIN_ES_PATIENCE], callbacks=[early_stopping_callback, scheduler_callback, tensorboard_callback, checkpoint_callback])
test_loss, test_acc = model.evaluate(X_test, y_test)
model.save('raf_model.h5')
