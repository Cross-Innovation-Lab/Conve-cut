# -*- coding: utf-8 -*-            
# @Author : BingYu Nan
# @Location : Wuxi
# @Time : 2024/9/5 10:46
import datetime

import h5py
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score
from sklearn.utils import shuffle
from tensorflow import keras
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# Model Building
NUM_CLASSES = 7
BATCH_SIZE = 16

TRAIN_EPOCH = 100
TRAIN_LR = 1e-3
TRAIN_ES_PATIENCE = 5
TRAIN_LR_PATIENCE = 3
TRAIN_MIN_LR = 1e-6
TRAIN_DROPOUT = 0.1

FT_EPOCH = 30
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


input_layer = tf.keras.Input(shape=(224, 224, 3), name='universal_input')
sample_resizing = tf.keras.layers.experimental.preprocessing.Resizing(224, 224, name="resize")
data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip(mode='horizontal'), tf.keras.layers.RandomContrast(factor=0.3)], name="augmentation")
preprocess_input = tf.keras.applications.convnext.preprocess_input
backbone = tf.keras.applications.convnext.ConvNeXtBase(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
backbone.trainable = False
base_model = tf.keras.Model(backbone.input, backbone.layers[-57].output, name='base_model')#-126,-49,-28,-33,-65
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
#x = layernorm_layer(x)
x = tf.keras.layers.SpatialDropout2D(0.1)(x)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = pre_classification(x)
x = self_attention([x, x])
x = tf.keras.layers.Dropout(0.1)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs, name='train-head')


# # 替换下面的路径为你的模型文件路径
model_path = r'.\2_raf_9782.h5'
#
# # 加载模型
model.load_weights(model_path)

# 检查模型结构
model.summary()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3, global_clipnorm=3.0), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
test_loss, test_acc = model.evaluate(X_test, y_test)

#在测试集上进行预测
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # 获取最大概率的索引作为预测类别

f1score = f1_score(y_test, y_pred_classes, average='macro')
print('F1', f1score)

# 生成混淆矩阵
cm = confusion_matrix(y_test, y_pred_classes)
row_sums = np.sum(cm, axis=1)
cm = (cm / row_sums[:, np.newaxis])*100

label_name = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

# 可视化混淆矩阵
plt.figure(figsize=(10, 8))
plt.imshow(cm, cmap='Reds')
plt.xticks(range(label_name.__len__()), label_name,)
plt.yticks(range(label_name.__len__()), label_name,)
plt.xlabel('Predicted Labels', horizontalalignment='center')
plt.ylabel('True Labels', horizontalalignment='center')

plt.title('Accuracy score:97.33%')
for i in range(label_name.__len__()):
    for j in range(label_name.__len__()):
        color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
        value = float(format('%.2f' % cm[j, i]))
        plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

plt.savefig('raf-acc.png', dpi=300)


label_colors = {
    0: 'purple',      # 标签0的颜色
    1: 'green',     # 标签1的颜色
    2: 'yellow',    # 标签2的颜色
    3: 'pink',
    4: 'brown',
    5: 'blue',
    6: 'cyan'
}

#tesn可视化
def get_model_features(model, data):
    features = model.predict(data)
    return features


# 获取训练集和验证集的特征
val_features = get_model_features(model, X_valid)

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, random_state=42)
val_features_2d = tsne.fit_transform(val_features)

# 绘制训练集的t-SNE图
plt.figure(figsize=(6, 6))
for label in np.unique(y_valid):
    plt.scatter(val_features_2d[y_valid == label, 0], val_features_2d[y_valid == label, 1],
                c=label_colors[label], label=f'Label {label}', s=2)

plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.savefig('tsne_raf_best.png')

