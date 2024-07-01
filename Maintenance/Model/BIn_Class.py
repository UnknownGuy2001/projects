import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM

np.random.seed(1234)
PYTHONHASHSEED = 0

MODEL_PATH = '../../Output/binary_model.h5'

train_df = pd.read_csv('../../Dataset/PM_train.txt', sep=" ", header=None).drop(columns=[26, 27])
train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + ['s' + str(i) for i in range(1, 22)]
train_df.sort_values(['id', 'cycle'], inplace=True)

test_df = pd.read_csv('../../Dataset/PM_test.txt', sep=" ", header=None).drop(columns=[26, 27])
test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + ['s' + str(i) for i in range(1, 22)]

truth_df = pd.read_csv('../../Dataset/PM_truth.txt', sep=" ", header=None).drop(columns=[1])

train_rul = train_df.groupby('id')['cycle'].max().reset_index()
train_rul.columns = ['id', 'max']
train_df = train_df.merge(train_rul, on='id', how='left')
train_df['RUL'] = train_df['max'] - train_df['cycle']
train_df.drop(columns=['max'], inplace=True)

W1, W0 = 30, 15
train_df['label1'] = np.where(train_df['RUL'] <= W1, 1, 0)
train_df['label2'] = train_df['label1']
train_df.loc[train_df['RUL'] <= W0, 'label2'] = 2

train_df['cycle_norm'] = train_df['cycle']
cols_normalize = train_df.columns.difference(['id', 'cycle', 'RUL', 'label1', 'label2'])
scaler = preprocessing.MinMaxScaler()
norm_train_df = pd.DataFrame(scaler.fit_transform(train_df[cols_normalize]), columns=cols_normalize, index=train_df.index)
train_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df).reindex(columns=train_df.columns)

test_df['cycle_norm'] = test_df['cycle']
norm_test_df = pd.DataFrame(scaler.transform(test_df[cols_normalize]), columns=cols_normalize, index=test_df.index)
test_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df).reindex(columns=test_df.columns).reset_index(drop=True)

test_rul = test_df.groupby('id')['cycle'].max().reset_index()
test_rul.columns = ['id', 'max']
truth_df.columns = ['more']
truth_df['id'] = truth_df.index + 1
truth_df['max'] = test_rul['max'] + truth_df['more']
truth_df.drop(columns=['more'], inplace=True)
test_df = test_df.merge(truth_df, on='id', how='left')
test_df['RUL'] = test_df['max'] - test_df['cycle']
test_df.drop(columns=['max'], inplace=True)

test_df['label1'] = np.where(test_df['RUL'] <= W1, 1, 0)
test_df['label2'] = test_df['label1']
test_df.loc[test_df['RUL'] <= W0, 'label2'] = 2

def gen_sequence(id_df, seq_length, seq_cols):
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]

sensor_cols = ['s' + str(i) for i in range(1, 22)]
sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm'] + sensor_cols

seq_gen = (list(gen_sequence(train_df[train_df['id']==id], 50, sequence_cols)) for id in train_df['id'].unique())
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)

def gen_labels(id_df, seq_length, label):
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_length:num_elements, :]

label_gen = [gen_labels(train_df[train_df['id']==id], 50, ['label1']) for id in train_df['id'].unique()]
label_array = np.concatenate(label_gen).astype(np.float32)

nb_features = seq_array.shape[2]
nb_out = label_array.shape[1]
model = Sequential([
    LSTM(input_shape=(50, nb_features), units=100, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=nb_out, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(seq_array, label_array, epochs=100, batch_size=200, validation_split=0.05, verbose=2,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='min'),
        keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, mode='min')
    ]
)

def plot_history(history, metric, title, ylabel, filename):
    fig = plt.figure(figsize=(10, 10))
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    fig.savefig(f"../../Output/{filename}.png")

plot_history(history, 'accuracy', 'Model Accuracy', 'accuracy', 'model_accuracy')
plot_history(history, 'loss', 'Model Loss', 'loss', 'model_loss')

model.evaluate(seq_array, label_array, verbose=1, batch_size=200)

y_pred = model.predict_classes(seq_array, verbose=1, batch_size=200)
y_true = label_array
print(confusion_matrix(y_true, y_pred))
print(precision_score(y_true, y_pred), recall_score(y_true, y_pred))

seq_array_test_last = [test_df[test_df['id']==id][sequence_cols].values[-50:] 
                       for id in test_df['id'].unique() if len(test_df[test_df['id']==id]) >= 50]
seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)

y_mask = [len(test_df[test_df['id']==id]) >= 50 for id in test_df['id'].unique()]
label_array_test_last = test_df.groupby('id')['label1'].nth(-1)[y_mask].values
label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0], 1).astype(np.float32)

if os.path.isfile(MODEL_PATH):
    best_model = load_model(MODEL_PATH)

best_model.evaluate(seq_array_test_last, label_array_test_last, verbose=2)

y_pred_test = best_model.predict_classes(seq_array_test_last)
y_true_test = label_array_test_last
print(confusion_matrix(y_true_test, y_pred_test))
precision_test = precision_score(y_true_test, y_pred_test)
recall_test = recall_score(y_true_test, y_pred_test)
f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)
print(precision_test, recall_test, f1_test)

fig_verify = plt.figure(figsize=(100, 50))
plt.plot(y_pred_test, color="blue")
plt.plot(y_true_test, color="green")
plt.title('Prediction vs Actual')
plt.ylabel('Value')
plt.xlabel('Row')
plt.legend(['Predicted', 'Actual'], loc='upper left')
plt.show()
fig_verify.savefig("../../Output/model_verify.png")
