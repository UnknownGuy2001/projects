import os
import keras
import keras.backend as K
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Activation
from sklearn import preprocessing

np.random.seed(1234)

MODEL_PATH = '../../Output/regression_model.h5'

train_df = pd.read_csv('../../Dataset/PM_train.txt', sep=" ", header=None).drop(columns=[26, 27])
train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + ['s' + str(i) for i in range(1, 22)]
train_df = train_df.sort_values(['id', 'cycle'])

test_df = pd.read_csv('../../Dataset/PM_test.txt', sep=" ", header=None).drop(columns=[26, 27])
test_df.columns = train_df.columns

truth_df = pd.read_csv('../../Dataset/PM_truth.txt', sep=" ", header=None).drop(columns=[1])

rul = train_df.groupby('id')['cycle'].max().reset_index()
rul.columns = ['id', 'max']
train_df = train_df.merge(rul, on='id')
train_df['RUL'] = train_df['max'] - train_df['cycle']
train_df.drop(columns=['max'], inplace=True)

train_df['cycle_norm'] = train_df['cycle']
cols_normalize = train_df.columns.difference(['id', 'cycle', 'RUL', 'label1', 'label2'])
scaler = preprocessing.MinMaxScaler()
norm_train_df = pd.DataFrame(scaler.fit_transform(train_df[cols_normalize]), columns=cols_normalize, index=train_df.index)
train_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df).reindex(columns=train_df.columns)

test_df['cycle_norm'] = test_df['cycle']
norm_test_df = pd.DataFrame(scaler.transform(test_df[cols_normalize]), columns=cols_normalize, index=test_df.index)
test_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df).reindex(columns=test_df.columns).reset_index(drop=True)

rul = test_df.groupby('id')['cycle'].max().reset_index()
rul.columns = ['id', 'max']
truth_df.columns = ['more']
truth_df['id'] = truth_df.index + 1
truth_df['max'] = rul['max'] + truth_df['more']
truth_df.drop(columns=['more'], inplace=True)
test_df = test_df.merge(truth_df, on='id')
test_df['RUL'] = test_df['max'] - test_df['cycle']
test_df.drop(columns=['max'], inplace=True)

def gen_sequence(id_df, seq_length, seq_cols):
    data_matrix = id_df[seq_cols].values
    for start, stop in zip(range(0, len(data_matrix) - seq_length), range(seq_length, len(data_matrix))):
        yield data_matrix[start:stop, :]

sensor_cols = ['s' + str(i) for i in range(1, 22)]
sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm'] + sensor_cols

seq_gen = (list(gen_sequence(train_df[train_df['id'] == id], 50, sequence_cols)) for id in train_df['id'].unique())
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)

def gen_labels(id_df, seq_length, label):
    data_matrix = id_df[label].values
    return data_matrix[seq_length:len(data_matrix), :]

label_gen = [gen_labels(train_df[train_df['id'] == id], 50, ['RUL']) for id in train_df['id'].unique()]
label_array = np.concatenate(label_gen).astype(np.float32)

def r2_keras(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())

nb_features = seq_array.shape[2]
nb_out = label_array.shape[1]

model = Sequential([
    LSTM(input_shape=(50, nb_features), units=100, return_sequences=True),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(nb_out),
    Activation("linear")
])
model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae', r2_keras])
model.fit(seq_array, label_array, epochs=100, batch_size=200, validation_split=0.05, verbose=2, callbacks=[
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min'),
    keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, mode='min')
])

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

plot_history(history, 'r2_keras', 'Model R^2', 'R^2', 'model_r2')
plot_history(history, 'mean_absolute_error', 'Model MAE', 'MAE', 'model_mae')
plot_history(history, 'loss', 'Model Loss', 'loss', 'model_regression_loss')

model.evaluate(seq_array, label_array, verbose=1, batch_size=200)

y_pred = model.predict(seq_array, verbose=1, batch_size=200)
pd.DataFrame(y_pred).to_csv('../../Output/submit_train.csv', index=None)

seq_array_test_last = [test_df[test_df['id'] == id][sequence_cols].values[-50:] for id in test_df['id'].unique() if
                       len(test_df[test_df['id'] == id]) >= 50]
seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)

y_mask = [len(test_df[test_df['id'] == id]) >= 50 for id in test_df['id'].unique()]
label_array_test_last = test_df.groupby('id')['RUL'].nth(-1)[y_mask].values
label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0], 1).astype(np.float32)

if os.path.isfile(MODEL_PATH):
    estimator = load_model(MODEL_PATH, custom_objects={'r2_keras': r2_keras})
    scores_test = estimator.evaluate(seq_array_test_last, label_array_test_last, verbose=2)
    print(f'\nMAE: {scores_test[1]}\nR^2: {scores_test[2]}')
    y_pred_test = estimator.predict(seq_array_test_last)
    pd.DataFrame(y_pred_test).to_csv('../../Output/submit_test.csv', index=None)
    fig_verify = plt.figure(figsize=(100, 50))
    plt.plot(y_pred_test, color="blue")
    plt.plot(label_array_test_last, color="green")
    plt.title('Prediction vs Actual')
    plt.ylabel('Value')
    plt.xlabel('Row')
    plt.legend(['Predicted', 'Actual'], loc='upper left')
    plt.show()
    fig_verify.savefig("../../Output/model_regression_verify.png")
