import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#Keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.models import model_from_json

from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

import os.path

def load_db():
    return pd.read_csv('../estudo/kaggle/Titanic/train.csv')

def deleta_colunas(data):
    del data['Name']
    del data['Ticket']
    del data['Cabin']
    del data['Embarked']
    del data['Fare']
    del data['PassengerId']

#transforma o campo Sex de categorico em numerico
def sexo(x):
    if x == 'male':
        return 0
    else:
        return 1

def matriz_confusao(y_true, y_pred):
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1])

    classes = ["0", "1"]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de confusão")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.text(0, 0, cm[0, 0])
    plt.text(0, 1, cm[0, 1])
    plt.text(1, 0, cm[1, 0])
    plt.text(1, 1, cm[1, 1])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    return cm


def ajusta_valores(data):
    col = data['Age'].copy()
    media_idade = col[col.isnull() == False].mean()
    col[col.isnull() == True] = media_idade
    data['Age'] = col

    # Normaliza coluna
    data['Age'] = preprocessing.scale(data['Age'])

    # transforma coluna categorica em numerica
    data['Sex'] = data['Sex'].apply(sexo)

def distribuir_dataset(data, validation=False):
    treinamento = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
    label = ['Survived']
    if validation:
        x_train, x_reman, y_train, y_reman = train_test_split(data[treinamento], data[label], test_size=0.5, random_state=42)
        x_test, x_validation, y_test, y_validation = train_test_split(x_reman, y_reman, test_size=0.5, random_state=42)
        return x_train, x_test, x_validation, y_train, y_test, y_validation
    else:
        x_train, x_test, y_train, y_test = train_test_split(data[treinamento], data[label], test_size=0.2, random_state=42)
        return x_train, x_test, y_train,y_test

def start(validation=False):
    data = load_db()
    deleta_colunas(data)
    ajusta_valores(data)
    return distribuir_dataset(data, validation=validation)

def save_model(model, nome_arq, count=1):
    # serialize model to JSON
    if not os.path.exists(nome_arq+".json"):
        model_json = model.to_json()
        with open(nome_arq+".json", "w") as json_file:
            json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(nome_arq + count + ".h5")
    print("Saved model to disk")

def load_model(nome_arq, num=1):
    # load json and create model
    json_file = open(nome_arq + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(nome_arq + str(num) + ".h5")
    print("Loaded model from disk")
    return loaded_model

def monta_modelo1():
    model = Sequential()
    model.add(Dense(5, activation='relu', input_dim=5))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))

    return model

def monta_modelo2():
    model = Sequential()
    model.add(Dense(5, activation='relu', input_dim=5))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))

    return model

def monta_modelo3():
    model = Sequential()
    model.add(Dense(5, activation='relu', input_dim=5))

    model.add(Dense(18, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(5, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model

def monta_modelo4():
    model = Sequential()
    model.add(Dense(5, activation='relu', input_dim=5))

    model.add(Dense(18, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(5, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))

    return model

#Prepara set de treino e test
x_train, x_test, x_validation, y_train, y_test, y_validation = start(validation=True)
y_train = keras.utils.to_categorical(y_train, num_classes=2)
y_test = keras.utils.to_categorical(y_test, num_classes=2)
y_validation = keras.utils.to_categorical(y_test, num_classes=2)

#Modelo
model = monta_modelo4()

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
#model = load_model('titanic')
adam = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

epochs = 200
save_epochs = 10000
cont_epoch = 0
'''
#for cont_epoch in range(1, int(epochs/save_epochs)+1):
for cont_epoch in range(1, 2):
    model.fit(x_train, y_train,
              validation_data= (x_test, y_test),
              epochs=epochs, #int(save_epochs),
              batch_size=20)
    save_model(model, "titanic", str(cont_epoch))
'''
model.fit(x_train, y_train,
              validation_data= (x_test, y_test),
              epochs=epochs, #int(save_epochs),
              batch_size=50)
scores = model.evaluate(x_test, y_test, batch_size=150)
print(scores)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print("\n\n\n")

predictions = model.predict(x_validation)
y_pred = np.array([round(x[0]) for x in predictions])
y_real = np.array(y_validation).flatten()

cm = matriz_confusao(y_real, y_pred)

acerto_mortos = (cm[0][0]/(cm[0][0]+cm[0][1]) * 100)
acerto_vivos = (cm[1][1]/(cm[1][0]+cm[1][1]) * 100)
acerto_total = ((cm[0][0] + cm[1][1])/(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])) * 100

print(" Acerto vivos: {:.2f}%".format(acerto_vivos))
print("Acerto mortos: {:.2f}%".format(acerto_mortos))
print(" Acerto total: {:.2f}%".format(acerto_total))