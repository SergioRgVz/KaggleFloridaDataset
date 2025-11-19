import pandas as pd
from autogluon.tabular import TabularPredictor

# 1. Cargar los datos
print("Cargando datos...")
train_data = pd.read_csv('data/cleaned_train_split_aux.csv')
test_data = pd.read_csv('data/cleaned_test_split_aux.csv')

# 2. Seleccionar columnas
# El usuario pidio usar todas las variables
# Eliminamos la columna 'id' ya que no aporta informacion predictiva
# La columna target es 'precio'
target = 'precio'

print("Preparando datos con todas las variables...")
# Hacemos una copia para no modificar el original
# Quitamos 'id' de entrenamiento
# train_subset = train_data.drop(columns=['id']).copy()

# Limpiar datos: eliminar filas donde el target es NaN
# train_subset = train_subset.dropna(subset=[target])
# print(f"Datos de entrenamiento despues de limpiar nulos en target: {len(train_subset)}")

# test_subset = test_data.drop(columns=['id']).copy()

# 3. Entrenar con AutoGluon
# Entrenar solo 1 modelo (LightGBM) como se solicito
print("Entrenando un solo modelo (GBM) con AutoGluon...")
predictor = TabularPredictor(label=target).fit(
    train_data=train_data,
    hyperparameters={'GBM': {}}, # Forzamos a usar solo LightGBM
)

# 4. Predecir en test
print("Haciendo predicciones en test...")
predictions = predictor.predict(test_data)

# 5. Crear archivo de submission
print("Creando archivo sample.csv...")
submission = pd.DataFrame({
    'id': test_data['id'],
    'precio': predictions
})

submission.to_csv('sample.csv', index=False)
print("¡Listo! Archivo sample.csv creado.")