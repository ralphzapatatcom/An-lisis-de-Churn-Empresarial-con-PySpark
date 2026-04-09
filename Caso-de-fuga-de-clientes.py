from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import GBTClassifier
from pyspark.sql import functions as F

# 1. Configuración del motor de Big Data (Spark)
spark = SparkSession.builder \
    .appName("Analisis_Churn_Empresarial") \
    .getOrCreate()

# 2. Simulación de Dataset Corporativo (Sustituir por carga de CSV/Base de Datos)
# Generamos 20 variables reales de una empresa de servicios
print("Cargando base de datos de clientes...")
data = spark.range(0, 50000).select(
    F.col("id"),
    (F.rand() * 24).alias("antiguedad_meses"),
    (F.rand() * 100).alias("facturacion_mensual"),
    (F.rand() * 10).cast("int").alias("tickets_soporte"),
    (F.rand() * 500).alias("minutos_voz"),
    (F.rand() * 50).alias("gb_datos_mes"),
    (F.rand() * 5).cast("int").alias("retrasos_pago"),
    (F.rand() * 2).cast("int").alias("plan_premium"),  # 0 o 1
    (F.rand() * 10).alias("score_satisfaccion"),
    (F.rand() * 60).alias("edad_cliente")
)

# Añadimos variables adicionales hasta completar 20 para el análisis
for i in range(11, 21):
    data = data.withColumn(f"metrica_uso_{i}", F.rand() * 100)

# Definimos el Objetivo (Label): 1 si abandona, 0 si se queda
# Un cliente con muchos tickets de soporte y baja satisfacción tiende al Churn
data = data.withColumn("churn", 
    F.when((F.col("tickets_soporte") > 7) | (F.col("score_satisfaccion") < 3), 1).otherwise(0)
)

# 3. Preparación de las 20 variables (Features)
# Excluimos 'id' y 'churn' para el entrenamiento
variables_estudio = [c for c in data.columns if c not in ['id', 'churn']]

assembler = VectorAssembler(inputCols=variables_estudio, outputCol="features")
dataset_preparado = assembler.transform(data)

# 4. Modelado Predictivo (Gradient Boosted Trees - Muy usado en finanzas/telco)
train, test = dataset_preparado.randomSplit([0.7, 0.3], seed=42)
gbt = GBTClassifier(labelCol="churn", featuresCol="features", maxIter=10)
modelo = gbt.fit(train)

# 5. Ejecución del estudio y Predicciones
predicciones = modelo.transform(test)

# 6. IMPRESIÓN DE RESULTADOS (Las 20 variables + Predicción de Riesgo)
print("\n" + "="*80)
print("INFORME DE RIESGO DE FUGA DE CLIENTES (PRIMERAS 20 VARIABLES)")
print("="*80)

# Seleccionamos las columnas para mostrar el reporte final
columnas_reporte = variables_estudio[:20] + ["churn", "prediction"]
predicciones.select(*columnas_reporte).show(10, truncate=False)

# Resumen estadístico del problema
print("\nResumen del estudio:")
predicciones.groupBy("prediction").count().show()

spark.stop()