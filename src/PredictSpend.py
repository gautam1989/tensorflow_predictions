import pandas as pd
import tensorflow as tf


CATEGORICAL_COLUMNS = [ "location", "weather"]
CONTINUOUS_COLUMNS = ["age"]

SURVIVED_COLUMN = "drink"

def build_estimator(model_dir):
  """Build an estimator."""
  # Categorical columns
  weather = tf.contrib.layers.sparse_column_with_keys(column_name="weather",
                                                     keys=["rainy", "sunny"])
  

  location = tf.contrib.layers.sparse_column_with_hash_bucket(
      "location", hash_bucket_size=1000)
 
  


  # Continuous columns
  age = tf.contrib.layers.real_valued_column("age")
  

  # Transformations.
  age_buckets = tf.contrib.layers.bucketized_column(age,
                                                    boundaries=[
                                                        5, 18, 25, 30, 35, 40,
                                                        45, 50, 55, 65
                                                    ])
   # Wide columns and deep columns.
  wide_columns = [location,weather,age_buckets,
                  tf.contrib.layers.crossed_column(
                      [weather, location],
                      hash_bucket_size=int(1e6)),
                  tf.contrib.layers.crossed_column([weather,age_buckets ],
                                                   hash_bucket_size=int(1e4))]
  deep_columns = [
      tf.contrib.layers.embedding_column(weather, dimension=8),
      tf.contrib.layers.embedding_column(location, dimension=8),
      age
  ]



  return tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50])

def input_fn(df, train=False):
  print("in input func")
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  print("in input func2")
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
    indices=[[i, 0] for i in range(df[k].size)],
    values=df[k].values,
    dense_shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  print("in input func3")
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  if train:
    label = tf.constant(df[SURVIVED_COLUMN].values)
      # Returns the feature columns and the label.
    return feature_cols, label
  else:
    return feature_cols


def train_and_eval():
  """Train and evaluate the model."""
  df_train = pd.read_csv(
      tf.gfile.Open("D:/neural_network_wrkspc/TensorFlow/res/usertraindata.csv"),
      skipinitialspace=True)
  df_test = pd.read_csv(
      tf.gfile.Open("D:/neural_network_wrkspc/TensorFlow/res/userevaldata.csv"),
      skipinitialspace=True)

  model_dir = "D:/neural_network_wrkspc/TensorFlow/res/models"
  print("model directory = %s" % model_dir)

  m = build_estimator(model_dir)
  print("estimator is built")
  #m.fit(input_fn=lambda: input_fn(df_train, True), steps=200)
  print("predictions ========= >")
  print (m.predict(input_fn=lambda: input_fn(df_test), as_iterable=False))
  
  
def main(_):
  train_and_eval()


if __name__ == "__main__":
  tf.app.run()