## Imdb reviews text classification

This notebook classifies movie reviews as positive or negative using the text of the review. This is an example of binary—or two-class—classification

Feature:
```
<tf.Tensor: shape=(10,), dtype=string, numpy=
array([b"This was an absolutely terrible movie. Don't be lured in by Christopher Walken..."], dtype=object)>
```
Long text is clipped for brevity

Label: 
```
<tf.Tensor: shape=(10,), dtype=int64, numpy=array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])>
```

The neural network is created by stacking layers— this requires three main architectural decisions:

1. How to represent the text?
2. How many layers to use in the model?
3. How many hidden units to use for each layer?

One way to represent the text is to convert sentences into embeddings vectors. We can use a pre-trained text embedding as the first layer, which will have three advantages:

1. we don't have to worry about text preprocessing,
2. we can benefit from transfer learning,
3. the embedding has a fixed size, so it's simpler to process.

For this example we will use a pre-trained text embedding model from TensorFlow Hub called 
> google/tf2-preview/gnews-swivel-20dim/1

Example:

```
import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1")
embeddings = embed(["cat is on the mat", "dog is in the fog"])

<tf.Tensor: shape=(2, 20), dtype=float32, numpy=
array([[ 0.8666395 ,  0.35917717,  0.00579667,  0.681002  , -0.54226625,
         0.22343187, -0.38796625,  0.62195706,  0.22117122, -0.48538068,
        -1.2674141 ,  0.886369  , -0.3284907 , -0.13924702, -0.53327686,
         0.5739708 , -0.05905761,  0.13629246, -1.1718255 , -0.31494334],
       [ 0.960218  ,  0.62520486,  0.06261905,  0.37425604,  0.24782333,
        -0.39351934, -0.7418429 ,  0.56599647, -0.26197797, -0.69016844,
        -0.76565284,  0.71412426, -0.4537978 , -0.50701594, -0.8499377 ,
         0.8917156 , -0.30278975,  0.2149126 , -1.1098894 , -0.46719775]],
      dtype=float32)>
```

