# kaggle-cats-dogs

Code for The [Kaggle Cats and Dogs](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition) challenge using the 
[Cortex](https://github.com/thinktopic/cortex) project. In particular, heavily using the example from the [suite-classification](https://github.com/thinktopic/cortex/tree/master/examples/suite-classification)

A blog post covering it [http://gigasquidsoftware.com/blog/2016/12/27/deep-learning-in-clojure-with-cortex/](http://gigasquidsoftware.com/blog/2016/12/27/deep-learning-in-clojure-with-cortex/)

## Usage

Get the datasets from the above Kaggle Site. Download the `train.zip` and `test.zip` to the resources directory and unzip them.

This example has been recently updated to cortex's master version, so to run it you will need to clone the cortex project and run the local-install script .

You will also need to compile the front end assets to see the web interface confusion matrix:

```
lein garden once
lein cljsbuild once
```

Then run 

`lein run to train`

You will see output about building the image sets, then you will see figwheel starting and compiling some ClojureScript files. This is for the confusion matrix visualization. After you see `Successfully compiled`, go ahead and open a web page at [http://localhost:8091/](http://localhost:8091/).  This will load the confusion matrix visualization.

## License

Copyright © 2016 Carin Meier

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
