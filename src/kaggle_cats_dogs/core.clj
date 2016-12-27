(ns kaggle-cats-dogs.core
  (:require [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [clojure.string :as string]
            [cortex.dataset :as ds]
            [cortex.nn.description :as desc]
            [cortex.suite.classification :as classification]
            [cortex.suite.inference :as infer]
            [cortex.suite.io :as suite-io]
            [cortex.suite.train :as suite-train]
            [mikera.image.core :as imagez]
            [mikera.vectorz.matrix-api] ;; loading this namespace enables vectorz implementation for core.matrix
            [think.compute.nn.train :as train]
            [think.gate.core :as gate]
            [think.image.data-augmentation :as image-aug]
            [think.image.image :as image]
            [think.image.image-util :as image-util]
            [think.image.patch :as patch]
            [think.compute.optimise :as opt]))

;;We have to setup the web server slightly different when running
;;from the repl; we enable live updates using figwheel and such.  When
;;running from an uberjar we just launch the server and expect the
;;particular resources to be available.  We ensure this with a makefile.
(def ^:dynamic *running-from-repl* true)

(def dataset-image-size 52)
(def dataset-num-classes 2)
(def dataset-num-channels 3)
(def dataset-datatype :float)
(def max-image-rotation-degrees 25)
(def original-data-dir "resources/train")
(def training-dir "data-cats-dogs/training")
(def testing-dir "data-cats-dogs/testing")

(defn produce-indexed-data-label-seq
  [files]
  (->> (map (fn [file] [file (-> (.getName file) (string/split #"\.") first)]) files)
       (map-indexed vector)))

(defn resize-and-write-data
  [output-dir [idx [file label]]]
  (let [img-path (str output-dir "/" label "/" idx ".png" )]
    (when-not (.exists (io/file img-path))
      (io/make-parents img-path)
      (-> (imagez/load-image file)
          (image/resize dataset-image-size dataset-image-size)
          (imagez/save img-path)))
    nil))

(defn- gather-files [path]
  (->> (io/file path)
       (file-seq)
       (filter #(.isFile %))))


(defn build-image-data
  []
  (let [files (gather-files original-data-dir)
        pfiles (partition (int (/ (count files) 2)) (shuffle files))
        training-observation-label-seq (produce-indexed-data-label-seq
                                        (first pfiles))
        testing-observation-label-seq (produce-indexed-data-label-seq
                                       (last pfiles))
        train-fn (partial resize-and-write-data training-dir)
        test-fn (partial resize-and-write-data  testing-dir)]
    (dorun (pmap train-fn training-observation-label-seq))
    (dorun (pmap test-fn training-observation-label-seq))))

(defn create-basic-network-description
  []
  [(desc/input dataset-image-size dataset-image-size dataset-num-channels)
   (desc/convolutional 5 0 1 20)
   (desc/max-pooling 2 0 2)
   (desc/relu)
   (desc/convolutional 5 0 1 50)
   (desc/max-pooling 2 0 2)
   (desc/relu)
   (desc/convolutional 1 0 1 50)
   (desc/relu)
   (desc/linear->relu 1000)
   (desc/dropout 0.5)
   (desc/linear->softmax dataset-num-classes)])

(def max-image-rotation-degrees 25)

(defn img-aug-pipeline
  [img]
  (-> img
      (image-aug/rotate (- (rand-int (* 2 max-image-rotation-degrees))
                           max-image-rotation-degrees)
                        false)
      (image-aug/inject-noise (* 0.25 (rand)))))

(defn png->observation
  "Create an observation from input.  "
  [datatype augment? img]
  ;;image->patch always returns [r-data g-data g-data]
  ;;since we know these are grayscale *and* we setup the
  ;;network for 1 channel we just take r-data
  (patch/image->patch (if augment?
                        (img-aug-pipeline img)
                        img)
                      (image-util/image->rect img) datatype))


(defn observation->image
  [observation]
  (patch/patch->image observation dataset-image-size))


;;Bumping this up and producing several images per source image means that you may need
;;to shuffle the training epoch data to keep your batches from being unbalanced...this has
;;somewhat severe performance impacts.
(def ^:dynamic *num-augmented-images-per-file* 1)

(defn observation-label-pairs
  "Create a possibly infinite sequence of [observation label].
  Asking for an infinite sequence implies some level of data augmentation
  to avoid overfitting the network to the training data."
  [augment? datatype [file label]]
  (let [img (imagez/load-image file)
        png->obs #(png->observation datatype augment? img)
        ;;When augmenting we can return any number of items from one image.
        ;;You want to be sure that at your epoch size you get a very random, fairly
        ;;balanced set of observations->labels.  Furthermore you want to be sure
        ;;that at the batch size you have rough balance when possible.
        ;;The infinite-dataset implementation will shuffle each epoch of data when
        ;;training so it isn't necessary to randomize these patches at this level.
        repeat-count (if augment?
                       *num-augmented-images-per-file*
                       1)]
    ;;Laziness is not your friend here.  The classification system is setup
    ;;to call this on another CPU thread while training *so* if you are lazy here
    ;;then this sequence will get realized on the main training thread thus blocking
    ;;the training process unnecessarily.
    (mapv vector
          (repeatedly repeat-count png->obs)
          (repeat label))))

(defn create-dataset
  []
  (println "checking that we have produced all images")
  (build-image-data)
  (println "building dataset")
  (classification/create-classification-dataset-from-labeled-data-subdirs
   training-dir testing-dir
   (ds/create-image-shape dataset-num-channels dataset-image-size dataset-image-size)
   (partial observation-label-pairs true dataset-datatype)
   (partial observation-label-pairs false dataset-datatype)
   :epoch-element-count 6000
   :shuffle-training-epochs? (> *num-augmented-images-per-file* 2)))

(defn load-trained-network-desc
  []
  (:network-description (suite-io/read-nippy-file "trained-network.nippy")))

(defn display-dataset-and-model
  ([dataset initial-description]
   (let [data-display-atom (atom {})
         confusion-matrix-atom (atom {})]
     (classification/reset-dataset-display data-display-atom dataset observation->image)
     (when-let [loaded-data (suite-train/load-network "trained-network.nippy"
                                                      initial-description)]
       (classification/reset-confusion-matrix confusion-matrix-atom observation->image
                                              (suite-train/evaluate-network
                                               dataset
                                               (:network-description loaded-data)
                                               :batch-type :cross-validation)))
     (let [open-message
           (gate/open (atom
                       (classification/create-routing-map confusion-matrix-atom
                                                          data-display-atom))
                      :clj-css-path "src/css"
                      :live-updates? *running-from-repl*
                      :port 8091)]
       (println open-message))
     confusion-matrix-atom))
  ([]
   (display-dataset-and-model (create-dataset) (create-basic-network-description))
   nil))

(defn train-forever
  []
  (let [dataset (create-dataset)
        initial-description (create-basic-network-description)
        confusion-matrix-atom (display-dataset-and-model dataset initial-description)]
    (classification/train-forever dataset observation->image
                                  initial-description
                                  :confusion-matrix-atom confusion-matrix-atom)))

(defn train-forever-uberjar
  []
  (with-bindings {#'*running-from-repl* false}
    (train-forever)))

(defn label-one
  "Take an arbitrary image and label it."
  []
  (let [file-label-pairs (shuffle (classification/directory->file-label-seq testing-dir
                                                                            false))
        [test-file test-label] (first file-label-pairs)
        test-img (imagez/load-image test-file)
        observation (png->observation dataset-datatype false test-img)]
    (imagez/show test-img)
    (infer/classify-one-observation (:network-description
                                     (suite-io/read-nippy-file "trained-network.nippy"))
                                    observation
                                    (ds/create-image-shape dataset-num-channels
                                                           dataset-image-size
                                                           dataset-image-size)
                                    dataset-datatype
                                    (classification/get-class-names-from-directory testing-dir))))

(defn kaggle-png-to-test-observation-pairs [file]
  (let [id (-> (.getName file) (string/split #"\.") (first))]
    (as-> (imagez/load-image file) $
        (image/resize $ dataset-image-size dataset-image-size)
        (png->observation dataset-datatype false $)
        (vector id $))))

(defn classify-kaggle-tests []
  (let [id-observation-pairs  (map kaggle-png-to-test-observation-pairs (gather-files "resources/test"))
        class-names (classification/get-class-names-from-directory testing-dir)
        observations (mapv second id-observation-pairs)
        class-names (classification/get-class-names-from-directory testing-dir)
        results (infer/infer-n-observations (:network-description
                                             (suite-io/read-nippy-file "trained-network.nippy"))
                                            observations
                                            (ds/create-image-shape dataset-num-channels
                                                                   dataset-image-size
                                                                   dataset-image-size)
                                            dataset-datatype)]
     (mapv (fn [x y] [(first x) (->> (vec y)
                                   (opt/max-index)
                                   (get class-names))])
          id-observation-pairs
          results)))

(defn write-kaggle-results [results]
  (with-open [out-file (io/writer "kaggle-results.csv")]
    (csv/write-csv out-file
                   (into [["id" "label"]]
                         (-> (mapv (fn [[id class]] [(Integer/parseInt id) (if (= "dog" class) 1 0)]) results)
                             (sort))))))

(comment

  (label-one)

  (-> (classify-kaggle-tests)
      (write-kaggle-results))
)
