(ns kaggle-cats-dogs.main
  (:gen-class))

(defn -main
  [& args]
  (require 'kaggle-cats-dogs.core)
  ((resolve 'kaggle-cats-dogs.core/train-forever-uberjar)))

