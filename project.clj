(defproject kaggle-cats-dogs "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [thinktopic/cortex.suite "0.3.1-SNAPSHOT"]
                 [thinktopic/think.gate "0.1.2"]
                 [org.clojure/clojurescript "1.8.51"]
                 [org.bytedeco.javacpp-presets/cuda "8.0-1.2"]
                 [org.clojure/data.csv "0.1.3"]]

  :plugins [[lein-cljsbuild "1.1.5"]
            [lein-garden "0.3.0"]]


  :garden {:builds [{:id "dev"
                     :source-paths ["src"]
                     :stylesheet css.styles/styles
                     :compiler {:output-to "resources/public/css/app.css"}}]}


  :cljsbuild {:builds
              [{:id "dev"
                :figwheel true
                :source-paths ["cljs"]
                :compiler {:main "kaggle-cats-dogs.classify"
                           :asset-path "out"
                           :output-to "resources/public/js/app.js"
                           :output-dir "resources/public/out"}}]}


  :figwheel {:css-dirs ["resources/public/css"]}
  :main kaggle-cats-dogs.main

)
