# FGV_Intro_DS
Introduction to Data Science @ FGV

Instructor: [Renato Rocha Souza](http://emap.fgv.br/corpo-docente/renato-rocha-souza)

This is the repository of code for the "Introduction to Data Science"

This class is about the Data Science process, in which we seek to gain useful predictions and insights from data.
Through real-world examples and code snippets, we introduce methods for:

+ data munging, scraping, sampling andcleaning in order to get an informative, manageable data set;
+ data storage and management in order to be able to access data (even if big data);
+ exploratory data analysis (EDA) to generate hypotheses and intuition about the data;
+ prediction based on statistical learning tools;
+ communication of results through visualization, stories, and interpretable summaries

Detailed Syllabus:

+ [Data Science Concepts and Methodologies](https://docs.google.com/presentation/d/1ysQroWAcUJBizt00v7q-Ss1lalJlojZBlRInLQTDJV8/edit?usp=sharing)
  + Data Science, Statistics, AI and Machine Learning [ref1](https://www.datasciencecentral.com/profiles/blogs/difference-between-machine-learning-data-science-ai-deep-learning), [ref2](https://towardsdatascience.com/introduction-to-statistics-e9d72d818745), [ref3](http://proquest.safaribooksonline.com/book/databases/9781449363871), [ref4](http://cs109.github.io/2015/index.html)
  + [Data Science process](https://www.amazon.com/Applied-Predictive-Analytics-Principles-Professional/dp/1118727967)
    + [Business Intelligence](https://en.wikipedia.org/wiki/Business_intelligence)
    + [CRISP-DM](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)
+ [Data Formats](https://en.wikipedia.org/wiki/Comparison_of_data_serialization_formats)
+ Data Engineering [ref1](https://medium.freecodecamp.org/the-rise-of-the-data-engineer-91be18f1e603), [ref2](https://medium.com/@rchang/a-beginners-guide-to-data-engineering-part-i-4227c5c457d7)
  + [Data Acquisition](http://metah.ch/blog/2014/09/introduction-to-machine-learning-from-data-acquisition-to-a-production-service-2/)
  + Data Preparation
    + [Handling Missing Values](http://www.ritchieng.com/pandas-handling-missing-values/)
    + [Changing Datatypes](http://www.ritchieng.com/pandas-changing-datatype/)
  + Exploratory Data Analysis [ref1](http://greenteapress.com/thinkstats2/html/index.html), [ref2](http://people.duke.edu/~ccc14/sta-663-2017/#), [ref3](oreilly.com/catalog/9780596802363/)  
  + Data Visualization [ref1](https://towardsdatascience.com/5-quick-and-easy-data-visualizations-in-python-with-code-a2284bae952f), [ref2](https://www.linuxlinks.com/best-free-python-visualization-packages/)  
+ Feature Engineering
+ [Feature Selection](http://scikit-learn.org/stable/modules/feature_selection.html)
  + [Numeric Data](https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b)
  + [Discrete/Categorical Data](https://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63)
  + Textual Data [ref1](https://towardsdatascience.com/understanding-feature-engineering-part-3-traditional-methods-for-text-data-f6f7d70acd41), [ref2](https://towardsdatascience.com/understanding-feature-engineering-part-4-deep-learning-methods-for-text-data-96c44370bbfa)
+ [Model Selection](https://towardsdatascience.com/data-science-simplified-part-6-model-selection-methods-2511cbdf7cb0)
+ Cross Validation [ref1](https://www.analyticsvidhya.com/blog/2018/05/improve-model-performance-cross-validation-in-python-r/), [video1](https://www.coursera.org/learn/deep-neural-network/lecture/cxG1s/train-dev-test-sets)
+ Oversampling and Undersampling [ref1](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume16/chawla02a-html/)
+ Regularization [ref1](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a), [ref2](https://www.analyticsvidhya.com/blog/2015/02/avoid-over-fitting-regularization/)
+ [Bias and Variance](https://towardsdatascience.com/balancing-bias-and-variance-to-control-errors-in-machine-learning-16ced95724db)
+ [Overfitting and Underfitting](https://towardsdatascience.com/overfitting-vs-underfitting-a-conceptual-explanation-d94ee20ca7f9)
+ Evaluation Metrics [ref1](https://towardsdatascience.com/choosing-the-right-metric-for-machine-learning-models-part-1-a99d7d7414e4), [ref2](https://towardsdatascience.com/choosing-the-right-metric-for-evaluating-machine-learning-models-part-2-86d5649a5428)

+ Machine Learning Algorithms [ref1](http://cdn.intechopen.com/pdfs-wm/10694.pdf), [ref2](https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/), [ref3](https://towardsdatascience.com/a-tour-of-the-top-10-algorithms-for-machine-learning-newbies-dde4edffae11), [ref4](https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/), [ref5](https://blogs.sas.com/content/subconsciousmusings/2017/04/12/machine-learning-algorithm-use/)
  + Unsupervised
    + [Dimensionality reduction](https://towardsdatascience.com/reducing-dimensionality-from-dimensionality-reduction-techniques-f658aec24dfe)
      + [PCA](https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60)
      + [SVD](https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/)
      + [t-SNE](https://towardsdatascience.com/checking-out-dimensionality-reduction-with-t-sne-78309b2ca67d)
    + Clustering [ref1](https://www.analyticsvidhya.com/blog/2016/11/an-introduction-to-clustering-and-different-methods-of-clustering/), [ref2](https://dataaspirant.com/2016/09/24/classification-clustering-alogrithms/), [ref3](https://www.analyticsvidhya.com/blog/2013/11/getting-clustering-right/)  
      + [K-Means](https://towardsdatascience.com/clustering-using-k-means-algorithm-81da00f156f6)  
      + [Hierarchical Clustering](https://towardsdatascience.com/clustering-unsupervised-learning-788b215b074b)  
      + [K-Modes](https://github.com/nicodv/kmodes)  
    + Topic Modeling [ref1](https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730), [ref2](https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/)
    + [Unsupervised Deep Learning](https://www.analyticsvidhya.com/blog/2018/05/essentials-of-deep-learning-trudging-into-unsupervised-deep-learning/)

  + Supervised
    + Regression [ref1](https://www.analyticsvidhya.com/blog/2015/10/regression-python-beginners/), [ref2](https://towardsdatascience.com/my-journey-into-machine-learning-class-5-regression-cb6f04006b29), [ref3](https://towardsdatascience.com/5-types-of-regression-and-their-properties-c5e1fa12d55e), [ref4](https://www.analyticsvidhya.com/blog/2015/08/comprehensive-guide-regression/), [ref5](https://towardsdatascience.com/selecting-the-best-machine-learning-algorithm-for-your-regression-problem-20c330bad4ef)  
    + Classification [ref1](https://medium.com/@sifium/machine-learning-types-of-classification-9497bd4f2e14), [ref2](https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/), [ref3](https://towardsdatascience.com/building-a-deployable-ml-classifier-in-python-46ba55e1d720)
    + Linear Models
      + [Linear Regression](https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/)
        + Polynomial Regression [ref1](https://towardsdatascience.com/machine-learning-with-python-easy-and-robust-method-to-fit-nonlinear-data-19e8a1ddbd49) [ref2](http://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions)
        + [Stepwise Regression](https://planspace.org/20150423-forward_selection_with_statsmodels/)
        + [Ridge Regression](https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/)
        + [Lasso Regression](https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/)
        + ElasticNet Regression
      + Logistic Regression (Categorical) [ref1](https://en.wikipedia.org/wiki/Logistic_regression), [ref2](https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8),  [ref3](https://www.analyticsvidhya.com/blog/2015/11/beginners-guide-on-logistic-regression-in-r)

      + Support Vector Machines [ref1](https://towardsdatascience.com/i-support-vector-machines-and-so-should-you-7af122b6748), [ref2](http://users.ecs.soton.ac.uk/srg/publications/pdf/SVM.pdf), [ref3](https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/), [video](https://www.youtube.com/watch?v=_PwhiWxHK8o)  
        + SVR [ref1](http://kernelsvm.tripod.com/), [ref2](https://alex.smola.org/papers/2003/SmoSch03b.pdf)    
        + SVC [ref1](http://www.robots.ox.ac.uk/~az/lectures/ml/lect2.pdf), [ref2](https://www.sciencedirect.com/science/article/pii/S0925231203003758)  
      + [Passive-Aggressive](https://www.bonaccorso.eu/2017/10/06/ml-algorithms-addendum-passive-aggressive-algorithms/)
      + [Perceptron](https://towardsdatascience.com/what-the-hell-is-perceptron-626217814f53)
    + Bayesian Models
      + [Naive Bayes](https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/)
      + [Gaussian Naive Bayes](http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf)
      + [Bernoulli Naive Bayes](http://mattshomepage.com/articles/2016/Jun/07/bernoulli_nb/)
    + [Guassian Mixtures](http://katbailey.github.io/post/gaussian-processes-for-dummies/) [package](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process)
    + [k Nearest Neighbors (kNN)](http://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/)
    + Decision Tree [ref1](https://towardsdatascience.com/random-forest-mystery-revealed-69ca18b82ff5), [ref2](https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/)
    + Ensemble Models [ref1](https://en.wikipedia.org/wiki/Ensemble_learning), [ref2](https://www.analyticsvidhya.com/blog/2015/08/introduction-ensemble-learning/) [ref3](https://www.analyticsvidhya.com/blog/2015/09/questions-ensemble-modeling/), [package](http://scikit-learn.org/stable/modules/ensemble.html)
      + [Bagging](https://machinelearningmastery.com/bagging-and-random-forest-ensemble-algorithms-for-machine-learning/)  
        + [Random Forest](https://www.analyticsvidhya.com/blog/2014/06/introduction-random-forest-simplified/)   
        + [Extremely Randomized Trees](https://orbi.uliege.be/bitstream/2268/9357/1/geurts-mlj-advance.pdf)
        + [Bagging Classifier](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume11/opitz99a-html/node3.html), [package](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)
      + [Boosting](https://en.wikipedia.org/wiki/Boosting_(machine_learning))
        + [Adaboost](https://towardsdatascience.com/boosting-algorithm-adaboost-b6737a9ee60c)
        + [GBM](https://towardsdatascience.com/boosting-algorithm-gbm-97737c63daa3)
        + [XGBoost](https://towardsdatascience.com/boosting-algorithm-xgboost-4d9ec0207d), [package](https://github.com/dmlc/xgboost)
        + [LightGBM](https://towardsdatascience.com/a-case-for-lightgbm-2d05a53c589c)
        + [CatBoost](https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db)
        + [Regularized Greedy Forests](https://www.analyticsvidhya.com/blog/2018/02/introductory-guide-regularized-greedy-forests-rgf-python/), [package](https://github.com/fukatani/rgf_python)
      + [Stacking](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/)
      + [Voting](https://towardsdatascience.com/ensemble-learning-in-machine-learning-getting-started-4ed85eb38e00)

    + [Genetic Algorithms](https://www.analyticsvidhya.com/blog/2017/07/introduction-to-genetic-algorithm/)  
        + [TPOT](https://github.com/EpistasisLab/tpot),
        + [Auto SKLearn](https://github.com/automl/auto-sklearn)  

    + Neural Networks and Deep Learning [ref1](http://neuralnetworksanddeeplearning.com/chap1.html), [ref2](https://towardsdatascience.com/neural-network-architectures-156e5bad51ba), [ref3](https://towardsdatascience.com/a-weird-introduction-to-deep-learning-7828803693b0),  [ref4](https://ujjwalkarn.me/2016/08/09/quick-intro-neural-networks/), [ref5](https://www.analyticsvidhya.com/blog/2018/05/deep-learning-faq/), [video](https://www.youtube.com/watch?v=aircAruvnKk&t=0s&index=1&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi), [meme](https://indico.io/blog/wp-content/uploads/2016/02/inception_meme.jpg)
      + Convolutional Neural Networks [ref1](https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/), [ref2](https://www.saama.com/blog/different-kinds-convolutional-filters/), [ref3](https://software.intel.com/en-us/articles/hands-on-ai-part-16-modern-deep-neural-network-architectures-for-image-classification)
        + [Convolution for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)
      + Sequence Models
        + Word2vec [ref1](https://towardsdatascience.com/using-word2vec-for-better-embeddings-of-categorical-features-de75020e1233), [trained-models](http://www.nilc.icmc.usp.br/nilc/index.php/repositorio-de-word-embeddings-do-nilc)
      + [Generative Adversarial Networks](https://www.analyticsvidhya.com/blog/2017/06/introductory-generative-adversarial-networks-gans/)
      + Neural Network concepts
        + [Weight Initialization](https://towardsdatascience.com/deep-learning-best-practices-1-weight-initialization-14e5c0295b94)
        + [Weight Averaging](https://towardsdatascience.com/stochastic-weight-averaging-a-new-way-to-get-state-of-the-art-results-in-deep-learning-c639ccf36a)
        + Gradient Descent [ref1](https://www.youtube.com/watch?v=IHZwWFHWa-w&t=0s&index=2&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi), [ref2](https://towardsdatascience.com/gradient-descent-algorithm-and-its-variants-10f652806a3)
        + [Backpropagation](https://www.youtube.com/watch?v=Ilg3gGewQ5U&t=0s&index=3&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

        + [Neural Network Visualization](https://playground.tensorflow.org/)


+ Data Science Tasks
  + Time Series Analysis [ref1](https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/), [ref2](https://www.analyticsvidhya.com/blog/2015/12/complete-tutorial-time-series-modeling/), [ref3](http://www.blackarbs.com/blog/time-series-analysis-in-python-linear-models-to-garch/11/1/2016)
  + NLP and Text Mining
  + Information Retrieval
  + Graphs and Network Analysis
  + Sentiment Analysis [ref1](https://towardsdatascience.com/another-twitter-sentiment-analysis-bb5b01ebad90), [ref2](https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-2-333514854913), [ref3](https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-3-zipfs-law-data-visualisation-fc9eadda71e7), [ref4](https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-4-count-vectorizer-b3f4944e51b5), [ref5](https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-5-50b4e87d9bdd)
  + Recommender Systems

+ Data Science and Visualization Tools
  + Versioning Tools
    + [Git](https://git-scm.com/book/en/v2)
    + [Github](https://guides.github.com/)
    + [Gitlab](https://about.gitlab.com/)
  + Exploratory Data Analysis Tools
    + Jupyter [ref1](http://jupyter.org/), [ref2](https://github.com/jupyterlab/jupyterlab) [ref3](https://www.analyticsvidhya.com/blog/2018/05/starters-guide-jupyter-notebook/)
    + Numpy [ref1](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html), [ref2](https://www.datacamp.com/community/tutorials/python-numpy-tutorial)
    + Pandas [ref1](http://proquest.safaribooksonline.com/9781449323592), [ref2](http://pandas.pydata.org/pandas-docs/stable/)
      + [Pandas Machine Learning](http://pandas-ml.readthedocs.io/en/stable/)
      + [Geopandas](http://geopandas.org/)
    + [Statsmodels](http://www.statsmodels.org/stable/index.html)
  + Machine Learning Tools
    + [Scikit-Learn](http://scikit-learn.org/stable/)
        + [Inbalanced Learn](http://contrib.scikit-learn.org/imbalanced-learn/stable/#)
        + [ForestCI](https://github.com/scikit-learn-contrib/forest-confidence-interval)
    + [Tensor Flow](https://www.tensorflow.org/)
    + [Keras](https://keras.io/)
    + [Gensim](https://radimrehurek.com/gensim/)  
    + [Orange](https://orange.biolab.si/)
  + NLP Tools
    + [NLTK](https://www.nltk.org/)
    + [Spacy](https://spacy.io/)
    + [TextBlob](http://textblob.readthedocs.io/en/dev/)  
  + Visualization Tools
    + [Matplotlib](https://matplotlib.org/)
    + [Seaborn](https://seaborn.pydata.org/)
    + [Bokeh](https://bokeh.pydata.org/en/latest/)
    + [Plotly](https://plot.ly/)
    + [Altair](https://altair-viz.github.io/)
    + [GGPlot2](http://ggplot.yhathq.com/)
    + [MPLD3](http://mpld3.github.io/)
      + [d3.js](https://d3js.org/)
    + [HoloViews](http://holoviews.org/)
    + [Folium](http://python-visualization.github.io/folium/)
      + [Leaflet](http://leafletjs.com/)
  + Big Data and Distributed computing
    + Map Reduce
    + [Spark](https://towardsdatascience.com/deploy-a-python-model-more-efficiently-over-spark-497fc03e0a8d)
  + Analytical Pipelines
    + [Scikit-Learn](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
    + [Luigi](https://github.com/spotify/luigi)
    + [Airflow](https://airflow.apache.org/)
  + Other Tools   
    + [NetworkX](https://networkx.github.io/)
    + [ETE Toolkit](http://etetoolkit.org/)
    + [ODO](https://odo.readthedocs.io/en/latest/)
  + Relational databases and SQL
  + NoSQL Databases
  + Graph Databases
