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
  + Data Science, Statistics, AI and Machine Learning [1](https://www.datasciencecentral.com/profiles/blogs/difference-between-machine-learning-data-science-ai-deep-learning), [2](https://towardsdatascience.com/introduction-to-statistics-e9d72d818745), [3](http://proquest.safaribooksonline.com/book/databases/9781449363871), [3](http://cs109.github.io/2015/index.html)
  
  + [Data Science process](https://www.amazon.com/Applied-Predictive-Analytics-Principles-Professional/dp/1118727967)
    + [Business Intelligence](https://en.wikipedia.org/wiki/Business_intelligence)
    + [CRISP-DM](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)
+ Data Science and Visualization Tools
  + Versioning Tools
    + [Git](https://git-scm.com/book/en/v2)
    + [Github](https://guides.github.com/)
    + [Gitlab](https://about.gitlab.com/)
  + Exploratory Data Analysis Tools
    + Jupyter [1](http://jupyter.org/), [2](https://github.com/jupyterlab/jupyterlab)
    + Numpy [1](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html), [2](https://www.datacamp.com/community/tutorials/python-numpy-tutorial)
    + Pandas [1](http://proquest.safaribooksonline.com/9781449323592), [2](http://pandas.pydata.org/pandas-docs/stable/)
      + [Pandas Machine Learning](http://pandas-ml.readthedocs.io/en/stable/)
      + [Geopandas](http://geopandas.org/)
    + [Statsmodels](http://www.statsmodels.org/stable/index.html)
  + Machine Learning Tools
    + [Scikit-Learn](http://scikit-learn.org/stable/)
        + [Inbalanced Learn](http://contrib.scikit-learn.org/imbalanced-learn/stable/#)
        + [ForestCI](https://github.com/scikit-learn-contrib/forest-confidence-interval)
        + [XGBoost](https://github.com/dmlc/xgboost)
        + [Regularized Greedy Forest (RGF)](https://github.com/fukatani/rgf_python)
        + [TPOT](https://github.com/EpistasisLab/tpot)
        + [Auto SKLearn](https://github.com/automl/auto-sklearn)
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
  + Other Tools   
    + [NetworkX](https://networkx.github.io/)
    + [ETE Toolkit](http://etetoolkit.org/)
    + [ODO](https://odo.readthedocs.io/en/latest/)

+ [Data Formats](https://en.wikipedia.org/wiki/Comparison_of_data_serialization_formats)
+ Data Engineering [1](https://medium.freecodecamp.org/the-rise-of-the-data-engineer-91be18f1e603), [2](https://medium.com/@rchang/a-beginners-guide-to-data-engineering-part-i-4227c5c457d7)
  + Data Acquisition
  + Data Preparation
    + [Handling Missing Values](http://www.ritchieng.com/pandas-handling-missing-values/)
    + [Changing Datatypes](http://www.ritchieng.com/pandas-changing-datatype/)
  + Exploratory Data Analysis [1](http://greenteapress.com/thinkstats2/html/index.html), [2](http://people.duke.edu/~ccc14/sta-663-2017/#), [3](oreilly.com/catalog/9780596802363/)
    + [Data Visualization](https://towardsdatascience.com/5-quick-and-easy-data-visualizations-in-python-with-code-a2284bae952f)
+ Feature Engineering and Selection
  + [Numeric Data](https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b)
  + [Discrete/Categorical Data](https://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63)
  + Textual Data [1](https://towardsdatascience.com/understanding-feature-engineering-part-3-traditional-methods-for-text-data-f6f7d70acd41), [2](https://towardsdatascience.com/understanding-feature-engineering-part-4-deep-learning-methods-for-text-data-96c44370bbfa)
+ [Model Selection](https://towardsdatascience.com/data-science-simplified-part-6-model-selection-methods-2511cbdf7cb0)
+ Regularization
+ Bias and Variance
+ Overfitting
+ [Evaluation Metrics](https://towardsdatascience.com/choosing-the-right-metric-for-machine-learning-models-part-1-a99d7d7414e4)

+ Machine Learning Algorithms [1](http://cdn.intechopen.com/pdfs-wm/10694.pdf), [2](https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/), [3](https://towardsdatascience.com/a-tour-of-the-top-10-algorithms-for-machine-learning-newbies-dde4edffae11), [4](https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/)
  + Unsupervised
    + [Dimensionality reduction](https://towardsdatascience.com/reducing-dimensionality-from-dimensionality-reduction-techniques-f658aec24dfe)
      + [PCA](https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60)
      + [SVD](https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/)
      + [t-SNE](https://towardsdatascience.com/checking-out-dimensionality-reduction-with-t-sne-78309b2ca67d) 
    + [Clustering](https://dataaspirant.com/2016/09/24/classification-clustering-alogrithms/)
      + K-Means
      + Hierarchical Clustering
      + [K-Modes](https://github.com/nicodv/kmodes)  
      
  + Supervised
    + Regression [1](https://towardsdatascience.com/5-types-of-regression-and-their-properties-c5e1fa12d55e), [2](https://www.analyticsvidhya.com/blog/2015/08/comprehensive-guide-regression/)  
      + [Linear Regression](https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/)
      + [Logistic Regression](https://www.analyticsvidhya.com/blog/2015/11/beginners-guide-on-logistic-regression-in-r)
      + Polynomial Regression [1](https://towardsdatascience.com/machine-learning-with-python-easy-and-robust-method-to-fit-nonlinear-data-19e8a1ddbd49) [2](http://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions)
      + [Stepwise Regression](https://planspace.org/20150423-forward_selection_with_statsmodels/)
      + [Ridge Regression](https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/)
      + [Lasso Regression](https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/)
      + ElasticNet Regression
    + [SVM](https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/)
    + [Naive Bayes](https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/)
    + [kNN](http://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/)
    + [Decision Tree](https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/)
    + Ensemble Models [1](https://en.wikipedia.org/wiki/Ensemble_learning), [2](https://www.analyticsvidhya.com/blog/2015/08/introduction-ensemble-learning/) [3](https://www.analyticsvidhya.com/blog/2015/09/questions-ensemble-modeling/)
      + [Bagging](https://machinelearningmastery.com/bagging-and-random-forest-ensemble-algorithms-for-machine-learning/)  
        + [Random Forest](https://www.analyticsvidhya.com/blog/2014/06/introduction-random-forest-simplified/)   
        + [Extremely Randomized Trees](https://orbi.uliege.be/bitstream/2268/9357/1/geurts-mlj-advance.pdf) 
        + [Regularized Greedy Forests](https://www.analyticsvidhya.com/blog/2018/02/introductory-guide-regularized-greedy-forests-rgf-python/)
      + Boosting
        + [Adaboost](https://towardsdatascience.com/boosting-algorithm-adaboost-b6737a9ee60c)
        + [GBM](https://towardsdatascience.com/boosting-algorithm-gbm-97737c63daa3)
        + [XGBoost](https://towardsdatascience.com/boosting-algorithm-xgboost-4d9ec0207d)
        + [LightGBM](https://towardsdatascience.com/a-case-for-lightgbm-2d05a53c589c)
        + [CatBoost](https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db)
      + [Stacking](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/)
      + [Voting](https://towardsdatascience.com/ensemble-learning-in-machine-learning-getting-started-4ed85eb38e00)
    + [Perceptron](https://towardsdatascience.com/what-the-hell-is-perceptron-626217814f53)
    + Neural Networks and Deep Learning [1](https://www.youtube.com/watch?v=aircAruvnKk&t=0s&index=1&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) [2](https://towardsdatascience.com/a-weird-introduction-to-deep-learning-7828803693b0)
      + Feedforward Neural Networks
      + Convolutional Neural Networks
      + Sequence Models
        + Word2vec & others [1](http://www.nilc.icmc.usp.br/nilc/index.php/repositorio-de-word-embeddings-do-nilc)
      + [Generative Adversarial Networks](https://www.analyticsvidhya.com/blog/2017/06/introductory-generative-adversarial-networks-gans/)
      + Neural Network concepts
        + [Weight Initialization](https://towardsdatascience.com/deep-learning-best-practices-1-weight-initialization-14e5c0295b94)
        + Gradient Descent [1](https://www.youtube.com/watch?v=IHZwWFHWa-w&t=0s&index=2&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi), [2](https://towardsdatascience.com/gradient-descent-algorithm-and-its-variants-10f652806a3)
        + [Backpropagation](https://www.youtube.com/watch?v=Ilg3gGewQ5U&t=0s&index=3&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

    + [Genetic Algorithms](https://www.analyticsvidhya.com/blog/2017/07/introduction-to-genetic-algorithm/)


+ Data Science Tasks
  + NLP and Text Mining
  + Information Retrieval
  + Graphs and Network Analysis
  + Sentiment Analysis [1](https://towardsdatascience.com/another-twitter-sentiment-analysis-bb5b01ebad90), [2](https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-2-333514854913), [3](https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-3-zipfs-law-data-visualisation-fc9eadda71e7), [4](https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-4-count-vectorizer-b3f4944e51b5), [5](https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-5-50b4e87d9bdd)
  + Recommender Systems
  + Relational databases and SQL
  + NoSQL Databases
  + Graph Databases
  + Big Data and Distributed computing
    + Map Reduce
    + [Spark](https://towardsdatascience.com/deploy-a-python-model-more-efficiently-over-spark-497fc03e0a8d)
  + Analytical Pipelines
    + [Scikit-Learn](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
    + [Luigi](https://github.com/spotify/luigi)
    + [Airflow](https://airflow.apache.org/)
