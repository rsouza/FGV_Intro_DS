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
  + [What is Data Science](http://proquest.safaribooksonline.com/book/databases/9781449363871)
  + [Data Science process](https://www.amazon.com/Applied-Predictive-Analytics-Principles-Professional/dp/1118727967)
    + [Business Intelligence](https://en.wikipedia.org/wiki/Business_intelligence)
    + [CRISP-DM](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)
  + [Data Science, AI and Machine Learning](https://www.datasciencecentral.com/profiles/blogs/difference-between-machine-learning-data-science-ai-deep-learning)
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
    + Exploratory Data Analysis [1](http://greenteapress.com/thinkstats2/html/index.html), [2](http://people.duke.edu/~ccc14/sta-663-2017/#) [3](oreilly.com/catalog/9780596802363/)
+ Feature Engineering and Selection
+ Model Selection and Evaluation
+ Machine Learning Algorithms [1](http://cdn.intechopen.com/pdfs-wm/10694.pdf), [2](https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/), [3](https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/)
  + Unsupervised
    + Dimensionality reduction
      + PCA
      + SVD
    + [Clustering](https://dataaspirant.com/2016/09/24/classification-clustering-alogrithms/)
      + K-Means
      + Hierarchical Clustering
  + Supervised
    + Linear Regression
    + Logistic Regression
    + SVM
    + Naive Bayes
    + Perceptron
    + Decision Tree
    + Ensemble Models
      + Random Forest
      + Other 
    + [kNN](http://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/)
    + Gradient Boosting algorithms
      + GBM
      + XGBoost
      + LightGBM
      + CatBoost
    + [Neural Networks and Deep Learning](https://www.youtube.com/watch?v=aircAruvnKk&t=0s&index=1&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
      + [Gradient Descent](https://www.youtube.com/watch?v=IHZwWFHWa-w&t=0s&index=2&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
      + [Backpropagation](https://www.youtube.com/watch?v=Ilg3gGewQ5U&t=0s&index=3&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
      + Convolutional Neural Networks
      + Sequence Models
        + Word2vec & others [1](http://www.nilc.icmc.usp.br/nilc/index.php/repositorio-de-word-embeddings-do-nilc)
    + Genetic Algorithms

+ Data Science Tasks
  + NLP and Text Mining
  + Information Retrieval
  + Graphs and Network Analysis
  + Recommender Systems
  + Relational databases and SQL
  + NoSQL Databases
  + Graph Databases
  + Dealing with Big Data!
  + Distributed computing
    + Map Reduce
    + Spark
  + Analytical Pipelines
    + [Scikit-Learn](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
    + [Luigi](https://github.com/spotify/luigi)
    + [Airflow](https://airflow.apache.org/)
