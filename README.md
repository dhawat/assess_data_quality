# Introduction 
Data is what guides today's decision making process, it is at the center of modern institutions, but according to the saying : GIGO (Garbage In Garbage Out), bad data may have detrimental consequences on the company that used it. It is then of crucial order that the data be of best quality possible, however, the process of cleaning the data usually relies on deterministic rules, which makes it hard, tedious and time consuming. Thus AMIES along with the company Foyer proposed a challenge about the automation of the process. This lead us to propose algorithms that need as little as possible of human intervention. 
* Fully automated code for the evaluation of data quality. 
* Several strategies to detect "bad data".
We refer the reader to the <url_arxiv_rapport> for more information on the used methods. 
# some useful git command
The folder code will contain the file .py and .ibynb
You can add this repo on your local machine using: git clone <url>
Before any work session use: git pull
this will prevent you from conflict
when your work is done use:
git add ... ,(to add your change)****
git commit -m "write a message explaining your change"
git push , (to add your change online on github).
for notebook please clear the output before push to prevent conflict, and load data.
# project information:

\
Bibliography:
\
https://www.win.tue.nl/~mpechen/publications/pubs/Gama_ACMCS_AdaptationCD_accepted.pdf :
dynamically changing environments, (problem related to the change of data (there distribution)  with time). he start to talk on supervised machine learning on dynamic evolving data (non stationary distribution of the target (to predict) variable)
\
https://mobidev.biz/blog/unsupervised-machine-learning-improve-data-quality Done
\
https://towardsdatascience.com/automated-data-quality-testing-at-scale-using-apache-spark93bb1e2c5cd0  PAGE NOT FOUND
https://www.vldb.org/pvldb/vol11/p1781-schelter.pdf good paper
\
https://arxiv.org/ftp/arxiv/papers/1810/1810.07132.pdf Done
\
https://res.mdpi.com/d_attachment/symmetry/symmetry-10-00248/article_deploy/symmetry-10-00248.pdf not bad specify some algo may be interesting, talk about cleaning data used for classification purspose there basic algo not present in the paper
\
https://arxiv.org/ftp/arxiv/papers/2009/2009.06672.pdf Done (bad paper)
**Important link and ref**
\
redit dataset used by amazon as a test sample:
https://www.kaggle.com/reddit/reddit-comments-may-2015
\
amazon github:
https://github.com/awslabs/python-deequ
\
difference entre pyspark and pandas:
https://www.geeksforgeeks.org/difference-between-spark-dataframe-and-pandas-dataframe/
