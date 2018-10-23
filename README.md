Cooperative Neural Networks - CoNN
=====
Cooperative Neural Networks(CoNN) : Exploiting prior independence structure for improved classification  
(also known as Joint Constraint Networks).  

This work is published in NIPS 2018 [link](https://nips.cc/Conferences/2018/Schedule?showEvent=11409). 

I am working on a [Blog](http://blog.harshshrivastava.com/2018/10/cooperative-neural-networks-conn-an-overview/) to highlight the novelties and main contributions of this work.  

Dependencies
=============

The file 'main.py' is tested on the following 

- Python 3
- Pytorch 0.2.0 
- Numpy 1.15.1 
- nltk (preprocessing text)
- P100 GPUs

Running CoNN-sLDA for Multi-Domain Sentiment Dataset
=============
Getting the data [link](http://www.cs.jhu.edu/~mdredze/datasets/sentiment/):
```
$ wget http://www.cs.jhu.edu/~mdredze/datasets/sentiment/unprocessed.tar.gz
$ tar -xvzf unprocessed.tar.gz
```
Running the script:
```
$ python main.py
```
with the default settings, I got the Area under ROC as (5 fold CV) =  


Development 
============

Code
----

I will be updating the repo with the following additions  

- Script compatible with latest Pytorch version  
- Script for latest Tensorflow version 


Contributing
------------
Issues can be reported at [issues section](https://github.com/Harshs27/CoNN/issues).

If you want to discuss or contribute, please feel free to drop a mail or raise an issue :) 


Collaboration
-------------
I will be happy to discuss and collaborate, if you want to use CoNN or its variant for some other Graphical models!


License
=======
CoNN is released under Apache License. You can read about our license at [here](https://github.com/Harshs27/CoNN/blob/master/LICENSE)
