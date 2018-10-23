CoNN
=====
Cooperative Neural Networks: Exploiting prior independence structure for improved classification  
(also known as Joint Constraint Networks)  

This work is published in NIPS 2018 [link](https://nips.cc/Conferences/2018/Schedule?showEvent=11409) 

I am working on a [Blog](http://blog.harshshrivastava.com/2018/10/cooperative-neural-networks-conn-an-overview/) to highlight the main contributions of this work 

Dependencies
=============

The file '342k\_multisent.py' is tested on the following 

- Python 3
- Pytorch 0.3.0 
- Numpy 

Running the script for Multi-Domain Sentiment Dataset
=============
Getting the data [link](http://www.cs.jhu.edu/~mdredze/datasets/sentiment/):
```
$ wget http://www.cs.jhu.edu/~mdredze/datasets/sentiment/unprocessed.tar.gz
$ tar -xvzf unprocessed.tar.gz
```
Running the script:
```
$ python 342k_multisent.py
```

Development
============

Code
----

I will be shortly updating the code with the following additions  

- Modular form of the code (separating the preprocessing section, model etc.)
- Script compatible with latest Pytorch version  
- Script for latest Tensorflow version


Contributing
------------
Issues can be reported at [issues section](https://github.com/Harshs27/CoNN/issues).

If you want to discuss or contribute, please feel free to raise an issue or mail.

License
=======
CoNN is released under Apache License. You can read about our license at [here](https://github.com/Harshs27/CoNN/blob/master/LICENSE)


