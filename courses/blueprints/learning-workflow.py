"""

The process proceeds through these six steps in the following order:
1> Acquisition
2> Inspection and exploration
3> Cleaning and preparation
4> Modeling
5> Evaluation
6> Deployment


"""



""" ================== Acquisition =================== """

""" HTTP """

import requests
r = requests.get("https://api.github.com/users/acombs/starred")
print(r.json())

""" --- Pandas --- """

import os
import pandas as pd