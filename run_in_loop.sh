#!/bin/bash

echo "python main.py --year 2016 --chains 2 --hist_dist_std .014 --iterations 2000"
python main.py --year 2016 --chains 2 --hist_dist_std .014 --iterations 2000

echo "python main.py --year 2008 --chains 2 --hist_dist_std 0.2236 --iterations 2000"
python main.py --year 2008 --chains 2 --hist_dist_std 0.2236 --iterations 2000

echo "python main.py --year 2016 --chains 2 --hist_dist_std 0.2236 --iterations 2000"
python main.py --year 2016 --chains 2 --hist_dist_std 0.2236 --iterations 2000
