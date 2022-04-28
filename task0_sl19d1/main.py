import csv
import numpy as np
from sklearn.metrics import mean_squared_error


submission = open('submission.csv', 'w', newline='\n')
writer = csv.writer(submission)
writer.writerow(["Id", "y"])

csvfile = open('test.csv', newline='\n')
csv_reader = csv.reader(csvfile)
next(csv_reader)

for row in csv_reader:
    #print("row =", row)
    x = np.array(row[1:11])
    #print("x= ",x)
    y = np.mean(x.astype(float))
    #print('my solution= ', y, '\n ground truth= ', row[1])
    writer.writerow([row[0], y])
""" RMSE = mean_squared_error(row[1], y) ** 0.5
    print(RMSE)"""