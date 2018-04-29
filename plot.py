import matplotlib.pyplot as plt
import numpy as np

one = open('test_results1.txt', 'r')
two = open('test_results2.txt', 'r')
#three = open('test_results2.txt', 'r')

base = [i+1 for i in range(20)]
models = []
top_models = []
cutoff = .9

# One hidden layer
line = one.readline()
for line in one:
  model = []
  include = False

  i = line.index('\t')
  x = line[:i]
  line = line[i+1:]

  i = line.index('\t')
  y = line[:i]
  line = line[i+1:]

  label = x+'x'+y

  i = line.index('\t')
  time = float(line[:i])
  line = line[i+1:]

  for j in range(19):
    i = line.index('\t')
    model.append(float(line[:i]))
    if float(line[:i]) > cutoff:
      include = True
    line = line[i+1:]
  model.append(float(line[:-1]))  

  models.append((label, model, time))
  if include:
    top_models.append((label, model, time))

# Two hidden layer
line = two.readline()
for line in two:
  model = []
  include = False

  i = line.index('\t')
  x = line[:i]
  line = line[i+1:]

  i = line.index('\t')
  y = line[:i]
  line = line[i+1:]

  i = line.index('\t')
  z = line[:i]
  line = line[i+1:]

  label = x+'x'+y+'x'+z

  i = line.index('\t')
  time = float(line[:i])
  line = line[i+1:]

  for j in range(19):
    i = line.index('\t')
    model.append(float(line[:i]))
    if float(line[:i]) > cutoff:
      include = True
    line = line[i+1:]
  model.append(float(line[:-1]))  

  models.append((label, model, time))
  if include:
    top_models.append((label, model, time))
  #plt.plot(base, model, label=labl)
"""
# Three hidden layers
line = three.readline()
for line in three:
  model = []
  include = False

  i = line.index('\t')
  x = line[:i]
  line = line[i+1:]

  i = line.index('\t')
  y = line[:i]
  line = line[i+1:]

  i = line.index('\t')
  z = line[:i]
  line = line[i+1:]

  i = line.index('\t')
  a = line[:i]
  line = line[i+1:]

  label = x+'x'+y+'x'+z+'x'+a

  i = line.index('\t')
  time = float(line[:i])
  line = line[i+1:]

  for j in range(19):
    i = line.index('\t')
    model.append(float(line[:i]))
    if float(line[:i]) > cutoff:
      include = True
    line = line[i+1:]
  model.append(float(line[:-1]))  

  models.append((label, model, time))
  if include:
    top_models.append((label, model, time))
"""

for model in top_models:
  plt.plot(base, model[1], label=model[0])

plt.xlabel('# Epoch Trainings')
plt.ylabel('Accuracy')

plt.title("NN Configuration Accuracy")

#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.legend()
plt.show()
#plt.savefig('foo.png')

fig, ax = plt.subplots()

for model in top_models:
  for epoch, point in enumerate(model[1]):
    if point > cutoff:
      plt.scatter(point, model[2], label=model[0])
      ax.annotate(model[0], (point, model[2]), ha='center')
      #print(model[0], point, model[2], epoch)
#plt.legend()
plt.show() 
  



