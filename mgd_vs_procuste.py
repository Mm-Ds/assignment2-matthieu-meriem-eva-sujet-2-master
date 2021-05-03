from language import Language
from translator import Translator
from translator import SupervisedTranslator
import pandas as pd
import matplotlib.pyplot as plt

print("Embeddings uploading")
en = Language("english", "wiki.en.vec")
fr = Language("french", "wiki.fr.vec")


print("\nDictionnaries uploading")
fr_en_dic = pd.read_csv('fr-en.0-5000.txt', sep=" ", header=None, names=["french", "english"], na_filter= False)
fr_en_test_dic = pd.read_csv('fr-en.5000-6500.txt', sep=" ", header=None, names=["french", "english"], na_filter= False)

en_fr_dic = pd.read_csv('en-fr.0-5000.txt', sep=" ", header=None, names=["english", "french"], na_filter= False)
en_fr_test_dic = pd.read_csv('en-fr.5000-6500.txt', sep=" ", header=None, names=["english", "french"], na_filter= False)

print("Procrustes")
stats = {}
for i in range(200, 8201, 400):
	print(i)
	fr_en_trslt = SupervisedTranslator(fr, en, fr_en_dic.loc[:i,:])
	fr_en_trslt.fit(method="procrustes")
	stats[len(fr_en_trslt.dictionary)] = fr_en_trslt.evaluate(fr_en_test_dic, [1, 5, 10])

print("MGD")
stats2 = {}
for i in range(200, 8201, 400):
	print(i)
	fr_en_trslt = SupervisedTranslator(fr, en, fr_en_dic.loc[:i,:])
	fr_en_trslt.fit(method="MGD", step_size=0.1, batch_size=4, epochs=60, verbose=0)
	stats2[len(fr_en_trslt.dictionary)] = fr_en_trslt.evaluate(fr_en_test_dic, [1, 5, 10])


colors = {1:"blue", 5:"green", 10:"red"}
x = list(stats.keys())
y = {}
y2 = {}
for i in stats.keys():
	for j in stats[i].keys():
		if j not in y.keys():
			y[j] = []
			y2[j] = []
		y[j].append(stats[i][j])
		y2[j].append(stats2[i][j])

for i in y.keys():
	plt.plot(x, y2[i], "--", c=colors[i], label=str("MGD : p@"+str(i)))
	plt.plot(x, y[i], c=colors[i], label=str("Procrustes : p@"+str(i)))
plt.xlabel("| D |")
plt.ylabel("Accuracy")
plt.legend();
plt.savefig("MGD_vs_Procrutes.jpg")