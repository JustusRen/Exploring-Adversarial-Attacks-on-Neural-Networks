import json

with open('dlfuzz_normal.json') as d:
    normal_dict = json.load(d)
    print(normal_dict)
with open('dlfuzz_random.json') as d:
    random_dict = json.load(d)
    print(random_dict)
    

import matplotlib.pyplot as plt
plt.style.use(['science', 'ieee', 'no-latex'])

#plt.plot(normal_dict["0.25"], label="Pert. Strength 25%")
#plt.plot(normal_dict["0.5"], label="Pert. Strength 50%")
plt.plot(normal_dict["1"], label="Calculated Pert. Strength 100%")
#plt.plot(normal_dict["2"], label="Pert. Strength 200%")
#plt.plot(normal_dict["4"], label="Pert. Strength 400%")

#plt.plot(random_dict["0.25"], label="Random Pert. 25%%")
#plt.plot(random_dict["0.5"], label="Random Pert. 50% ")
plt.plot(random_dict["1"], label="Random Pert. Strength 100%") 
#plt.plot(random_dict["2"], label="Random Pert. 200%")
#plt.plot(random_dict["4"], label="Random Pert. 400%")

plt.xlabel("Steps")
plt.ylabel("Neuron Coverage")
plt.xlim(0, 19)

plt.legend()
plt.savefig("NeuronCoverage.pdf")


