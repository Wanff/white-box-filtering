# white box filtering

white_box is a package that enables activation caching, training probes, and running adversarial attacks on base models and monitors (which can be probes or textclassifiers). it currently supports the memorized text and jailbreak settings. read the notebooks in the experiments folder to see how functions are used. 

We first get activations using scripts/get_activations.py. We have a dataset of jailbreaks available from a different paper called HarmBench. After getting activations, we run some probing experiments which are shown in the experiments folder. 
