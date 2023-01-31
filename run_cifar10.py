import subprocess 
import os 

os.chdir('./')
ST = 'python '

stand = dict()
conf = dict()

stand = dict()
stand['ds'] = 'cifar10' 
stand['bs'] = 128
stand['defense'] = 'ascl_pgd_2trans'
stand['model'] = 'resnet18'
stand['epsilon'] = 0.031
stand['trades_beta'] = 1.0
stand['lccomw'] = 0.0
stand['lcsmtw'] = 2.0
stand['gbcomw'] = 1.0
stand['gbsmtw'] = 0.0
stand['confw'] = 0.0
stand['feat_dim'] = 128
stand['eval_linear'] = False
stand['alpha'] = 0.0
stand['combine_type'] = 1
stand['dist'] = 'linf'
stand['hidden_norm'] = True
stand['tau'] = 0.2
stand['inf'] = 'none'

conf['resnet18'] = stand.copy()
conf['resnet18']['model'] = 'resnet18'


conf['wideresnet'] = stand.copy()
conf['wideresnet']['model'] = 'wideresnet'


skip = ['_', '_', '_', '_']

progs = [
	'02a_adversarial_training.py ',
    '02e_evaluate_robustness.py ',
]

for k in list(conf.keys()):
	if k in skip: 
		continue

	for chST in progs: 

		exp = conf[k]
		sub = ' '.join(['--{}={}'.format(t, exp[t]) for t in exp.keys()])
		print(sub)
		subprocess.call([ST + chST + sub], shell=True)