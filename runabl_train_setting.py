import subprocess 
import os 

os.chdir('./')
ST = 'python '

stand = dict()
conf = dict()

stand = dict()
stand['ds'] = 'cifar10' 
stand['bs'] = 128
stand['defense'] = 'pgd_train'
stand['model'] = 'resnet18'
stand['epsilon'] = 0.031
stand['trades_beta'] = 1.0
stand['feat_dim'] = -1

conf['rice_2020'] = stand.copy()
conf['rice_2020']['epochs'] = 200 
conf['rice_2020']['inf'] = 'rice_2020_rancrop'

conf['pang_2020'] = stand.copy()
conf['pang_2020']['epochs'] = 110
conf['pang_2020']['inf'] = 'pang_2020_rancrop'

conf['zhange_2019'] = stand.copy()
conf['zhange_2019']['epochs'] = 105
conf['zhange_2019']['inf'] = 'zhange_2019_rancrop'

conf['zhange_2020'] = stand.copy()
conf['zhange_2020']['epochs'] = 120
conf['zhange_2020']['inf'] = 'zhange_2020_rancrop'

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