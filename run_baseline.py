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
stand['lccomw'] = 1.0
stand['lcsmtw'] = 1.0
stand['gbcomw'] = 1.0
stand['gbsmtw'] = 0.0
stand['confw'] = 1.0
stand['feat_dim'] = -1
stand['alpha'] = 1.0
stand['eval_multi'] = True 
stand['eval_multi_auto'] = True 

conf['pgd_train'] = stand.copy()
conf['pgd_train']['defense'] = 'pgd_train'
conf['pgd_train']['alpha'] = 0

conf['adr_pgd'] = stand.copy()
conf['adr_pgd']['defense'] = 'adr_pgd'
conf['adr_pgd']['alpha'] = 0

conf['trades_train'] = stand.copy()
conf['trades_train']['defense'] = 'trades_train'
conf['trades_train']['trades_beta'] = 6.0

conf['adr_trades'] = stand.copy()
conf['adr_trades']['defense'] = 'adr_trades'
conf['adr_trades']['trades_beta'] = 6.0

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