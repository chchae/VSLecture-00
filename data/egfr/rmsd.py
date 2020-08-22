from rdkit.Chem.rdmolfiles import MaeMolSupplier
from rdkit.Chem import AllChem, MolToSmiles, rdmolfiles, rdchem
from rdkit.Chem import ChemicalFeatures
import numpy as np
import gzip


def func_rmsd_subr( refm, movm, idx ) :
	d2sum = 0
	for i in range( len(idx) ) :
		a1 = refm.GetConformer().GetAtomPosition(i)
		a2 = movm.GetConformer().GetAtomPosition( idx[i] )
		d = np.linalg.norm( a1 - a2 )
		d2sum += d * d
	return np.sqrt( d2sum / (len(idx) ) )


def func_rmsd( refm, m ) :
	hts = list( m.GetSubstructMatch( refm) )
	if not hts:
		return 0, 0
	rmsd = func_rmsd_subr( refm, m, hts )
#	print( rmsd, hts, MolToSmiles(mr), MolToSmiles(mm) )
	return rmsd, len(hts)


def func() :
	refms = rdmolfiles.MaeMolSupplier( gzip.open( 'core.maegz' ) )
	docks = rdmolfiles.MaeMolSupplier( gzip.open( 'dock-diverse.maegz' ) )

	count = 0
	for mr in refms:
		for mm in docks:
			d, cnt = func_rmsd( mr, mm )
			print( "%4d : %.2f  %2d" % ( count, d, cnt ) )
			count = count + 1


func()

