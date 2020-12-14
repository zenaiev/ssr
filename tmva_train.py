from __future__ import print_function
import ROOT
import array
import os

'''_g_keep = []
def filterTree(t, cuts = ROOT.TCut(''), events = ROOT.TTree.kMaxEntries, clone = False, title = None, drop_friends=True):
  _g_keep.append(t)
  cuts = ROOT.TCut(cuts)
  utils.fout().cd()
  #obj = ROOT.gDirectory.Get(t.GetName() + '_' + title)
  #print 'obj: ', id(obj)
  #print 'obj: ', obj.IsZombie()
  #print 'obj: ', obj
  #dir0 = t.GetDirectory()
  print 'filterTree cuts = {} to tree {}[{}] in {} ---> '.format(cuts.GetTitle(), t.GetName(), t.GetTitle(), t.GetDirectory().GetName() if t.GetDirectory() else '0'),
  n0 = t.GetEntries()
  t1 = t.CopyTree(cuts.GetTitle(), '')
  #t1.SetDirectory(0)
  if events != TTree.kMaxEntries:
    t1 = t1.CopyTree('', '', events)
    #t1.CopyEntries(t, events)
  #else:
  #  t1.CopyEntries(t)
  n1 = t1.GetEntries()
  print '{} -> {}'.format(n0, n1),
  if clone:
    raise Exception('clone not supported anymore')
    #t1 = t1.CloneTree()
    #t1.SetDirectory(0)
    #print '(cloned)',
  if title:
    t1.SetName(t1.GetName() + '_' + title)
    t1.SetTitle(t1.GetTitle() + '_' + title)
    #print 'saved as {} in {}'.format(t1.GetName() + '_' + title, utils.fout().GetName())
    print 'saved as {} in {}'.format(t1.GetName(), utils.fout().GetName())
  else:
    print ''
  #  print 'WARNING: no title for new tree'
  #ROOT.SetOwnership(t1, 0)
  # need to drop friends to prevent crash in the end
  if drop_friends:
    #print 'dropping 1'
    friends = ROOT.TIter(t1.GetListOfFriends())
    fr = friends()
    #print 'dropping 2'
    while fr:
      #print 'dropping 2.5'
      fr_t = fr.GetTree()
      #print 'removing friend: {}'.format(fr_t.GetName())
      #print 'dropping 3'
      t1.RemoveFriend(fr_t)
      #print 'dropping 4'
      fr = friends()
  #print 'dropping end'
  return t1'''

def do_tmva_plots(tmva_fout_name='TMVA.root', prefix='tmva/'):
  """ produce various TMVA plots """
  tmva_file = os.path.abspath(tmva_fout_name)
  ROOT.TMVA.variables(prefix, tmva_file, 'InputVariables_Deco')
  ROOT.TMVA.correlations(prefix, tmva_file)
  ROOT.TMVA.mvas(prefix, tmva_file, ROOT.TMVA.kCompareType)
  ROOT.TMVA.mvaeffs(prefix, tmva_file)
  ROOT.TMVA.efficiencies(prefix, tmva_file)
  ROOT.TMVA.efficiencies(prefix, tmva_file, 3)
  ROOT.gStyle.Reset() # need to reset style

def train(t_sig, t_bkg, methods = '', tmva_fout_name = 'TMVA.root'):
  outputFile = ROOT.TFile.Open(tmva_fout_name, "RECREATE")
  factory = ROOT.TMVA.Factory("TMVAClassification", outputFile, "!V:!Silent:Color:DrawProgressBar:Transformations=D:AnalysisType=Classification")
  dataloader = ROOT.TMVA.DataLoader('tmva')
  vars = [b.GetName() for b in t_sig.GetListOfBranches()]
  #vars = vars[:2]
  #vars = [v for iv,v in enumerate(vars) if (iv+1)%4 == 0] + vars[-2:]
  vars = [v for v in vars if not v.endswith('_v')]
  print('vars = {}'.format(vars))
  for var in vars:
    dataloader.AddVariable(var)
  dataloader.AddSignalTree(t_sig)
  dataloader.AddBackgroundTree(t_bkg)
  dataloader.PrepareTrainingAndTestTree(ROOT.TCut(''), ROOT.TCut(''), "SplitMode=Random:SplitSeed=100:NormMode=None:!V") # nTrain_Signal=1000:nTrain_Background=1000:

  if 'BDTG' in methods: 
    method = factory.BookMethod(dataloader,ROOT.TMVA.Types.kBDT, "BDTG",
                      "!H:!V:NTrees=1000:MinNodeSize=2.5%:BoostType=Grad:Shrinkage=0.10:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=20:MaxDepth=2")
  # Boosted Decision Trees with adaptive boosting
  if 'BDT' in methods:
    method = factory.BookMethod(dataloader,ROOT.TMVA.Types.kBDT, "BDT",
                      "!V:NTrees=200:MinNodeSize=2.5%:MaxDepth=2:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20" )
  # Multi-Layer Perceptron (Neural Network)
  if 'MLP' in methods:
    method = factory.BookMethod(dataloader, ROOT.TMVA.Types.kMLP, "MLP",
                      "!H:!V:NeuronType=tanh:VarTransform=N:NCycles=100:HiddenLayers=N+5:TestRate=5:!UseRegulator" )
  
  factory.TrainAllMethods()
  factory.TestAllMethods()
  factory.EvaluateAllMethods()
  outputFile.Close()
  do_tmva_plots()

def prepare_reader(prefix, vars=vars, method='BDT'):
  reader = ROOT.TMVA.Reader("!Color:!Silent")
  #vars = reader.DataInfo().GetVariableInfos()
  print('vars = {}'.format(vars))
  vars_local_float = [array.array('f',[0]) for _ in vars] # need to keep memory
  for ivar,var in enumerate(vars):
    reader.AddVariable(var, vars_local_float[ivar])
  reader.BookMVA(method + ' method', prefix + '/weights/TMVAClassification_' + method + '.weights.xml')
  return reader, vars_local_float

#def apply(reader, vars_local_float, t_sig):
#  reader.EvaluateMVA()

def presel(t_sig, t_bkg, filter=True, cuts=None):
  if cuts is None:
    vars = [b.GetName() for b in t_sig.GetListOfBranches()]
    cut_vals = [round(min(getattr(e, var) for e in t_sig)-1e-3, 3) for var in vars]
    print('cuts:\n', '\n'.join(['{} >= {}'.format(var, cut) for var,cut in zip(vars, cut_vals)]))
    cuts = '&&'.join(['{}>={}'.format(var, cut) for var,cut in zip(vars, cut_vals)])
    print('cuts = "{}"'.format(cuts))
  else:
    print('using provided cuts = {}'.format(cuts))
  if filter:
    ftmp = ROOT.TFile.Open('tmp_presel.root', 'recreate')
    ROOT.SetOwnership(ftmp, 0)
    ROOT.SetOwnership(t_bkg, 0)
    n0 = t_bkg.GetEntries()
    t_bkg = t_bkg.CopyTree(cuts)
    n1 = t_bkg.GetEntries()
    print('presel t_bkg: {} -> {}'.format(n0, n1))
    print('presel t_sig: {} -> {}'.format(t_sig.GetEntries(), t_sig.CopyTree(cuts, '').GetEntries()))
    assert t_sig.GetEntries() == t_sig.CopyTree(cuts, '').GetEntries()
    return t_sig, t_bkg

if __name__ == '__main__':
  #fin = ROOT.TFile.Open('trees-all.root')
  fin = ROOT.TFile.Open('trees1.root')
  t_sig = fin.Get('t_sig_xy')
  t_bkg = fin.Get('t_bkg_xy')
  cuts="tsum_max_b>=-72.814&&tsum_max_g>=-73.569&&tsum_max_r>=-74.756&&tsum_max_v>=-73.713&&tasum_max_b>=-85.795&&tasum_max_g>=-87.001&&tasum_max_r>=-89.633&&tasum_max_v>=-87.205&&tsumrat_max_b>=-91.058&&tsumrat_max_g>=-118.928&&tsumrat_max_r>=-103.931&&tsumrat_max_v>=-103.81&&tasumrat_max_b>=-101.502&&tasumrat_max_g>=-125.346&&tasumrat_max_r>=-107.187&&tasumrat_max_v>=-107.66&&bsum_max_b>=-13.42&&bsum_max_g>=-22.162&&bsum_max_r>=-75.695&&bsum_max_v>=-37.7&&basum_max_b>=-113.962&&basum_max_g>=-109.807&&basum_max_r>=-104.678&&basum_max_v>=-108.777&&bsumrat_max_b>=-41.773&&bsumrat_max_g>=-69.185&&bsumrat_max_r>=-103.675&&bsumrat_max_v>=-90.315&&basumrat_max_b>=-109.565&&basumrat_max_g>=-114.301&&basumrat_max_r>=-115.795&&basumrat_max_v>=-104.947&&lsum_b>=-108.251&&lsum_g>=-89.876&&lsum_r>=-102.814&&lsum_v>=-92.293&&lasum_b>=-117.126&&lasum_g>=-115.251&&lasum_r>=-106.939&&lasum_v>=-111.084&&lsumrat_b>=-209.18&&lsumrat_g>=-119.254&&lsumrat_r>=-135.169&&lsumrat_v>=-123.073&&lasumrat_b>=-226.329&&lasumrat_g>=-149.434&&lasumrat_r>=-140.593&&lasumrat_v>=-145.366&&lbmsum_b>=-144.801&&lbmsum_g>=-151.601&&lbmsum_r>=-148.001&&lbmsum_v>=-148.134&&lbmasum_b>=-144.801&&lbmasum_g>=-151.601&&lbmasum_r>=-148.001&&lbmasum_v>=-148.134&&lbmsumrat_b>=-246.43&&lbmsumrat_g>=-249.343&&lbmsumrat_r>=-250.001&&lbmsumrat_v>=-247.992&&lbmasumrat_b>=-300.001&&lbmasumrat_g>=-249.343&&lbmasumrat_r>=-300.001&&lbmasumrat_v>=-247.992&&boxsum_max>=23.545&&marlsum>=-36.657"
  vars = [v.split('=')[0][:-1] for v in cuts.split('&&')]
  vars = filter(lambda x: not x.endswith('_v'), vars)
  print('vars = {}'.format(vars))
  t_sig, t_bkg = presel(t_sig, t_bkg, filter=1, cuts=cuts)
  #aaa
  '''ROOT.TRandom.SetSeed(ROOT.gRandom, 42)
  ROOT.SetOwnership(t_bkg, 0)
  ftmp = ROOT.TFile.Open('tmp.root', 'recreate')
  t_bkg = t_bkg.CopyTree('rndm() < 0.02', '')'''
  # train(t_sig, t_bkg, methods=['BDT'])
  reader,vars_local_float = prepare_reader(prefix='tmva-best', vars=vars)
  for ie,e in enumerate(t_sig):
    for iv,v in enumerate(vars):
      vars_local_float[iv][0] = getattr(t_sig, v)
    bdt = reader.EvaluateMVA('BDT method')
    print('bdt = {}'.format(bdt))
    if ie == 10: break
