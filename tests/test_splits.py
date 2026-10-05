from ppg_cvae.data.splits import all_folds
def test_no_leakage():
 for f in all_folds().values():
  assert not f.train_subjects&f.val_subjects; assert not f.train_subjects&f.test_subjects; assert not f.val_subjects&f.test_subjects
