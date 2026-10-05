import numpy as np
import pytest
from scipy.io import savemat
from ppg_cvae.data.dalia import subject_id_from_path, discover_subject_files, load_subject
from ppg_cvae.data.preprocessing import window_segment
from ppg_cvae.data.prepare import prepare_fold, load_prepared


@pytest.mark.parametrize('i', range(1, 16))
def test_complete_subject_ids(tmp_path, i):
    path = tmp_path/f'S{i}.npz'
    np.savez(path, ppg=np.ones(640), hr=np.array([60., 80.]))
    assert subject_id_from_path(path) == f'S{i}'
    assert load_subject(path).subject_id == f'S{i}'


def test_duplicate_subject_files_fail(tmp_path):
    for name in ('S10.npz', 'copy_S10.npz'): np.savez(tmp_path/name, ppg=np.ones(512))
    with pytest.raises(ValueError, match='Duplicate'): discover_subject_files(tmp_path)


def test_nested_mat_schema(tmp_path):
    path = tmp_path/'S15.mat'
    savemat(path, {'signal': {'wrist': {'BVP': np.arange(512), 'ACC': np.ones((256,3))}}, 'label': [60.]})
    record = load_subject(path)
    assert record.ppg.shape == (512,)
    assert record.hr.shape == (1,)
    assert record.acc.shape == (256,3)


def test_window_labels_never_indexed_as_signal_samples():
    rows = window_segment(np.arange(1024), [60, 70, 80, 90, 100])
    assert [r['hr_reference_bpm'] for r in rows] == [60, 70, 80, 90, 100]
    with pytest.raises(ValueError, match='alignment'): window_segment(np.arange(1024), [60, 70])


def synthetic_subjects(root, gap=False, mismatch=False):
    t = np.arange(2048)/64
    for i in range(1,16):
        x = (1 if i>=7 else 100)*np.sin(2*np.pi*t)
        if gap and i == 7: x[800:900] = np.nan
        np.savez(root/f'S{i}.npz', ppg=x, hr=np.full(12 if mismatch and i==1 else 13, 60.))


def test_fold_preparation_no_test_statistics_or_gap_bridging(tmp_path):
    data = tmp_path/'raw'
    data.mkdir()
    synthetic_subjects(data, gap=True)
    manifest = prepare_fold(data, tmp_path/'out', label_start_sample=0, alignment_source='synthetic window labels')
    train_x, train_c = load_prepared(tmp_path/'out', 'train')
    assert np.isfinite(train_x).all()
    assert np.max(np.abs(train_x)) < 3  # val/test amplitude is 100x train
    assert np.all(train_c == 1)
    s7 = manifest[manifest.subject_id=='S7']
    crossing = s7[(s7.start_sample < 900) & (s7.end_sample > 800)]
    assert not crossing.valid.any()
    assert manifest[manifest.subject_id=='S10'].split.eq('train').all()
    with np.load(tmp_path/'out'/'val.npz') as val:
        assert np.max(np.abs(val['x'])) > 50


def test_alignment_count_mismatch_fails(tmp_path):
    synthetic_subjects(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match='13 windows but 12 labels'):
        prepare_fold(tmp_path, tmp_path/'out', label_start_sample=0, alignment_source='synthetic')
