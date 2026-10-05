"""Fixed subject-wise folds from the corrected proposal."""
from dataclasses import dataclass
from typing import Dict, FrozenSet

GROUPS: Dict[str, FrozenSet[str]] = {
    "G1": frozenset({"S1", "S2", "S3"}),
    "G2": frozenset({"S4", "S5", "S6"}),
    "G3": frozenset({"S7", "S8", "S9"}),
    "G4": frozenset({"S10", "S11", "S12"}),
    "G5": frozenset({"S13", "S14", "S15"}),
}

@dataclass(frozen=True)
class FoldSubjects:
    fold_id: int
    train_subjects: FrozenSet[str]
    val_subjects: FrozenSet[str]
    test_subjects: FrozenSet[str]

_FOLD_GROUPS = {1:("G3|G4|G5","G2","G1"),2:("G1|G4|G5","G3","G2"),3:("G1|G2|G5","G4","G3"),4:("G1|G2|G3","G5","G4"),5:("G2|G3|G4","G1","G5")}
def _expand(spec: str) -> FrozenSet[str]:
    out=set()
    for g in spec.split('|'): out.update(GROUPS[g])
    return frozenset(out)

def get_fold(fold_id: int) -> FoldSubjects:
    if fold_id not in _FOLD_GROUPS: raise ValueError("fold_id must be one of 1..5")
    train,val,test=_FOLD_GROUPS[fold_id]
    f=FoldSubjects(fold_id,_expand(train),_expand(val),_expand(test))
    if f.train_subjects & f.val_subjects or f.train_subjects & f.test_subjects or f.val_subjects & f.test_subjects:
        raise RuntimeError(f"Subject leakage in fold {fold_id}")
    return f

def all_folds() -> Dict[int,FoldSubjects]: return {i:get_fold(i) for i in range(1,6)}
